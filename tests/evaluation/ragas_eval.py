#!/usr/bin/env python3
"""
RAGAS Evaluation Suite for RenewIQ Agent
-----------------------------------------
Evaluates the deployed agent against the 20-question eval dataset using:
  - Faithfulness       > 0.85  (answer supported by retrieved context)
  - Answer Relevancy   > 0.85  (answer addresses the question)
  - Context Precision  > 0.80  (retrieved clauses are relevant)

Usage:
  # Against local graph (mock mode):
  RENEWIQ_USE_MOCK_ENDPOINT=true RENEWIQ_RERANKER_IDENTITY=true \
    python tests/evaluation/ragas_eval.py --fail-below 0.85

  # Against Databricks endpoint:
  python tests/evaluation/ragas_eval.py \
    --endpoint https://<host>/serving-endpoints/renewiq-agent-endpoint/invocations \
    --token <token> --fail-below 0.85

Exit code 0 = all metrics above threshold. Exit code 1 = gate failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent / "agent_eval_dataset.json"
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
}


def load_eval_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def run_agent_local(question: str, context: Optional[str]) -> dict:
    """Run the local LangGraph graph for a single eval question."""
    from src.agents.orchestrator.graph import get_graph
    from src.agents.orchestrator.state import make_initial_state
    import uuid

    state = make_initial_state(
        query=question,
        session_id=str(uuid.uuid4()),
        contract_ids=[context] if context else None,
    )
    result = get_graph().invoke(state)
    report = result.get("report", {})
    return {
        "answer": report.get("narrative", ""),
        "contexts": [c.get("chunk_text", "") for c in result.get("retrieved_clauses", [])],
        "risk_flags": report.get("risk_flags", []),
    }


def run_agent_endpoint(question: str, context: Optional[str], endpoint_url: str, token: str) -> dict:
    """Call a Databricks Model Serving endpoint for a single eval question."""
    import httpx

    payload = {
        "messages": [{"role": "user", "content": question}],
        "contracts": [context] if context else [],
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    with httpx.Client(timeout=120) as client:
        resp = client.post(endpoint_url, json=payload, headers=headers)
        resp.raise_for_status()

    data = resp.json()
    return {
        "answer": data.get("response", ""),
        "contexts": [],
        "risk_flags": data.get("risk_flags", []),
    }


def compute_ragas_metrics(samples: list[dict]) -> dict[str, float]:
    """
    Compute RAGAS metrics using the ragas library.
    Falls back to heuristic metrics when ragas is not installed (CI without LLM).
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        ds = Dataset.from_list([
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"] or ["No context retrieved."],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ])

        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
        }

    except ImportError:
        logger.warning("ragas not installed — using heuristic metrics (install ragas[all] for real eval)")
        return _heuristic_metrics(samples)


def _heuristic_metrics(samples: list[dict]) -> dict[str, float]:
    """
    Heuristic approximation of RAGAS metrics for CI without LLM access.
    Based on keyword overlap between answer and ground truth / context.
    """
    import re

    def token_overlap(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"\w+", a.lower()))
        b_tokens = set(re.findall(r"\w+", b.lower()))
        if not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(b_tokens)

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    for s in samples:
        answer = s["answer"]
        ground_truth = s["ground_truth"]
        contexts = s["contexts"]
        question = s["question"]

        # Faithfulness: answer tokens supported by context
        context_text = " ".join(contexts)
        faithfulness_scores.append(token_overlap(answer, context_text) if context_text.strip() else 0.5)

        # Answer relevancy: overlap between answer and ground truth
        relevancy_scores.append(token_overlap(answer, ground_truth))

        # Context precision: how much of the context is relevant to the question
        precision_scores.append(token_overlap(context_text, question) if context_text.strip() else 0.5)

    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        "answer_relevancy": round(sum(relevancy_scores) / len(relevancy_scores), 3),
        "context_precision": round(sum(precision_scores) / len(precision_scores), 3),
    }


def log_to_mlflow(metrics: dict, run_name: str = "ragas_eval") -> None:
    try:
        import mlflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(metrics)
            logger.info(f"[RAGAS] Metrics logged to MLflow run: {run_name}")
    except Exception as exc:
        logger.warning(f"[RAGAS] MLflow logging skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RenewIQ RAGAS Evaluation Gate")
    parser.add_argument("--fail-below", type=float, default=0.85,
                        help="Minimum score for all metrics (default: 0.85)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Databricks serving endpoint URL (omit for local graph)")
    parser.add_argument("--token", type=str, default=os.getenv("DATABRICKS_TOKEN", ""),
                        help="Databricks API token")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Max eval samples to run (default: all 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this path")
    parser.add_argument("--mlflow", action="store_true",
                        help="Log metrics to MLflow")
    args = parser.parse_args()

    dataset = load_eval_dataset()[: args.max_samples]
    logger.info(f"[RAGAS] Running eval on {len(dataset)} samples")

    samples = []
    for i, item in enumerate(dataset, 1):
        logger.info(f"[RAGAS] {i}/{len(dataset)} — {item['id']}")
        try:
            if args.endpoint:
                result = run_agent_endpoint(
                    item["question"], item.get("context"), args.endpoint, args.token
                )
            else:
                result = run_agent_local(item["question"], item.get("context"))

            samples.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
                "risk_flags": result["risk_flags"],
            })
            time.sleep(0.5)  # Rate limit buffer

        except Exception as exc:
            logger.error(f"[RAGAS] Sample {item['id']} failed: {exc}")
            samples.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": f"ERROR: {exc}",
                "contexts": [],
                "risk_flags": [],
            })

    # Compute metrics
    metrics = compute_ragas_metrics(samples)

    logger.info("[RAGAS] ─── Results ───────────────────────────────────")
    for metric, score in metrics.items():
        threshold = args.fail_below if metric != "context_precision" else 0.80
        status = "PASS" if score >= threshold else "FAIL"
        logger.info(f"[RAGAS]   {metric:<25} {score:.3f}  [{status}]")
    logger.info("[RAGAS] ─────────────────────────────────────────────")

    # Write output if requested
    output = {
        "metrics": metrics,
        "samples_evaluated": len(samples),
        "thresholds": DEFAULT_THRESHOLDS,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info(f"[RAGAS] Results written to {args.output}")

    if args.mlflow:
        log_to_mlflow(metrics)

    # Gate check
    failed = []
    for metric, threshold in DEFAULT_THRESHOLDS.items():
        actual = metrics.get(metric, 0.0)
        if actual < threshold:
            failed.append(f"{metric}={actual:.3f} < {threshold}")

    if failed:
        logger.error(f"[RAGAS] GATE FAILED: {', '.join(failed)}")
        return 1

    logger.info("[RAGAS] All metrics above threshold — gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
