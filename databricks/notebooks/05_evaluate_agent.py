# Databricks notebook source
# MAGIC %md
# MAGIC # RenewIQ — Phase 6: Agent Evaluation with Databricks Mosaic AI
# MAGIC
# MAGIC Evaluates the deployed `renewiq-agent-endpoint` against the 20-question eval dataset.
# MAGIC Logs all results to MLflow. Fails if faithfulness < 0.85.

# COMMAND ----------

# MAGIC %pip install -q ragas[all] datasets mlflow>=2.13 httpx

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import mlflow
from pathlib import Path

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/renewiq/agent_evaluation")

ENDPOINT_NAME = "renewiq-agent-endpoint"
DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
ENDPOINT_URL = f"https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

# Load eval dataset from repo
dataset_path = "/Repos/venkatesh8484@gmail.com/renewIQ/tests/evaluation/agent_eval_dataset.json"
with open(dataset_path) as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} eval questions")

# COMMAND ----------

# MAGIC %md ## 1. Collect agent responses

# COMMAND ----------

import httpx
import time

def call_endpoint(question: str, context: str | None) -> dict:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "contracts": [context] if context else [],
    }
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(ENDPOINT_URL, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json()


samples = []
for i, item in enumerate(dataset, 1):
    print(f"[{i}/{len(dataset)}] {item['id']} — {item['question'][:60]}...")
    try:
        result = call_endpoint(item["question"], item.get("context"))
        samples.append({
            "question": item["question"],
            "answer": result.get("response", ""),
            "contexts": [f["description"] for f in result.get("risk_flags", [])] or ["No clauses retrieved."],
            "ground_truth": item["ground_truth"],
            "id": item["id"],
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        samples.append({
            "question": item["question"],
            "answer": f"ERROR: {e}",
            "contexts": ["No clauses retrieved."],
            "ground_truth": item["ground_truth"],
            "id": item["id"],
        })
    time.sleep(1)

print(f"\nCollected {len(samples)} responses")

# COMMAND ----------

# MAGIC %md ## 2. Run RAGAS evaluation

# COMMAND ----------

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

ds = Dataset.from_list([{
    "question": s["question"],
    "answer": s["answer"],
    "contexts": s["contexts"],
    "ground_truth": s["ground_truth"],
} for s in samples])

result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
print(result)

# COMMAND ----------

# MAGIC %md ## 3. Log to MLflow and check gate

# COMMAND ----------

THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
}

metrics = {
    "faithfulness": float(result["faithfulness"]),
    "answer_relevancy": float(result["answer_relevancy"]),
    "context_precision": float(result["context_precision"]),
}

with mlflow.start_run(run_name="ragas_eval_prod") as run:
    mlflow.log_params({
        "endpoint": ENDPOINT_NAME,
        "eval_samples": len(samples),
        "thresholds": str(THRESHOLDS),
    })
    mlflow.log_metrics(metrics)
    mlflow.log_dict({"samples": samples}, "eval_samples.json")
    run_id = run.info.run_id

print(f"\nMLflow run: {run_id}")
print("\n=== RAGAS Results ===")
for metric, score in metrics.items():
    threshold = THRESHOLDS[metric]
    status = "PASS" if score >= threshold else "FAIL"
    print(f"  {metric:<25} {score:.3f}  [{status}] (threshold: {threshold})")

# COMMAND ----------

# Gate: fail notebook if any metric below threshold
failed = [(m, s) for m, s in metrics.items() if s < THRESHOLDS[m]]
if failed:
    msg = "RAGAS gate FAILED: " + ", ".join(f"{m}={s:.3f}<{THRESHOLDS[m]}" for m, s in failed)
    raise Exception(msg)

print("\nAll RAGAS metrics above threshold — evaluation gate PASSED")

# COMMAND ----------

# MAGIC %md ## 4. Per-question breakdown

# COMMAND ----------

import pandas as pd

df = pd.DataFrame([{
    "id": s["id"],
    "question": s["question"][:60],
    "answer_len": len(s["answer"]),
    "ground_truth_len": len(s["ground_truth"]),
} for s in samples])

display(df)
