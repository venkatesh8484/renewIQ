"""
Contract RAG Agent
-------------------
Retrieves relevant PPA clause chunks for the user's query, reranks them
with the cross-encoder, and returns:
  - retrieved_clauses: top-N RankedPassage dicts
  - rag_context:       formatted clause snippets for downstream LLM prompts

LangGraph node: AgentState → dict of updates
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.agents.contract_rag.retriever import get_retriever
from src.agents.contract_rag.reranker import get_reranker

logger = logging.getLogger(__name__)

# How many VS hits to fetch before reranking, and how many to keep after
_RETRIEVAL_TOP_K = 15
_RERANK_TOP_N    = 6


def run(state: dict) -> dict:
    """
    LangGraph node function.

    1. Retrieve top-K chunks from Vector Search (or local Parquet)
    2. Rerank with cross-encoder (or identity reranker in CI)
    3. Format rag_context for LLM prompt
    """
    query        = state.get("query", "")
    contract_ids = state.get("contract_ids") or None

    logger.info(f"[ContractRAGAgent] query={query[:80]!r} contracts={contract_ids}")

    # Step 1: retrieve
    retriever = get_retriever()
    hits = retriever.search(
        query        = query,
        top_k        = _RETRIEVAL_TOP_K,
        contract_ids = contract_ids,
    )

    if not hits:
        logger.warning("[ContractRAGAgent] No hits retrieved — returning empty context")
        return {
            "retrieved_clauses": [],
            "rag_context":       "No relevant contract clauses found.",
        }

    # Step 2: rerank
    use_identity = os.getenv("RENEWIQ_RERANKER_IDENTITY", "false").lower() == "true"
    reranker = get_reranker(use_identity=use_identity)
    ranked   = reranker.rerank(query=query, passages=hits, top_n=_RERANK_TOP_N)

    # Step 3: convert to ContractClause dicts
    clauses = [
        {
            "chunk_id":      r.chunk_id,
            "contract_id":   r.contract_id,
            "clause_id":     r.clause_id,
            "section_title": r.section_title,
            "risk_category": r.risk_category,
            "chunk_text":    r.chunk_text,
            "rerank_score":  r.score,
        }
        for r in ranked
    ]

    rag_context = _format_context(clauses)
    logger.info(f"[ContractRAGAgent] returning {len(clauses)} clauses")

    return {
        "retrieved_clauses": clauses,
        "rag_context":       rag_context,
    }


def _format_context(clauses: list[dict]) -> str:
    """
    Format retrieved clauses into a compact prompt block.
    Each clause is shown with contract ID, clause reference, and text.
    """
    if not clauses:
        return "No relevant contract clauses found."

    lines = ["## Retrieved Contract Clauses\n"]
    for i, c in enumerate(clauses, start=1):
        risk_tag = f" [{c['risk_category']}]" if c.get("risk_category") else ""
        lines.append(
            f"**[{i}] {c['contract_id']} — {c['clause_id']} "
            f"{c['section_title']}{risk_tag}** (score: {c['rerank_score']:.2f})\n"
            f"> {c['chunk_text'][:400]}{'...' if len(c['chunk_text']) > 400 else ''}\n"
        )

    return "\n".join(lines)
