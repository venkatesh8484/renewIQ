"""
Cross-Encoder Reranker
-----------------------
Reranks top-K vector search results using a FlashRank cross-encoder.
Runs locally (CPU) — no GPU or external API required.

FlashRank is a lightweight re-ranking library that wraps ONNX cross-encoder
models. It's significantly faster than HuggingFace inference at runtime.

Usage:
    from src.agents.contract_rag.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query="negative price floor", passages=hits, top_n=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RankedPassage:
    """A single reranked result with score and original metadata."""
    rank:         int
    score:        float
    chunk_id:     str
    contract_id:  str
    clause_id:    str
    section_title: str
    risk_category: Optional[str]
    chunk_text:   str


class Reranker:
    """
    Wraps FlashRank's cross-encoder for query-passage reranking.

    Model: `ms-marco-MiniLM-L-12-v2` (default) — ~33 MB ONNX model,
    downloads once to ~/.cache/flashrank/ on first use.

    Set `model_name="ms-marco-TinyBERT-L-2-v2"` for faster but less accurate ranking.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._ranker = None  # lazy load — avoids import cost at startup

    def _get_ranker(self):
        if self._ranker is None:
            try:
                from flashrank import Ranker  # type: ignore[import]
                self._ranker = Ranker(model_name=self.model_name, max_length=self.max_length)
                logger.info(f"FlashRank ranker loaded: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "FlashRank not installed. Run: pip install flashrank --break-system-packages"
                )
        return self._ranker

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[RankedPassage]:
        """
        Rerank passages against a query using the cross-encoder.

        Args:
            query:    The user's natural-language question.
            passages: List of dicts from Vector Search. Expected keys:
                        chunk_id, contract_id, clause_id, section_title,
                        risk_category, chunk_text
            top_n:    Number of top results to return.

        Returns:
            List of RankedPassage sorted by descending relevance score.
        """
        if not passages:
            return []

        from flashrank import RerankRequest  # type: ignore[import]

        ranker = self._get_ranker()

        # Build passage list for FlashRank
        flash_passages = [
            {"id": i, "text": p.get("chunk_text", ""), "meta": p}
            for i, p in enumerate(passages)
        ]

        request = RerankRequest(query=query, passages=flash_passages)
        results = ranker.rerank(request)

        ranked: list[RankedPassage] = []
        for rank_pos, result in enumerate(results[:top_n], start=1):
            meta = result.get("meta", {})
            ranked.append(RankedPassage(
                rank          = rank_pos,
                score         = float(result.get("score", 0.0)),
                chunk_id      = meta.get("chunk_id", ""),
                contract_id   = meta.get("contract_id", ""),
                clause_id     = meta.get("clause_id", ""),
                section_title = meta.get("section_title", ""),
                risk_category = meta.get("risk_category"),
                chunk_text    = meta.get("chunk_text", ""),
            ))

        return ranked

    def rerank_texts(
        self,
        query: str,
        texts: list[str],
        top_n: int = 5,
    ) -> list[tuple[int, float, str]]:
        """
        Simplified interface: rerank plain text strings.
        Returns list of (original_index, score, text) sorted by score desc.
        """
        passages = [{"chunk_id": str(i), "contract_id": "", "clause_id": "",
                     "section_title": "", "risk_category": None, "chunk_text": t}
                    for i, t in enumerate(texts)]
        ranked = self.rerank(query, passages, top_n=top_n)
        return [(int(r.chunk_id), r.score, r.chunk_text) for r in ranked]


# ---------------------------------------------------------------------------
# Fallback: identity reranker for environments without FlashRank (e.g., CI)
# ---------------------------------------------------------------------------

class IdentityReranker:
    """
    No-op reranker that returns passages in original order with uniform scores.
    Used in unit tests and environments where FlashRank is not installed.
    """

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[RankedPassage]:
        results = []
        for rank_pos, p in enumerate(passages[:top_n], start=1):
            results.append(RankedPassage(
                rank          = rank_pos,
                score         = 1.0 - (rank_pos * 0.01),
                chunk_id      = p.get("chunk_id", ""),
                contract_id   = p.get("contract_id", ""),
                clause_id     = p.get("clause_id", ""),
                section_title = p.get("section_title", ""),
                risk_category = p.get("risk_category"),
                chunk_text    = p.get("chunk_text", ""),
            ))
        return results


def get_reranker(use_identity: bool = False) -> "Reranker | IdentityReranker":
    """
    Factory: returns real Reranker unless use_identity=True or
    RENEWIQ_RERANKER_IDENTITY env var is set (useful in CI).
    """
    import os
    if use_identity or os.getenv("RENEWIQ_RERANKER_IDENTITY", "").lower() == "true":
        return IdentityReranker()
    return Reranker()
