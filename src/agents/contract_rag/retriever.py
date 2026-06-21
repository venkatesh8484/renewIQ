"""
Contract RAG Retriever
-----------------------
Wraps Databricks Vector Search (production) and a local Parquet fallback
(offline / CI) behind a single interface.

Usage:
    retriever = get_retriever()
    hits = retriever.search("negative price floor curtailment", top_k=10)
    # → list[dict] with keys: chunk_id, contract_id, clause_id, section_title,
    #                          risk_category, chunk_text, score
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# VS config — matches what 02_setup_vector_search.py writes to gold.vector_search_config
VS_ENDPOINT = "renewiq-vs-endpoint"
VS_INDEX    = "renewiq.silver.ppa_chunks_vs_index"
_RESULT_COLS = [
    "chunk_id", "contract_id", "clause_id",
    "section_title", "risk_category", "chunk_text",
]


class VectorSearchRetriever:
    """Production retriever backed by Databricks Vector Search."""

    def __init__(
        self,
        endpoint: str = VS_ENDPOINT,
        index:    str = VS_INDEX,
    ):
        self.endpoint = endpoint
        self.index    = index
        self._client  = None

    def _get_client(self):
        if self._client is None:
            from databricks.vector_search.client import VectorSearchClient  # type: ignore
            self._client = VectorSearchClient(disable_notice=True)
        return self._client

    def search(
        self,
        query:          str,
        top_k:          int = 10,
        contract_ids:   Optional[list[str]] = None,
        risk_categories: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Run similarity search against the Vector Search index.

        Args:
            query:           Natural language query
            top_k:           Number of results to return
            contract_ids:    Filter to specific contracts (None = all)
            risk_categories: Filter to specific risk categories (None = all)

        Returns:
            List of hit dicts sorted by descending similarity score.
        """
        client = self._get_client()
        index  = client.get_index(self.endpoint, self.index)

        filters: dict = {}
        if contract_ids:
            filters["contract_id"] = contract_ids
        if risk_categories:
            filters["risk_category"] = risk_categories

        resp = index.similarity_search(
            query_text  = query,
            columns     = _RESULT_COLS,
            num_results = top_k,
            filters     = filters or {},
        )

        hits = []
        for row in resp.get("result", {}).get("data_array", []):
            # data_array rows: [col1, col2, ..., score]
            hit = dict(zip(_RESULT_COLS, row[:-1]))
            hit["score"] = float(row[-1])
            hits.append(hit)

        logger.info(f"[Retriever] {len(hits)} hits for query={query[:60]!r}")
        return hits


class LocalParquetRetriever:
    """
    Offline retriever for local dev and CI.
    Reads from data/silver/ppa_contract_chunks.parquet and does
    keyword-based scoring (TF-IDF approximation via token overlap).
    """

    def __init__(self, parquet_path: Optional[Path] = None):
        self._path = parquet_path or Path("data/silver/ppa_contract_chunks.parquet")
        self._df   = None

    def _load(self):
        if self._df is None:
            import pandas as pd
            if not self._path.exists():
                logger.warning(
                    f"[LocalParquetRetriever] {self._path} not found — "
                    "run: python scripts/ingest_contracts.py"
                )
                self._df = pd.DataFrame(columns=_RESULT_COLS)
            else:
                self._df = pd.read_parquet(self._path)
        return self._df

    def search(
        self,
        query:           str,
        top_k:           int = 10,
        contract_ids:    Optional[list[str]] = None,
        risk_categories: Optional[list[str]] = None,
    ) -> list[dict]:
        df = self._load().copy()

        if df.empty:
            return _mock_hits(query)

        # Filter
        if contract_ids:
            df = df[df["contract_id"].isin(contract_ids)]
        if risk_categories:
            df = df[df["risk_category"].isin(risk_categories)]

        # Score: token overlap between query tokens and chunk_text
        query_tokens = set(query.lower().split())
        def _score(text: str) -> float:
            if not isinstance(text, str):
                return 0.0
            chunk_tokens = set(text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            return overlap / max(len(query_tokens), 1)

        df["score"] = df["chunk_text"].apply(_score)
        df = df.sort_values("score", ascending=False).head(top_k)

        results = []
        for _, row in df.iterrows():
            results.append({
                "chunk_id":      row.get("chunk_id", ""),
                "contract_id":   row.get("contract_id", ""),
                "clause_id":     row.get("clause_id", ""),
                "section_title": row.get("section_title", ""),
                "risk_category": row.get("risk_category"),
                "chunk_text":    row.get("chunk_text", ""),
                "score":         float(row["score"]),
            })
        return results


def _mock_hits(query: str) -> list[dict]:
    """
    Deterministic mock hits — used when no parquet file exists.
    Returns realistic PPA clause snippets that trigger risk detection.
    """
    return [
        {
            "chunk_id":      "mock-001",
            "contract_id":   "zeeland-wind-physical-ppa-v1",
            "clause_id":     "7.2",
            "section_title": "Negative Price Provisions",
            "risk_category": "price_risk",
            "chunk_text": (
                "7.2 Negative Price Provisions. No price floor shall apply. "
                "If the day-ahead settlement price is negative for any delivery hour, "
                "the Seller shall pay the Buyer an amount equal to the absolute value "
                "of the negative price multiplied by the metered output for that hour."
            ),
            "score": 0.92,
        },
        {
            "chunk_id":      "mock-002",
            "contract_id":   "zeeland-wind-physical-ppa-v1",
            "clause_id":     "9.3",
            "section_title": "Curtailment Compensation",
            "risk_category": "curtailment_risk",
            "chunk_text": (
                "9.3 Curtailment Compensation. No compensation shall be payable to the Seller "
                "in respect of curtailment instructed by the relevant DSO or system operator. "
                "The Seller accepts full curtailment risk without deemed output provisions."
            ),
            "score": 0.87,
        },
        {
            "chunk_id":      "mock-003",
            "contract_id":   "nordsee-offshore-ppa-v2",
            "clause_id":     "7.1",
            "section_title": "Price Floor",
            "risk_category": "price_risk",
            "chunk_text": (
                "7.1 Price Floor. The settlement price shall not fall below zero EUR/MWh. "
                "In the event the day-ahead market price is negative, the applicable settlement "
                "price for this agreement shall be deemed to be zero EUR/MWh."
            ),
            "score": 0.81,
        },
    ]


def get_retriever(force_local: bool = False) -> "VectorSearchRetriever | LocalParquetRetriever":
    """
    Factory: returns VectorSearchRetriever in production,
    LocalParquetRetriever when running locally or in CI.
    """
    use_local = (
        force_local
        or os.getenv("RENEWIQ_USE_MOCK_ENDPOINT", "false").lower() == "true"
        or not os.getenv("DATABRICKS_HOST")
    )
    if use_local:
        logger.info("[Retriever] Using local Parquet retriever")
        return LocalParquetRetriever()

    logger.info("[Retriever] Using Databricks Vector Search retriever")
    return VectorSearchRetriever()
