# ADR-005: FlashRank Cross-Encoder Reranker for RAG Precision

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

Databricks Vector Search uses approximate nearest-neighbor (ANN) search over dense embeddings to retrieve candidate PPA clause chunks. ANN retrieval optimizes for recall — it returns the `num_results=15` chunks most likely to be relevant — but embedding similarity is an imperfect proxy for relevance. A chunk may score high on embedding similarity because it shares vocabulary with the query while being semantically irrelevant in context (e.g., a definitions clause that uses the word "curtailment" in a different sense than the risk query).

The LLM context window passed to `report_writer` is intentionally limited to 3 clause chunks (~1500 tokens) to control LLM latency and cost. Passing all 15 ANN results directly would either overflow the context window or dilute the LLM's attention with irrelevant passages, degrading report quality.

A reranker sitting between ANN retrieval and LLM context injection — the standard two-stage RAG pattern — addresses this by applying a more expensive but more precise relevance model to the small candidate set.

Three reranking options were evaluated:

- **No reranker**: Pass the top-3 ANN results directly by embedding score. Fast and simple, but embedding similarity scores are poorly calibrated for cross-modal relevance (query about financial exposure vs. contract clause about delivery obligations).
- **Cohere Rerank API** (`rerank-english-v3.0`): State-of-the-art reranking quality, especially for domain-specific legal and financial text. Adds ~150–300ms latency per request (network round-trip to Cohere's API). Cost is $1.00 per 1,000 reranking calls — significant at scale. Introduces an external API dependency that creates an availability risk and data residency concern (contract text transmitted to Cohere).
- **FlashRank with ms-marco-MiniLM-L-12-v2**: Open-source Python library wrapping a cross-encoder model fine-tuned on MS MARCO passage ranking. Runs entirely in-process (no network call). Model loads once at application startup and is cached in memory. At 15 passages × ~200 tokens each, reranking completes in 40–60ms on a single CPU core.

## Decision

Use **FlashRank** with the `ms-marco-MiniLM-L-12-v2` cross-encoder model as the reranking stage in the `contract_rag` agent.

Implementation:
```python
from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")

def rerank_clauses(query: str, candidates: list[dict]) -> list[dict]:
    passages = [{"id": c["chunk_id"], "text": c["chunk_text"]} for c in candidates]
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    top3_ids = {r["id"] for r in results[:3]}
    return [c for c in candidates if c["chunk_id"] in top3_ids]
```

The model is loaded once at FastAPI application startup and held in memory (~90MB). The `/health` endpoint verifies the ranker is loaded before the service accepts traffic.

The choice of `ms-marco-MiniLM-L-12-v2` over the smaller `ms-marco-MiniLM-L-6-v2` reflects a deliberate trade-off: the L-12 model adds ~15ms latency but achieves meaningfully better ranking quality on legal passage retrieval tasks (evaluated on a held-out set of 50 PPA query-clause pairs).

Contract text does not leave the deployment environment, satisfying data residency requirements under the client's data processing agreements.

## Consequences

**Positive:**
- Zero marginal cost per reranking call — no API fees regardless of query volume.
- Fully offline-capable: operates in air-gapped or network-restricted deployment environments.
- Low latency: 40–60ms adds minimal overhead to the ~200ms ANN retrieval step.
- No external API dependency eliminates an availability risk and simplifies the service's dependency graph.
- Contract clause text never leaves the deployment environment, satisfying data residency and confidentiality requirements.

**Negative:**
- `ms-marco-MiniLM-L-12-v2` was fine-tuned on web passage retrieval (MS MARCO), not legal or energy contract text. Reranking quality on highly domain-specific PPA language may be lower than Cohere Rerank v3 (estimated 5–10% lower nDCG on held-out evaluation set).
- The ~90MB model occupies memory in each API pod; at 3 replicas this is 270MB of non-application memory, which must be accounted for in Kubernetes resource limits.
- FlashRank is a smaller open-source project with less frequent releases; long-term maintenance is less certain than a commercial API.
- Future fine-tuning on PPA-specific clause pairs would require custom model training infrastructure not currently in the MLflow registry.
