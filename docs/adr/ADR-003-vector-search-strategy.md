# ADR-003: Databricks Vector Search for PPA Clause Retrieval

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

The `contract_rag` agent must perform semantic similarity search over PPA clause chunks to retrieve the most relevant contract provisions for a given risk query (e.g., "what are the curtailment and force majeure provisions?"). PPA contracts are pre-processed into overlapping text chunks of ~512 tokens with metadata (contract ID, clause type, page number, party names) and stored in `silver.ppa_clauses_chunked`.

The embedding model (text-embedding-3-small via Azure OpenAI) produces 1536-dimensional vectors. At the expected scale of 10,000–500,000 clause chunks (supporting a portfolio of hundreds of PPAs), approximate nearest-neighbor (ANN) search is required.

Three vector store options were evaluated:

- **Pinecone**: Fully managed, high-performance vector database with a mature Python SDK. Excellent ANN performance and filtering capabilities. However, requires a separate ETL pipeline to sync clause chunks from Delta Lake to Pinecone, adding operational complexity, potential data lag, and a separate access control boundary outside Unity Catalog. Monthly cost at 500K vectors is approximately $70–100/month.
- **pgvector (PostgreSQL extension)**: Runs on Azure Database for PostgreSQL. Low additional cost if PostgreSQL is already in the stack. However, RenewIQ does not use PostgreSQL elsewhere, adding an entirely new infrastructure dependency. Synchronization from Delta Lake to pgvector requires a custom connector. ANN performance degrades significantly beyond ~100K vectors without careful HNSW index tuning.
- **Databricks Vector Search (Delta Sync index)**: Managed vector search integrated directly into the Databricks platform. A Delta Sync index automatically reads from a Delta table (in this case `silver.ppa_clauses_chunked`) and maintains the vector index incrementally as new chunks are written. Access is governed by Unity Catalog permissions — no separate credential management. Queried via the Databricks Python SDK (`WorkspaceClient.vector_search_indexes.query_index()`).

## Decision

Use **Databricks Vector Search** with a Delta Sync index backed by `silver.ppa_clauses_chunked`.

The index configuration:
- **Source table:** `silver.ppa_clauses_chunked`
- **Embedding column:** `chunk_embedding` (1536-dim float array, pre-computed by an embedding DLT pipeline)
- **Primary key:** `chunk_id` (UUID)
- **Sync mode:** Triggered (syncs on each Delta table commit; acceptable given PPA uploads are infrequent)
- **Metadata columns returned:** `contract_id`, `clause_type`, `chunk_text`, `page_number`, `party_a`, `party_b`, `effective_date`

The `contract_rag` agent queries the index with `num_results=15`, then passes the 15 candidates through the FlashRank cross-encoder reranker (see ADR-005) to select the top 3 for LLM context injection.

Filtering on `contract_id` allows scoping retrieval to a specific PPA when the user has selected a contract, rather than searching across all contracts in the portfolio.

## Consequences

**Positive:**
- Zero-copy integration: clause chunks written to `silver.ppa_clauses_chunked` by the DLT pipeline are automatically indexed — no separate ETL pipeline or sync job required.
- Unity Catalog governs access to the vector index using the same service principal and row-level security policies as the underlying Delta table.
- No additional infrastructure to manage or cost center to track; Vector Search is billed as Databricks DBUs.
- Metadata filtering (by `contract_id`, `clause_type`) is natively supported, enabling precise scoped searches.
- Eliminates a cross-system data consistency risk: the vector index and the source Delta table are always synchronized through the same Databricks transaction log.

**Negative:**
- Significant vendor lock-in: migrating to a different vector store would require rewriting the embedding pipeline, index management, and query layer.
- Databricks Vector Search is not available outside the Databricks platform; local development and testing require mocking the `WorkspaceClient` or using a lightweight in-process alternative (e.g., FAISS).
- Delta Sync index warm-up after workspace restart can take 2–5 minutes before the first query returns results.
- ANN recall benchmarks for Databricks Vector Search are not publicly disclosed; Pinecone publishes more transparent performance metrics.
