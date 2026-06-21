# ADR-001: Medallion Architecture for Data Lake Storage

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

RenewIQ ingests heterogeneous energy market data from multiple sources: EPEX NL spot prices (hourly auction results), ENTSO-E generation forecasts (per bidding zone, 15-minute resolution), GOPACS congestion event notifications, and PPA contract documents uploaded by users. These sources differ in schema stability, update frequency, data quality guarantees, and downstream consumption patterns.

Several competing storage strategies were evaluated:

- **Single-layer storage**: all data lands in one schema, transformations happen at query time via views. Simple to implement but mixing raw and curated data creates confusion and makes it hard to reprocess historical data after schema changes.
- **Two-layer (raw + curated)**: a staging area plus a production layer. Adequate for simple pipelines but provides no separation between ML feature requirements and dashboard aggregation requirements.
- **Three-layer Medallion (Bronze → Silver → Gold)**: industry-standard pattern for Lakehouse architectures, natively supported by Databricks Delta Live Tables (DLT) and Unity Catalog.

The system must satisfy the following constraints:
1. Raw data must be preserved indefinitely to allow full historical reprocessing when upstream schemas change (e.g., ENTSO-E API version upgrades).
2. ML feature engineering (hourly price volatility windows, rolling load factors) must operate on clean, type-validated data.
3. Risk scoring dashboards need pre-aggregated Gold tables to avoid recomputing expensive joins at query time.
4. Data lineage must be auditable for regulatory compliance in EU energy trading contexts.

## Decision

Adopt the three-layer Delta Lake Medallion architecture hosted in Databricks Unity Catalog:

**Bronze (Raw):** Append-only Delta tables receiving data exactly as it arrives from source systems. No schema enforcement beyond basic JSON parsing. Includes `_ingested_at` and `_source_system` metadata columns. EPEX data lands in `bronze.epex_nl_raw`, ENTSO-E in `bronze.entso_e_generation_raw`, GOPACS events in `bronze.gopacs_congestion_raw`. PPA contract text is stored as binary blobs in `bronze.ppa_documents_raw`.

**Silver (Validated + Typed):** DLT pipelines apply schema enforcement, null checks, range validation (e.g., prices must be within physically plausible EUR/MWh bounds), deduplication by natural keys, and unit normalization. Produces `silver.epex_prices_hourly`, `silver.entso_generation_15min`, `silver.gopacs_events`, `silver.ppa_clauses_chunked`. SCD Type 1 upserts for reference data.

**Gold (ML-Ready Features + Aggregates):** Pre-joined, feature-engineered tables consumed by the `market_data` agent and risk scoring models. Includes `gold.hourly_price_features` (rolling 30/90-day volatility, capture rate metrics), `gold.market_risk_signals` (composite risk indicators), and `gold.portfolio_exposure_daily` (EUR exposure by counterparty and delivery period).

Orchestration uses Databricks Delta Live Tables for Bronze→Silver and Databricks Workflows for Silver→Gold feature pipelines, triggered on arrival of new Bronze data.

## Consequences

**Positive:**
- Full data lineage from raw source to ML feature is traceable through Unity Catalog's column-level lineage graph.
- Historical reprocessing is possible by re-running DLT pipelines against unchanged Bronze tables after fixing transformation bugs.
- Incremental DLT processing minimizes compute costs — only new or changed records flow through the pipeline.
- Gold tables serve both the LangGraph `market_data` agent (via Databricks SQL connector) and Streamlit dashboards without duplicated query logic.
- Unity Catalog row/column-level access control can be applied uniformly across all layers.

**Negative:**
- Additional latency: end-to-end Bronze→Gold pipeline adds 10–20 minutes from raw data arrival to query-ready features. Acceptable given EPEX prices are hourly.
- Schema evolution in Bronze requires careful DLT pipeline versioning to avoid downstream breakage.
- Three layers increase storage costs marginally (~2-3x raw data size due to Delta log overhead and duplication).
