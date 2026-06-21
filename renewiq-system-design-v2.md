# RenewIQ — Renewable Energy PPA Intelligence Copilot
## System Design Document v2.0 — Databricks Lakehouse Edition

> **Author:** Venkatesan Mariappan  
> **GitHub Target:** `github.com/venkatesanmariappan/renewiq`  
> **Platform:** Azure Databricks (Lakehouse + Mosaic AI)  
> **Architecture:** Medallion (Bronze → Silver → Gold) + Multi-Agent LLM  
> **Domain:** Renewable Energy · PPA Contract Intelligence · Market Risk  
> **Date:** June 2026

---

## Executive Summary

**RenewIQ** is a production-grade, multi-agent LLM system built on the **Azure Databricks Lakehouse** that helps energy traders, corporate sustainability managers, and renewable energy developers:

1. **Parse and understand** Power Purchase Agreement (PPA) contracts written in dense legal language
2. **Score and surface risks** — price exposure, volume shortfalls, curtailment clauses, force majeure gaps
3. **Cross-reference live market data** — EPEX SPOT NL day-ahead prices, ENTSO-E generation mix, GOPACS congestion signals
4. **Generate audit-ready risk reports** in plain language for both legal and commercial stakeholders

The entire data foundation is built on a **Medallion Architecture in Delta Lake**, governed by **Unity Catalog**. The agent layer is built with **LangGraph**, served via **Databricks Model Serving**, and traced end-to-end with **MLflow + Databricks Managed MLflow (MLflow 3)**.

---

## 1. Problem Statement

A corporate energy manager at a Dutch manufacturing company has signed a 15-year physical PPA with a wind farm in Zeeland at a fixed strike price of €68/MWh. On a Sunday in March 2026, EPEX SPOT NL day-ahead prices go **negative (−€42/MWh)** for 6 consecutive hours. The questions they now need to answer — manually, by hunting through a 120-page PDF:

- Does our contract have a **negative price floor clause**?
- Are we obligated to **take-or-pay** during negative price hours?
- What is our **imbalance exposure** if we deviate from our scheduled consumption?
- Is the wind farm under a **curtailment order** from TenneT right now?
- What is the **total financial exposure** over this 6-hour window?

Currently this takes 2–4 hours of manual cross-referencing. For a portfolio of 5 PPAs, this is a full-time job. RenewIQ answers all of this in under 30 seconds.

---

## 2. Platform Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AZURE DATABRICKS WORKSPACE                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    UNITY CATALOG (Governance Layer)                  │   │
│  │   Catalog: renewiq  |  Schemas: bronze · silver · gold · agents     │   │
│  │   Data lineage · Access control · Vector Search indexes · Models    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                  MEDALLION DATA LAYERS (Delta Lake)                   │  │
│  │                                                                       │  │
│  │  [BRONZE]─────────────►[SILVER]─────────────►[GOLD]                 │  │
│  │  Raw ingestion          Cleaned, validated     Consumption-ready      │  │
│  │  As-is from sources     Conformed schemas      Feature tables         │  │
│  │  Append-only Delta      Deduplicated           Agent-ready datasets   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                  MOSAIC AI AGENT LAYER                                │  │
│  │                                                                       │  │
│  │  Orchestrator Agent (LangGraph StateGraph)                           │  │
│  │  ├── Contract RAG Agent  → Databricks Vector Search (Unity Catalog)  │  │
│  │  ├── Market Data Agent   → Gold Delta tables via UC SQL tools        │  │
│  │  ├── Risk Scoring Agent  → Deterministic Python functions (UC tools) │  │
│  │  └── Report Agent        → Markdown + PDF export                     │  │
│  │                                                                       │  │
│  │  Served via: Databricks Model Serving endpoint                       │  │
│  │  Traced via: MLflow 3 (LangGraph native integration)                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
         │                            │                         │
┌────────▼──────────┐    ┌────────────▼──────────┐  ┌──────────▼────────────┐
│  External Sources  │    │   Azure ADLS Gen2      │  │  MLflow Registry      │
│                    │    │   (Delta Lake storage) │  │  Unity Catalog Models │
│  EPEX SPOT NL API  │    │   Bronze / Silver /    │  │  Agent versions       │
│  ENTSO-E REST API  │    │   Gold containers      │  │  Experiment tracking  │
│  GOPACS API        │    └───────────────────────┘  └───────────────────────┘
│  PPA PDFs (Blob)   │
└────────────────────┘
```

---

## 3. Medallion Architecture — Layer by Layer

### 3.1 BRONZE Layer — Raw Ingestion

The Bronze layer captures all data **as-is** from external sources. No transformations. Append-only Delta tables. Full historical archive for reprocessing.

#### Bronze Tables

```
renewiq.bronze
├── epex_dayahead_raw          -- Raw EPEX NL hourly price JSON from Stekker API
├── entso_generation_raw       -- Raw ENTSO-E generation mix XML/JSON per country
├── gopacs_announcements_raw   -- Raw GOPACS congestion market announcements
├── ppa_documents_raw          -- Raw PDF binary + metadata (contract_id, filename, upload_ts)
└── imbalance_prices_raw       -- Raw TenneT imbalance price feed
```

#### Bronze Ingestion Pipeline (Databricks Lakeflow / Auto Loader)

```python
# bronze/ingest_epex.py — Databricks notebook / Lakeflow job
import dlt  # Databricks Lakeflow Declarative Pipelines

@dlt.table(
    name="epex_dayahead_raw",
    comment="Raw EPEX day-ahead hourly prices for NL market from Stekker API",
    table_properties={"quality": "bronze"},
    schema="""
        ingestion_ts  TIMESTAMP,
        source_api    STRING,
        raw_payload   STRING,   -- full JSON response, unparsed
        market        STRING,
        fetch_date    DATE
    """
)
def ingest_epex_raw():
    return (
        spark.readStream
        .format("cloudFiles")          # Auto Loader
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", "/bronze/epex_schema")
        .load("abfss://raw@renewiqstorage.dfs.core.windows.net/epex/")
    )
```

```python
# bronze/ingest_ppa_docs.py — PDF metadata registration
@dlt.table(
    name="ppa_documents_raw",
    comment="Raw PPA PDF binaries metadata — content extracted in Silver",
    table_properties={"quality": "bronze"}
)
def ingest_ppa_docs():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .load("abfss://contracts@renewiqstorage.dfs.core.windows.net/ppa/")
        .select(
            "path", "modificationTime", "length",
            F.current_timestamp().alias("ingestion_ts")
        )
    )
```

---

### 3.2 SILVER Layer — Cleaned, Validated, Conformed

The Silver layer parses, validates, deduplicates, and standardises Bronze data into **consistent business schemas**. This is where PPA PDFs get parsed into text chunks and market data gets typed and validated.

#### Silver Tables

```
renewiq.silver
├── epex_dayahead              -- Parsed, typed hourly price rows (timestamp, price_eur_mwh, is_negative)
├── entso_generation           -- Generation mix per fuel type, per hour, per country
├── gopacs_congestion_events   -- Parsed congestion events (dso_zone, start, end, mw_needed, direction)
├── imbalance_prices           -- Typed TenneT imbalance price per settlement period
├── ppa_contract_metadata      -- Parsed PPA header info (contract_id, parties, strike_price, tenor, type)
└── ppa_contract_chunks        -- Text chunks from PDFs with risk_category tags (for vector indexing)
```

#### Silver Transformations

```python
# silver/transform_epex.py
@dlt.table(
    name="epex_dayahead",
    comment="Cleaned EPEX day-ahead prices with negative price flag",
    table_properties={"quality": "silver"},
    expectations={
        "valid_price_range": "price_eur_mwh BETWEEN -500 AND 4000",
        "no_null_timestamp": "delivery_timestamp IS NOT NULL"
    }
)
def transform_epex():
    return (
        dlt.read("epex_dayahead_raw")
        .select(
            F.from_json("raw_payload", epex_schema).alias("data")
        )
        .select(
            F.col("data.delivery_date").cast("date").alias("delivery_date"),
            F.col("data.hour").cast("int").alias("hour"),
            F.make_timestamp("data.delivery_date", "data.hour").alias("delivery_timestamp"),
            F.col("data.price").cast("double").alias("price_eur_mwh"),
            F.col("data.market").alias("market"),
            (F.col("data.price") < 0).alias("is_negative"),
            F.current_timestamp().alias("processed_ts")
        )
    )
```

```python
# silver/parse_ppa_chunks.py — PDF text extraction + chunking + risk tagging
@dlt.table(
    name="ppa_contract_chunks",
    comment="Section-aware text chunks from PPA PDFs with risk category metadata",
    table_properties={"quality": "silver"}
)
def parse_ppa_chunks():
    """
    UDF that:
    1. Reads PDF binary from ADLS via contract_id
    2. Extracts text using PyMuPDF (section-aware)
    3. Chunks to 512 tokens with 64-token overlap
    4. Tags each chunk with risk_category using a lightweight classifier
    5. Returns one row per chunk
    """
    return (
        dlt.read("ppa_documents_raw")
        .withColumn("chunks", parse_pdf_udf(F.col("path")))
        .select(
            F.col("contract_id"),
            F.explode("chunks").alias("chunk")
        )
        .select(
            "contract_id",
            F.col("chunk.text").alias("chunk_text"),
            F.col("chunk.clause_id").alias("clause_id"),
            F.col("chunk.section_title").alias("section_title"),
            F.col("chunk.page_number").alias("page_number"),
            F.col("chunk.risk_category").alias("risk_category"),  # price_risk, volume_risk, etc.
            F.col("chunk.token_count").alias("token_count")
        )
    )
```

---

### 3.3 GOLD Layer — Agent-Ready, Consumption-Ready

The Gold layer contains **denormalised, pre-aggregated, feature-engineered datasets** optimised for agent queries and LLM context assembly. This is what the agents actually query.

#### Gold Tables

```
renewiq.gold
├── market_risk_signals        -- Pre-computed: negative price windows, congestion alerts, imbalance spikes
├── ppa_risk_profile           -- Per-contract risk scores across all 6 risk categories
├── hourly_price_features      -- Feature table: price, rolling avg, volatility, renewable share
├── contract_clause_index      -- Vector Search index over silver.ppa_contract_chunks
└── portfolio_exposure_daily   -- Daily aggregated financial exposure per PPA per risk type
```

#### Gold Feature Engineering

```python
# gold/market_risk_signals.py
@dlt.table(
    name="market_risk_signals",
    comment="Pre-computed market stress signals for agent consumption",
    table_properties={"quality": "gold"}
)
def compute_market_signals():
    epex = dlt.read("epex_dayahead")
    gopacs = dlt.read("gopacs_congestion_events")
    entso = dlt.read("entso_generation")

    # Negative price windows
    negative_windows = (
        epex.filter("is_negative = true")
        .groupBy(F.window("delivery_timestamp", "1 hour"))
        .agg(
            F.avg("price_eur_mwh").alias("avg_negative_price"),
            F.min("price_eur_mwh").alias("min_price"),
            F.count("*").alias("negative_hours_in_window")
        )
    )

    # Renewable oversupply flag (wind + solar > 70% of load)
    renewable_share = (
        entso
        .withColumn(
            "renewable_pct",
            (F.col("wind_onshore_mw") + F.col("solar_mw")) / F.col("total_load_mw")
        )
        .filter("renewable_pct > 0.70")
        .withColumn("oversupply_flag", F.lit(True))
    )

    return negative_windows.join(renewable_share, on="window", how="left")
```

---

## 4. Unity Catalog — Governance & Vector Search

**Unity Catalog** is the single governance layer across all data assets, vector indexes, ML models, and agent tools.

```
Unity Catalog Structure:
renewiq (catalog)
├── bronze (schema)      → Raw Delta tables
├── silver (schema)      → Cleaned Delta tables
├── gold (schema)        → Feature/analytics tables
├── agents (schema)      → Agent tools (UC SQL/Python functions)
│   ├── get_market_signals()        -- SQL function over gold.market_risk_signals
│   ├── query_contract_clauses()    -- Python function → Vector Search query
│   ├── calculate_exposure()        -- Python function → deterministic risk calc
│   └── generate_risk_report()      -- Python function → Markdown report
└── models (schema)      → MLflow registered models
    ├── renewiq-orchestrator-agent  -- LangGraph StateGraph
    └── renewiq-embedding-model     -- nomic-embed-text (self-hosted)
```

### Vector Search Index (Contract Clause RAG)

```python
# Setup: Databricks Vector Search index over Silver chunks table
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create Vector Search endpoint
vsc.create_endpoint(
    name="renewiq-contract-search",
    endpoint_type="STANDARD"
)

# Create Delta Sync index — auto-updates when silver.ppa_contract_chunks changes
vsc.create_delta_sync_index(
    endpoint_name="renewiq-contract-search",
    index_name="renewiq.agents.ppa_clause_index",
    source_table_name="renewiq.silver.ppa_contract_chunks",
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_source_column="chunk_text",
    embedding_model_endpoint_name="renewiq-embedding-endpoint",  # nomic-embed-text
    columns_to_sync=["contract_id", "clause_id", "section_title", 
                     "page_number", "risk_category", "chunk_text"]
)
```

---

## 5. Agent Layer (Mosaic AI + LangGraph)

### 5.1 Orchestrator Agent

```python
# agents/orchestrator/graph.py
from langgraph.graph import StateGraph, END
from databricks_langchain import ChatDatabricks
from databricks.sdk import WorkspaceClient

# Use Databricks-hosted LLM (DBRX or GPT-4o via AI Gateway)
llm = ChatDatabricks(endpoint="databricks-dbrx-instruct")

class RenewIQState(TypedDict):
    messages: Annotated[list, add_messages]
    contracts_in_scope: list[str]
    risk_flags: list[dict]
    agent_outputs: dict
    final_report: Optional[str]

def build_graph():
    graph = StateGraph(RenewIQState)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("contract_rag", contract_rag_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_to_agents)
    graph.add_edge("aggregate", "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()
```

### 5.2 Unity Catalog Tools — Agent Tool Binding

Agents interact with Gold data through **Unity Catalog functions** registered as LangChain/LangGraph tools. This gives full data lineage, access control, and auditability through Unity Catalog.

```python
# agents/tools/uc_tools.py
from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
toolkit = UCFunctionToolkit(
    function_names=[
        "renewiq.agents.get_market_signals",       # Query gold.market_risk_signals
        "renewiq.agents.query_contract_clauses",   # Vector Search over silver chunks
        "renewiq.agents.calculate_exposure",       # Deterministic risk calculator
        "renewiq.agents.get_gopacs_congestion",    # Query gold congestion events
    ]
)

# Tools are now available to any LangGraph agent node
tools = toolkit.get_tools()
```

```sql
-- Example Unity Catalog SQL tool (registered in UC)
CREATE OR REPLACE FUNCTION renewiq.agents.get_market_signals(
    market STRING DEFAULT 'NL',
    lookback_hours INT DEFAULT 24
)
RETURNS TABLE (
    window_start TIMESTAMP,
    window_end   TIMESTAMP,
    signal_type  STRING,      -- 'negative_price', 'congestion', 'imbalance_spike'
    severity     STRING,      -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    avg_price    DOUBLE,
    details      STRING
)
COMMENT 'Fetch pre-computed market stress signals from Gold layer'
RETURN
    SELECT window_start, window_end, signal_type, severity, avg_negative_price, details
    FROM renewiq.gold.market_risk_signals
    WHERE market = market
      AND window_start >= CURRENT_TIMESTAMP() - MAKE_INTERVAL(0, 0, 0, 0, lookback_hours)
    ORDER BY window_start DESC;
```

### 5.3 Contract RAG Agent (Vector Search)

```python
# agents/contract_rag/agent.py
from databricks_langchain import DatabricksVectorSearch
from langchain_core.tools import tool

vector_store = DatabricksVectorSearch(
    endpoint="renewiq-contract-search",
    index_name="renewiq.agents.ppa_clause_index",
    columns=["chunk_text", "clause_id", "section_title", 
             "page_number", "risk_category", "contract_id"]
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 8,
        "filters": {"risk_category": "price_risk"}   # optional metadata filter
    }
)

@tool
def query_contract_clauses(query: str, contract_id: str, 
                           risk_category: str = None) -> list[dict]:
    """
    Retrieve relevant PPA contract clauses using semantic search.
    Optionally filter by risk_category (price_risk, volume_risk, 
    curtailment_risk, basis_risk, counterparty_risk, legal_regulatory).
    """
    filters = {"contract_id": contract_id}
    if risk_category:
        filters["risk_category"] = risk_category

    docs = vector_store.similarity_search_with_score(
        query, k=8, filter=filters
    )
    # Re-rank top results
    return rerank_results(docs, query, top_k=3)
```

### 5.4 Risk Scoring Agent (Deterministic)

Financial calculations are **always deterministic Python** — never delegated to LLM reasoning.

```python
# agents/risk_scoring/calculators.py
from pyspark.sql import SparkSession

def score_negative_price_exposure(
    contract_id: str,
    strike_price_eur: float,
    volume_mw: float,
    negative_hours: DataFrame   # from gold.market_risk_signals
) -> RiskFlag:
    """
    Exposure = Σ (strike_price - epex_price) × volume_mwh per negative hour
    Pure Python/Spark — zero LLM involvement in the calculation.
    """
    exposure_df = negative_hours.withColumn(
        "hourly_exposure",
        (F.lit(strike_price_eur) - F.col("avg_negative_price")) * F.lit(volume_mw)
    )
    total_exposure = exposure_df.agg(F.sum("hourly_exposure")).collect()[0][0]

    return RiskFlag(
        risk_category="price_risk",
        severity=classify_severity(total_exposure),
        financial_exposure_eur=round(total_exposure, 2),
        contract_id=contract_id,
        market_trigger=f"EPEX NL negative: avg {negative_hours.avg_negative_price}€/MWh"
    )
```

---

## 6. MLflow 3 — LLMOps & Experiment Tracking

Databricks MLflow 3 provides **native LangGraph tracing** — every agent call, tool invocation, and LLM completion is automatically traced without additional instrumentation code.

```python
# Enable MLflow auto-tracing for LangGraph
import mlflow
mlflow.langchain.autolog()   # Captures all LangGraph runs automatically

# Log the agent to MLflow
with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=build_graph(),
        artifact_path="renewiq-agent",
        registered_model_name="renewiq.models.orchestrator-agent"
    )
```

### What Gets Traced Per Query

```
MLflow Trace: query_20260621_093012
├── classify_intent
│   ├── input: "What is our exposure if EPEX goes negative?"
│   ├── llm_call: dbrx-instruct (latency: 0.8s, tokens: 312)
│   └── output: ["contract_rag", "market_data", "risk_scoring"]
├── contract_rag
│   ├── tool_call: query_contract_clauses (latency: 1.2s)
│   │   ├── vector_search_results: 8 chunks retrieved
│   │   └── reranked: 3 chunks returned
│   └── llm_call: dbrx-instruct (latency: 1.4s)
├── market_data
│   ├── tool_call: get_market_signals (latency: 0.3s)
│   └── output: {negative_hours: 5, avg_price: -28.4}
├── risk_scoring
│   ├── tool_call: calculate_exposure (latency: 0.1s, deterministic)
│   └── output: {exposure_eur: 5760.00, severity: "HIGH"}
└── report_generation (latency: 2.1s)
Total latency: 6.9s  |  Total tokens: 1,847  |  Cost: $0.024
```

---

## 7. Lakeflow Pipeline DAG

All data transformations run as **Databricks Lakeflow Declarative Pipelines** — fully managed, with automatic dependency inference and incremental processing.

```
LAKEFLOW PIPELINE: renewiq_data_pipeline
                                                           
  [External Sources]                                        
        │                                                  
        ├── Stekker API (EPEX)      ──► bronze.epex_dayahead_raw
        ├── ENTSO-E REST API        ──► bronze.entso_generation_raw
        ├── GOPACS API              ──► bronze.gopacs_announcements_raw
        └── ADLS Blob (PDFs)       ──► bronze.ppa_documents_raw
                                                           
  [Bronze → Silver]                                        
        ├── epex_dayahead_raw       ──► silver.epex_dayahead
        ├── entso_generation_raw    ──► silver.entso_generation
        ├── gopacs_announcements_raw──► silver.gopacs_congestion_events
        └── ppa_documents_raw       ──► silver.ppa_contract_chunks
                                         └──► Vector Search Index (auto-sync)
                                                           
  [Silver → Gold]                                         
        ├── silver.epex_dayahead        ┐
        ├── silver.entso_generation     ├──► gold.market_risk_signals
        └── silver.gopacs_congestion    ┘
        
        ├── silver.epex_dayahead        ──► gold.hourly_price_features
        
        └── silver.ppa_contract_metadata ┐
            gold.market_risk_signals      ├──► gold.ppa_risk_profile
            silver.ppa_contract_chunks   ┘

  Schedule: 
    ├── Market data: every 15 minutes (streaming)
    └── PPA documents: triggered on new file upload
```

---

## 8. Model Serving — Production Deployment

The agent is deployed as a **Databricks Model Serving endpoint** — no AKS, no Kubernetes manifests to manage. Databricks handles autoscaling, blue/green deployment, and load balancing natively.

```python
# Deploy agent to Model Serving
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()

w.serving_endpoints.create(
    name="renewiq-agent-endpoint",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="renewiq.models.orchestrator-agent",
                entity_version="3",
                scale_to_zero_enabled=True,
                workload_size="Small"
            )
        ]
    )
)
```

### REST Interface

```bash
# Query the deployed agent
curl -X POST https://<workspace>.azuredatabricks.net/serving-endpoints/renewiq-agent-endpoint/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "EPEX is negative right now. What is our exposure on the Zeeland PPA?"
    }],
    "contracts": ["zeeland-wind-ppa-v1"]
  }'
```

---

## 9. Observability Stack

| Concern | Tool | What's Tracked |
|---|---|---|
| LLM + Agent traces | MLflow 3 (native LangGraph integration) | Every prompt, completion, tool call, latency, token cost per trace |
| RAG quality | Databricks Agent Evaluation + RAGAS | Faithfulness, answer relevancy, context precision — run in CI |
| Data quality | Lakeflow DLT Expectations | Schema validation, null checks, price range constraints at Silver layer |
| Data lineage | Unity Catalog lineage graph | Column-level lineage from raw source to Gold feature table |
| Pipeline health | Databricks Lakeflow monitoring | Job run status, backlog lag, failed batches |
| Infrastructure | Databricks cluster metrics | CPU/memory/autoscaling per cluster per layer |
| Feedback loop | Delta table: `gold.agent_feedback` | User thumbs up/down + comment per agent response session |
| Cost tracking | MLflow trace cost field | Token cost per query tracked and aggregated in MLflow UI |

---

## 10. Repository Structure

```
renewiq/
├── README.md
├── docs/
│   ├── system-design.md               ← This document
│   ├── data-dictionary.md             ← PPA, EPEX, GOPACS, ENTSO-E glossary
│   ├── medallion-layer-guide.md       ← What belongs in each layer and why
│   └── adr/
│       ├── ADR-001-databricks-over-aks.md
│       ├── ADR-002-unity-catalog-vector-search.md
│       ├── ADR-003-lakeflow-declarative-pipelines.md
│       ├── ADR-004-uc-tools-over-custom-apis.md
│       └── ADR-005-synthetic-ppa-contracts.md
├── databricks/
│   ├── pipelines/
│   │   ├── bronze_ingestion.py        ← Lakeflow DLT: all bronze tables
│   │   ├── silver_transforms.py       ← Lakeflow DLT: all silver tables
│   │   └── gold_features.py           ← Lakeflow DLT: all gold tables
│   ├── notebooks/
│   │   ├── 01_setup_unity_catalog.py  ← Create catalog, schemas, grants
│   │   ├── 02_setup_vector_search.py  ← Create VS endpoint + delta sync index
│   │   ├── 03_register_uc_tools.py    ← Register SQL/Python functions in UC
│   │   ├── 04_deploy_agent.py         ← Log agent to MLflow + deploy to serving
│   │   └── 05_evaluate_agent.py       ← RAGAS + Databricks Agent Evaluation
│   └── bundle/
│       └── databricks.yml             ← Databricks Asset Bundle (IaC)
├── src/
│   ├── agents/
│   │   ├── orchestrator/
│   │   │   ├── graph.py               ← LangGraph StateGraph
│   │   │   ├── state.py               ← RenewIQState TypedDict
│   │   │   ├── router.py              ← Intent → agent routing
│   │   │   └── prompts.py
│   │   ├── contract_rag/
│   │   │   ├── agent.py
│   │   │   └── reranker.py            ← FlashRank cross-encoder
│   │   ├── market_data/
│   │   │   └── agent.py
│   │   ├── risk_scoring/
│   │   │   ├── agent.py
│   │   │   ├── calculators.py         ← Deterministic exposure formulas
│   │   │   └── risk_models.py         ← RiskFlag dataclass + 6 risk checks
│   │   └── report_generation/
│   │       ├── agent.py
│   │       └── templates/             ← Jinja2 Markdown templates
│   ├── ingestion/
│   │   ├── epex_fetcher.py            ← Stekker API client
│   │   ├── entso_fetcher.py           ← ENTSO-E REST API client
│   │   ├── gopacs_fetcher.py          ← GOPACS announcement scraper
│   │   └── pdf_parser.py              ← PyMuPDF section-aware chunker
│   └── uc_tools/
│       ├── market_signals.sql         ← UC SQL function definitions
│       ├── exposure_calculator.py     ← UC Python function
│       └── register_tools.py          ← Tool registration script
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│       ├── ragas_eval.py              ← RAGAS faithfulness + relevancy
│       └── agent_eval_dataset.json    ← Q&A pairs for agent evaluation
├── scripts/
│   ├── seed_market_data.py
│   ├── generate_synthetic_ppas.py
│   └── ingest_contracts.py
└── .github/
    └── workflows/
        ├── ci.yml                     ← Test + Lint + RAGAS eval gate
        └── cd.yml                     ← Databricks Asset Bundle deploy
```

---

## 11. Architecture Decision Records (ADRs)

### ADR-001: Databricks over AKS + Custom Stack
**Decision:** Run everything on Azure Databricks instead of a self-managed AKS cluster with TimescaleDB + pgvector.  
**Rationale:** Databricks collocates the data layer (Delta Lake), vector search (Databricks Vector Search), model serving, MLflow tracing, and pipeline orchestration (Lakeflow) in one governed platform. This eliminates: separate Kubernetes cluster management, separate vector DB ops, separate time-series DB ops, and cross-service networking. Unity Catalog provides column-level lineage and access control across all layers — impossible to replicate across 5 separate services.

### ADR-002: Databricks Vector Search over pgvector
**Decision:** Databricks Vector Search (Unity Catalog managed) for contract clause retrieval.  
**Rationale:** Databricks Vector Search automatically syncs with the Silver Delta table (`ppa_contract_chunks`) when new contracts are ingested — zero manual embedding re-indexing. The index lives inside Unity Catalog, meaning the same governance policies (access control, lineage, audit) that apply to raw PPA data also apply to the vector index. pgvector requires separate index refresh jobs and lives outside Unity Catalog governance.

### ADR-003: Lakeflow Declarative Pipelines over Custom PySpark Jobs
**Decision:** Databricks Lakeflow (DLT) for all Bronze → Silver → Gold transformations.  
**Rationale:** Lakeflow infers the DAG from table references — no explicit dependency wiring needed. Built-in DLT Expectations provide row-level data quality enforcement with quarantine tables, replacing custom validation code. Auto Loader handles new file arrivals natively.

### ADR-004: Unity Catalog Functions as Agent Tools
**Decision:** Register all agent tools as Unity Catalog SQL/Python functions.  
**Rationale:** UC functions give every tool call full data lineage (which Gold table was queried, by which agent, at what time), access control (tools can be granted to specific service principals), and discoverability (tools documented in UC UI). Any future agent can reuse registered tools without new API development.

### ADR-005: Synthetic PPA Contracts
**Decision:** Use synthetically generated but structurally realistic PPA PDFs.  
**Rationale:** Real PPA contracts are commercially sensitive. Synthetic contracts based on EIB, DLA Piper, and ACM templates faithfully represent the risk clause structures that matter. Noted prominently in README — standard practice in legal NLP research.

---

## 12. Full Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Lakehouse Platform** | Azure Databricks | Unified platform: data + AI + governance |
| **Data Format** | Delta Lake | ACID transactions, time travel, schema evolution |
| **Governance** | Unity Catalog | Lineage, access control, model & tool registry |
| **Pipeline Orchestration** | Databricks Lakeflow (DLT) | Bronze → Silver → Gold declarative pipelines |
| **Vector Search** | Databricks Vector Search | Delta-sync RAG index over PPA chunks |
| **Agent Framework** | LangGraph 0.3 | Multi-agent StateGraph orchestration |
| **LLM** | DBRX (Databricks) / GPT-4o via AI Gateway | Reasoning + clause interpretation |
| **Embeddings** | nomic-embed-text (Model Serving, self-hosted) | Contract chunk + query embedding |
| **UC Tool Integration** | `databricks-langchain` + `unitycatalog-ai` | Bind UC functions as LangGraph tools |
| **LLM Tracing** | MLflow 3 (LangGraph autolog) | End-to-end agent trace per query |
| **Agent Evaluation** | Databricks Agent Evaluation + RAGAS | Faithfulness, relevancy, CI quality gate |
| **Model Registry** | MLflow + Unity Catalog Models | Agent versioning and promotion |
| **Model Serving** | Databricks Model Serving | Managed endpoint, autoscaling, zero-config |
| **API Layer** | FastAPI (thin wrapper over Databricks endpoint) | REST interface for external consumers |
| **PDF Parsing** | PyMuPDF | Section-aware text extraction |
| **Re-ranking** | FlashRank (cross-encoder, local) | Improve RAG precision post-retrieval |
| **PDF Export** | WeasyPrint | Markdown → styled PDF report |
| **IaC** | Databricks Asset Bundle (DAB) | Workspace resource deployment |
| **CI/CD** | GitHub Actions | Test → RAGAS eval gate → DAB deploy |
| **Data Sources** | Stekker API, ENTSO-E, GOPACS, Azure Blob | Market data + PPA documents |

---

## 13. Containerisation Strategy

The JD calls out Docker and Kubernetes explicitly. Here is where each layer lives and why the approach is coherent.

### Local Development — Docker Compose

All local development runs in Docker. The `docker-compose.yml` spins up:

- **FastAPI service** (containerised) — the thin REST wrapper that proxies to the Databricks Model Serving endpoint
- **Redis** — response cache for market data (avoids repeated API calls during dev)
- **Mock Databricks stub** — a lightweight FastAPI mock that simulates the Databricks endpoint response shape for offline development and unit testing (no live Databricks credits needed locally)

```yaml
# docker-compose.yml
version: "3.9"

services:

  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABRICKS_HOST=${DATABRICKS_HOST}
      - DATABRICKS_TOKEN=${DATABRICKS_TOKEN}
      - SERVING_ENDPOINT=renewiq-agent-endpoint
      - REDIS_URL=redis://redis:6379
      - USE_MOCK_ENDPOINT=${USE_MOCK_ENDPOINT:-false}
    depends_on:
      - redis
    volumes:
      - ./src:/app/src
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  mock-databricks:
    build:
      context: .
      dockerfile: docker/mock/Dockerfile
    ports:
      - "8001:8001"
    volumes:
      - ./tests/fixtures:/app/fixtures   # pre-recorded response fixtures
    profiles:
      - offline   # only starts with: docker compose --profile offline up
```

```dockerfile
# docker/api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir ".[api]"

COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Local dev workflow:**
```bash
# Online mode — proxies to real Databricks endpoint
docker compose up api redis

# Offline mode — uses mock stub for unit/integration testing
docker compose --profile offline up
pytest tests/integration/ --mock-endpoint
```

### Production — Databricks Model Serving (Kubernetes-backed)

In production, the agent is deployed via **Databricks Model Serving**, which runs on Kubernetes internally — managed, auto-scaling, with blue/green deployment handled by Databricks. The FastAPI service is the only component that needs external Kubernetes if you want edge deployment outside the Databricks workspace.

**Production topology:**
```
[Client / External System]
         │
         ▼
[FastAPI Container]  ← Deployed to AKS (single lightweight service)
         │              Dockerfile: docker/api/Dockerfile
         │              Helm chart: infra/helm/renewiq-api/
         ▼
[Databricks Model Serving Endpoint]  ← Kubernetes-managed internally by Databricks
         │
         ├── LangGraph Orchestrator Agent
         ├── Unity Catalog Tools (SQL + Python functions)
         └── Databricks Vector Search
```

**Why this split is intentional (ADR-006):**

Containerising the entire agent stack (LangGraph + Spark + Vector Search) would be a significant ops burden with no benefit — that's exactly what Databricks Model Serving solves. The right boundary is to containerise the **external-facing API layer** (stateless, lightweight, easily replaceable) and let Databricks manage the stateful AI/data layer. This is the pattern used in production at scale.

```yaml
# infra/helm/renewiq-api/values.yaml
replicaCount: 2

image:
  repository: renewiqacr.azurecr.io/renewiq-api
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.renewiq.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: "250m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70

env:
  - name: DATABRICKS_HOST
    valueFrom:
      secretKeyRef:
        name: renewiq-secrets
        key: databricks-host
  - name: DATABRICKS_TOKEN
    valueFrom:
      secretKeyRef:
        name: renewiq-secrets
        key: databricks-token
```

```yaml
# .github/workflows/cd.yml (container build + AKS deploy step)
- name: Build and push API image
  run: |
    docker build -t renewiqacr.azurecr.io/renewiq-api:${{ github.sha }} \
      -f docker/api/Dockerfile .
    docker push renewiqacr.azurecr.io/renewiq-api:${{ github.sha }}

- name: Deploy to AKS
  run: |
    helm upgrade --install renewiq-api infra/helm/renewiq-api/ \
      --set image.tag=${{ github.sha }} \
      --namespace renewiq \
      --create-namespace
```

### ADR-006: Kubernetes at the API Boundary Only

**Decision:** Containerise only the FastAPI gateway; deploy the agent via Databricks Model Serving.

**Rationale:** The agent layer requires Spark compute, managed Vector Search, and Unity Catalog access — none of which run cleanly in a self-managed container. Databricks Model Serving provides Kubernetes-backed auto-scaling, blue/green deployment, and GPU access for embedding models without cluster management overhead. The FastAPI gateway is stateless, lightweight, and the correct boundary for containerisation: it can be deployed to any Kubernetes cluster (AKS, EKS, GKE) independent of the Databricks workspace. This gives full container/K8s demonstrated skills without re-introducing the AKS complexity that Databricks eliminates.

---

## 14. Getting Started (Local → Databricks)

```bash
# Clone
git clone https://github.com/venkatesanmariappan/renewiq
cd renewiq

# Install Databricks CLI + authenticate
pip install databricks-cli
databricks configure --token

# Deploy workspace resources via Databricks Asset Bundle
databricks bundle deploy --target dev

# Seed 90 days of market data into Bronze
databricks jobs run-now --job-id $(cat .bundle/dev/job_ids.json | jq '.seed_market_data')

# Generate + ingest synthetic PPA contracts
python scripts/generate_synthetic_ppas.py
python scripts/ingest_contracts.py

# Run Lakeflow pipeline (Bronze → Silver → Gold)
databricks pipelines start --pipeline-id $(cat .bundle/dev/pipeline_ids.json | jq '.renewiq_pipeline')

# Deploy agent to Model Serving
databricks notebooks run databricks/notebooks/04_deploy_agent.py

# Test the endpoint
curl -X POST https://<workspace>.azuredatabricks.net/serving-endpoints/renewiq-agent-endpoint/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is our exposure if EPEX goes negative tonight?"}]}'
```

---

## 14. Skills Demonstrated vs Enjins JD

| Enjins Requirement | How RenewIQ (v2) Demonstrates It |
|---|---|
| **Python** | Agents, Lakeflow DLT pipelines, UC tools, ingestion clients |
| **SQL** | UC SQL tool functions, Gold layer queries, Silver DLT expectations |
| **Cloud (Azure)** | Azure Databricks, ADLS Gen2, Azure Key Vault — all via DAB |
| **Data engineering** | Full Medallion pipeline: Auto Loader → Bronze → Silver → Gold |
| **LLMs into production** | DBRX via Databricks Model Serving + MLflow registry |
| **RAG** | Databricks Vector Search (Delta sync) + cross-encoder reranking |
| **Multi-agent systems** | LangGraph: Orchestrator + 4 specialist agents, parallel fan-out |
| **Tool-use** | Unity Catalog functions as LangGraph tools |
| **MLOps / LLMOps** | MLflow 3 autolog, Agent Evaluation, RAGAS CI gate, feedback loop |
| **ETL / ELT** | Lakeflow DLT with Auto Loader, incremental processing, DQ expectations |
| **Docker** | Dockerfile for FastAPI service; docker-compose for local dev with offline mock mode |
| **Kubernetes** | Helm chart for FastAPI on AKS; production agent on Databricks Model Serving (K8s-backed) |
| **SAS/Databricks migration background** | Medallion design + Lakeflow = direct parallel to SAS→Databricks migration work |
| **Climate / Energy domain** | EPEX, GOPACS, ENTSO-E data — directly relevant to Enjins' energy client portfolio |

---

*RenewIQ v2 is purpose-built on Databricks — the same platform Enjins uses for production data and AI work with their energy clients. Every component, from Lakeflow pipelines to Unity Catalog governance to MLflow tracing, is a direct signal to Enjins that you can be productive in their stack from day one.*
