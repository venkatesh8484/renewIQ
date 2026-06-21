# RenewIQ — Renewable Energy PPA Intelligence Copilot
## System Design Document v1.0

> **Author:** Venkatesan Mariappan  
> **GitHub Target:** `github.com/venkatesanmariappan/renewiq`  
> **Domain:** Renewable Energy · Contract Intelligence · Market Risk  
> **Date:** June 2026

---

## Executive Summary

**RenewIQ** is a production-grade, multi-agent LLM system that helps energy traders, corporate sustainability managers, and renewable energy developers:

1. **Parse and understand** Power Purchase Agreement (PPA) contracts written in dense legal language
2. **Score and surface risks** — price exposure, volume shortfalls, curtailment clauses, force majeure gaps
3. **Cross-reference live market data** — EPEX SPOT day-ahead prices (NL), ENTSO-E generation mix, GOPACS congestion signals
4. **Generate audit-ready risk reports** in plain language for both legal and commercial stakeholders

**Why this problem exists:** A corporate PPA is typically a 10–20 year contract locking in a fixed electricity price. Within that contract, 6 distinct risk categories (price, volume, curtailment, basis, counterparty, legal/regulatory) are buried across 80–150 pages of legal text. When market conditions shift — a negative price spike, a grid congestion event, a regulatory change — no operator today has a system that automatically cross-references the live market signal with the contract clause and surfaces the financial exposure. That gap is what RenewIQ fills.

---

## 1. Problem Statement

### The Manual Status Quo

A corporate energy manager at a Dutch manufacturing company has signed a 15-year physical PPA with a wind farm in Zeeland at a fixed strike price of €68/MWh. On a Sunday in March 2026, EPEX SPOT NL day-ahead prices go **negative** (−€42/MWh) for 6 consecutive hours. The questions the energy manager now needs to answer — manually, today, by hunting through a 120-page PDF:

- Does our contract have a **negative price floor clause**? (Section 8.4, possibly)
- Are we obligated to **take-or-pay** during negative price hours? (Clause 12.2)
- What is our **imbalance exposure** if we deviate from our scheduled consumption?
- Is the wind farm under a **curtailment order** from TenneT right now? (GOPACS data)
- What is the **total financial exposure** over this 6-hour window?

Currently this takes 2–4 hours of manual cross-referencing. For a portfolio of 5 PPAs across different wind/solar assets, this is a full-time job. RenewIQ answers all of this in under 30 seconds.

### The User Personas

| Persona | Job | Pain |
|---|---|---|
| **Corporate Energy Manager** | Manages company's electricity procurement and sustainability targets | Buried in contract PDFs; can't monitor live market risk in real-time |
| **PPA Developer / Originator** | Structures and sells PPAs at renewable energy companies | Manually reviews competitor contract terms; slow risk assessments |
| **Energy Trader** | Manages short-term portfolio balancing | Can't quickly query contract constraints during fast market movements |
| **Legal / Compliance Officer** | Reviews PPA terms for regulatory changes | Manual clause extraction across dozens of contracts |

---

## 2. System Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RenewIQ Copilot                                │
│                     Conversational + Agentic Layer                       │
│                                                                          │
│   "What is our financial exposure if EPEX NL goes negative tonight?"     │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    Orchestrator Agent                              │   │
│  │  LangGraph StateGraph — routes to 1–N specialist agents           │   │
│  │  Tracks state: {contracts_loaded, market_context, risk_flags}     │   │
│  └──────┬─────────────────┬──────────────────┬──────────────────────┘   │
│         │                 │                  │                           │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌────────▼─────────┐               │
│  │  Contract   │  │  Market      │  │  Risk Scoring    │               │
│  │  RAG Agent  │  │  Data Agent  │  │  Agent           │               │
│  │             │  │              │  │                  │               │
│  │  Extracts   │  │  Fetches     │  │  Calculates      │               │
│  │  clauses,   │  │  live EPEX,  │  │  financial       │               │
│  │  obligations│  │  ENTSO-E,    │  │  exposure from   │               │
│  │  risk terms │  │  GOPACS      │  │  contract +      │               │
│  │  from PDFs  │  │  signals     │  │  market data     │               │
│  └─────────────┘  └──────────────┘  └──────────────────┘               │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Report Generation Agent                         │  │
│  │  Synthesises all agent outputs → Markdown + PDF report            │  │
│  │  Two formats: Executive (non-technical) + Legal (clause refs)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
          │                                              │
┌─────────▼───────────┐                    ┌────────────▼──────────────┐
│    Data Layer        │                    │   Infra / Ops Layer       │
│                      │                    │                           │
│  PostgreSQL +        │                    │  Azure Kubernetes (AKS)   │
│  pgvector            │                    │  Terraform IaC            │
│  (contract chunks)   │                    │  GitHub Actions CI/CD     │
│                      │                    │                           │
│  TimescaleDB         │                    │  LangSmith (LLM traces)   │
│  (EPEX, ENTSO-E,     │                    │  Langfuse (open-source    │
│   GOPACS time-series)│                    │  fallback)                │
│                      │                    │                           │
│  Azure Blob          │                    │  Prometheus + Grafana     │
│  (raw PPA PDFs,      │                    │  RAGAS evaluation suite   │
│   regulatory docs)   │                    │  MLflow model registry    │
└──────────────────────┘                    └───────────────────────────┘
```

---

## 3. Data Sources (All Public / Open)

This is a fully real, runnable project — every data source is publicly accessible.

| Data Source | What It Provides | Access Method |
|---|---|---|
| **EPEX SPOT NL** | Day-ahead electricity prices (€/MWh per hour), Netherlands | Stekker API (free, open) · goEpexSpot Go client |
| **ENTSO-E Transparency Platform** | Generation mix (wind/solar/gas), cross-border flows, NL load | REST API (free, registration required) |
| **GOPACS** | Grid congestion announcements, redispatch volumes, congestion areas per DSO zone | GOPACS website + market announcements via API |
| **PPA PDF Contracts** | Synthetic realistic contracts (3 types: physical PPA, virtual PPA, sleeved PPA) | Generated using open PPA templates (EIB, DLA Piper, Genie AI NL) |
| **Dutch Regulatory Docs** | Electricity Act 1998, ACM guidelines, ENTSO-E grid codes | Public government sources (overheid.nl) |

### Why Synthetic PPA Contracts Are Acceptable
Real PPA contracts are commercially sensitive. The project uses **synthetically generated but structurally realistic** PDFs based on:
- EIB Advisory Services PPA template framework
- DLA Piper international PPA risk allocation standards
- Genie AI Netherlands Corporate PPA template (Dutch Electricity Act 1998 compliant)

This is standard practice in legal AI projects and will be noted prominently in the README.

---

## 4. Agent Deep-Dive

### 4.1 Orchestrator Agent (LangGraph StateGraph)

The orchestrator is a stateful LangGraph graph. It maintains session state across turns and routes queries to specialist agents based on intent classification.

```python
# State schema
class RenewIQState(TypedDict):
    messages: Annotated[list, add_messages]
    contracts_in_scope: list[str]       # PPA IDs loaded for this session
    market_window: dict                  # {start, end, resolution}
    risk_flags: list[RiskFlag]           # accumulated risk signals
    agent_outputs: dict[str, Any]        # outputs from each specialist agent
    final_report: Optional[str]

# Routing logic
def route_intent(state: RenewIQState) -> list[str]:
    """
    Classify query and return list of agents to invoke.
    Examples:
      "What does clause 8.4 say?" → ["contract_rag"]
      "What are prices tonight?"  → ["market_data"]
      "What's our exposure?"      → ["contract_rag", "market_data", "risk_scoring"]
      "Generate report"           → ["report_generation"]
    """
```

**Graph topology:**
```
START → classify_intent → [contract_rag | market_data | risk_scoring] → aggregate → report_agent → END
                            (parallel fan-out when multiple agents needed)
```

---

### 4.2 Contract RAG Agent

**Purpose:** Extract specific clauses, obligations, and risk terms from PPA PDF contracts stored in the vector database.

#### Document Ingestion Pipeline

```
PPA PDF (Azure Blob)
    │
    ▼
PyMuPDF — extract text with section headers preserved
    │
    ▼
Section-Aware Chunker
    ├── Detects clause numbers (e.g., "8.4", "Article 12")
    ├── Chunks: 512 tokens, 64-token overlap
    └── Preserves: {clause_id, section_title, page_number, contract_id, ppa_type}
    │
    ▼
Embedding Model
    ├── Primary:   text-embedding-3-small (OpenAI)
    └── Fallback:  nomic-embed-text via Ollama (self-hosted, data sovereignty)
    │
    ▼
pgvector store (PostgreSQL extension)
    └── Metadata index on {contract_id, clause_type, risk_category}
```

#### Clause Taxonomy (Risk-Aware Metadata)

Every chunk is tagged with a `risk_category` label during ingestion:

| Risk Category | Example Clause Content |
|---|---|
| `price_risk` | Fixed strike price, price floor, negative price provisions |
| `volume_risk` | Take-or-pay obligations, minimum delivery volumes, shape risk |
| `curtailment_risk` | Curtailment compensation, proxy generation settlement |
| `basis_risk` | Delivery point specifications, balancing responsibility |
| `counterparty_risk` | Credit support annex, termination events, step-in rights |
| `legal_regulatory` | Change-in-law provisions, force majeure, governing law |

#### Retrieval Strategy

```python
def retrieve_clauses(query: str, contract_ids: list[str], 
                     risk_category: Optional[str] = None) -> list[Document]:
    """
    1. Semantic search: pgvector cosine similarity, top-k=8
    2. Metadata filter: contract_id IN (...) AND risk_category = ...
    3. Cross-encoder re-ranking: top-k=3 after rerank
    4. Self-query fallback: if relevance < 0.7, rewrite query and retry (max 3x)
    """
```

---

### 4.3 Market Data Agent

**Purpose:** Fetch, cache, and contextualise live and historical energy market data relevant to the user's query.

#### Data Fetchers

```python
class MarketDataAgent:

    def fetch_epex_dayahead(self, date: date, market: str = "NL") -> DataFrame:
        """
        Source: Stekker API / ENTSO-E Transparency Platform
        Returns: hourly {timestamp, price_eur_mwh, forecast_lower, forecast_upper}
        Caches in TimescaleDB hypertable: epex_dayahead_prices
        """

    def fetch_entso_generation(self, date: date, country: str = "NL") -> DataFrame:
        """
        Source: ENTSO-E REST API (transparency.entsoe.eu)
        Returns: hourly generation by type {wind_onshore, solar, gas, nuclear, ...}
        Detects: high renewable share → flag potential negative price risk
        """

    def fetch_gopacs_announcements(self, region: Optional[str] = None) -> list[dict]:
        """
        Source: GOPACS website + market announcements
        Returns: active congestion events {dso_zone, start, end, mw_needed, direction}
        Flags: if PPA delivery point is in congested zone → curtailment risk elevated
        """

    def detect_market_stress(self, window_hours: int = 24) -> list[MarketStressEvent]:
        """
        Combines all three sources to flag:
        - Negative price periods (EPEX < 0)
        - High curtailment risk (GOPACS congestion in PPA delivery zone)
        - Imbalance price spikes (>2x day-ahead price)
        """
```

#### TimescaleDB Schema

```sql
-- EPEX day-ahead prices hypertable
CREATE TABLE epex_dayahead (
    timestamp   TIMESTAMPTZ NOT NULL,
    market      TEXT NOT NULL,          -- 'NL', 'BE', 'DE'
    price_eur_mwh NUMERIC(10,4),
    is_negative BOOLEAN GENERATED ALWAYS AS (price_eur_mwh < 0) STORED
);
SELECT create_hypertable('epex_dayahead', 'timestamp');

-- GOPACS congestion events
CREATE TABLE gopacs_congestion (
    event_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dso_zone    TEXT,                   -- 'Liander-Noord', 'Enexis-Zuid', etc.
    direction   TEXT,                   -- 'upward' / 'downward'
    start_time  TIMESTAMPTZ,
    end_time    TIMESTAMPTZ,
    mw_needed   NUMERIC(8,2),
    price_eur_mwh NUMERIC(10,4)
);
```

---

### 4.4 Risk Scoring Agent

**Purpose:** Cross-reference contract clause outputs from the RAG agent with market data outputs to produce a quantified, actionable risk assessment.

#### Risk Matrix

```python
@dataclass
class RiskFlag:
    risk_category: str          # from PPA taxonomy above
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    contract_clause: str        # e.g., "Article 8.4 — Negative Price Provisions"
    market_trigger: str         # e.g., "EPEX NL: -€42/MWh at 2026-03-15 11:00"
    financial_exposure_eur: float
    recommendation: str
    confidence: float           # 0.0–1.0, from RAG retrieval score

def score_negative_price_exposure(
    contract_clause: Document,
    epex_prices: DataFrame,
    ppa_volume_mw: float,
    strike_price_eur: float
) -> RiskFlag:
    """
    If contract has no negative price floor AND EPEX < 0:
    Exposure = Σ (strike_price - epex_price) × volume_mwh per negative hour
    """
    negative_hours = epex_prices[epex_prices["price_eur_mwh"] < 0]
    exposure = ((strike_price_eur - negative_hours["price_eur_mwh"]) 
                * ppa_volume_mw).sum()
    ...
```

#### The Six Risk Checks

| # | Risk Check | Data Inputs | Output |
|---|---|---|---|
| 1 | **Negative price exposure** | Contract (negative price clause) + EPEX prices | € exposure over negative window |
| 2 | **Take-or-pay obligation** | Contract (take-or-pay clause) + volume forecast | Minimum payment obligation |
| 3 | **Curtailment risk** | Contract (curtailment clause) + GOPACS congestion zone | Uncompensated volume estimate |
| 4 | **Imbalance exposure** | Contract (balancing responsibility) + imbalance price | Cost of schedule deviation |
| 5 | **Basis risk** | Contract (delivery point) + ENTSO-E cross-border flow data | Price differential exposure |
| 6 | **Regulatory change flag** | Contract (change-in-law clause) + regulatory doc RAG | Qualitative flag + clause reference |

---

### 4.5 Report Generation Agent

**Purpose:** Synthesise all agent outputs into two report variants — one for commercial stakeholders, one for legal/compliance review.

#### Report Structure

```markdown
# PPA Risk Assessment Report
Generated: {timestamp} | Contract: {ppa_id} | Market Window: {start}–{end}

## Executive Summary
[3-bullet non-technical summary for C-suite / sustainability manager]

## Active Risk Flags
| Risk | Severity | Financial Exposure | Contract Clause | Action Required |
|------|----------|--------------------|-----------------|-----------------|
| Negative price exposure | HIGH | €14,280 | Article 8.4 | Review floor clause |
| GOPACS congestion — Zeeland zone | MEDIUM | €3,100 est. | Clause 11.2 | Monitor curtailment |

## Contract Clause Detail
[Verbatim extracted clause with page reference + interpretation]

## Market Data Context
[EPEX price chart reference + GOPACS congestion event summary]

## Recommended Actions (Priority Order)
1. [Immediate action with contract clause and deadline]
2. [Medium-term action]
3. [Legal review recommendation]

## Appendix
- Raw EPEX data (CSV)
- GOPACS congestion event details
- Full clause extraction with confidence scores
```

---

## 5. Example Query Flow

**User:** *"We have a physical PPA with WindPark Zeeland. EPEX prices are negative right now. What is our exposure and what does our contract say about this?"*

```
User Query
    │
    ▼
Orchestrator Agent
    │── Intent: [contract_rag, market_data, risk_scoring, report_generation]
    │── Parallel fan-out:
    │
    ├─► Contract RAG Agent
    │       └── Query: "negative price provisions take-or-pay WindPark Zeeland"
    │           ├── Retrieved: Article 8.4 — "No price floor applies. Offtaker
    │           │             obligated to pay strike price regardless of market price."
    │           ├── Retrieved: Clause 12.1 — "Take-or-pay: minimum 85% of forecasted
    │           │             monthly generation volume"
    │           └── Returns: {clauses: [...], risk_categories: ["price_risk", "volume_risk"]}
    │
    ├─► Market Data Agent
    │       └── fetch_epex_dayahead(today, "NL")
    │           ├── Hours 11:00–16:00: prices range from -€18 to -€42/MWh
    │           └── Returns: {negative_hours: 5, avg_negative_price: -€28/MWh}
    │
    └─► Risk Scoring Agent (waits for both above)
            ├── No negative price floor found in Article 8.4
            ├── Assumed volume: 12 MW (from contract header metadata)
            ├── Exposure = (€68 strike − (−€28 avg)) × 12 MW × 5 hrs
            │           = €96/MWh × 12 MW × 5 hrs = €5,760
            └── Returns: RiskFlag(severity="HIGH", exposure=5760.00, 
                                  clause="Article 8.4", 
                                  recommendation="Negotiate negative price floor 
                                  in next contract review. Consider vPPA structure.")

Report Generation Agent
    └── Produces: Markdown report + PDF via WeasyPrint
        + CSV of hourly exposure breakdown
```

---

## 6. Repository Structure

```
renewiq/
├── README.md                          ← Problem, architecture, demo GIF
├── docs/
│   ├── system-design.md               ← This document
│   ├── data-dictionary.md             ← PPA, EPEX, GOPACS, ENTSO-E glossary
│   ├── adr/
│   │   ├── ADR-001-langgraph-vs-crewai.md
│   │   ├── ADR-002-pgvector-vs-qdrant.md
│   │   ├── ADR-003-timescaledb-for-market-data.md
│   │   ├── ADR-004-synthetic-ppa-contracts.md
│   │   └── ADR-005-self-hosted-embeddings.md
│   └── diagrams/
│       ├── architecture.png
│       └── agent-flow.png
├── src/
│   ├── orchestrator/
│   │   ├── graph.py                   ← LangGraph StateGraph definition
│   │   ├── state.py                   ← RenewIQState TypedDict
│   │   ├── router.py                  ← Intent classification → agent routing
│   │   └── prompts.py                 ← All system/user prompt templates
│   ├── agents/
│   │   ├── contract_rag/
│   │   │   ├── agent.py
│   │   │   ├── retriever.py           ← pgvector semantic + metadata retrieval
│   │   │   ├── reranker.py            ← Cross-encoder reranking
│   │   │   └── ingestion/
│   │   │       ├── pdf_parser.py      ← PyMuPDF section-aware chunking
│   │   │       ├── embedder.py        ← OpenAI / Ollama embedding
│   │   │       └── clause_tagger.py   ← Risk category metadata tagging
│   │   ├── market_data/
│   │   │   ├── agent.py
│   │   │   ├── epex_fetcher.py        ← Stekker API + ENTSO-E client
│   │   │   ├── gopacs_fetcher.py      ← GOPACS congestion announcements
│   │   │   └── stress_detector.py    ← Negative price / congestion flags
│   │   ├── risk_scoring/
│   │   │   ├── agent.py
│   │   │   ├── risk_models.py         ← RiskFlag dataclass + 6 risk calculators
│   │   │   └── financial_calc.py      ← Exposure quantification formulas
│   │   └── report_generation/
│   │       ├── agent.py
│   │       ├── templates/
│   │       │   ├── executive_report.md.jinja
│   │       │   └── legal_report.md.jinja
│   │       └── pdf_exporter.py        ← WeasyPrint Markdown → PDF
│   ├── data/
│   │   ├── schemas/                   ← Pydantic models for all data types
│   │   ├── db/
│   │   │   ├── migrations/            ← Alembic SQL migrations
│   │   │   └── seed/                  ← Demo data seeding scripts
│   │   └── synthetic_ppa/
│   │       ├── generator.py           ← Synthetic PPA PDF generator
│   │       └── templates/             ← Physical / virtual / sleeved PPA templates
│   └── api/
│       ├── main.py                    ← FastAPI gateway
│       ├── routes/
│       │   ├── chat.py                ← POST /chat — conversational interface
│       │   ├── contracts.py           ← POST /contracts/upload, GET /contracts
│       │   └── reports.py             ← GET /reports/{id}/pdf
│       └── middleware/
│           └── auth.py                ← API key auth
├── infra/
│   ├── terraform/
│   │   ├── main.tf                    ← AKS, ACR, Key Vault, Blob Storage
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── kubernetes/
│   │   ├── namespaces.yaml
│   │   ├── deployments/
│   │   │   ├── orchestrator-api.yaml
│   │   │   ├── contract-rag-worker.yaml
│   │   │   ├── market-data-worker.yaml
│   │   │   └── report-generator.yaml
│   │   ├── services/
│   │   └── configmaps/
│   └── monitoring/
│       ├── prometheus/
│       └── grafana/
│           └── dashboards/
│               ├── agent-latency.json
│               ├── rag-quality.json
│               └── market-data-freshness.json
├── tests/
│   ├── unit/
│   │   ├── test_risk_calculators.py
│   │   ├── test_clause_tagger.py
│   │   └── test_market_fetchers.py
│   ├── integration/
│   │   ├── test_contract_rag_pipeline.py
│   │   └── test_agent_orchestration.py
│   ├── e2e/
│   │   └── test_full_query_flow.py
│   └── ragas/
│       └── evaluate_rag_quality.py    ← RAGAS faithfulness + relevancy eval
├── notebooks/
│   ├── 01_epex_eda.ipynb              ← Negative price patterns analysis
│   ├── 02_ppa_clause_taxonomy.ipynb   ← Building the risk category taxonomy
│   ├── 03_rag_evaluation.ipynb        ← RAGAS metrics baseline
│   └── 04_risk_exposure_calc.ipynb    ← Exposure formula validation
├── scripts/
│   ├── seed_market_data.py            ← Pull 90 days EPEX + ENTSO-E + GOPACS
│   ├── generate_synthetic_ppas.py     ← Create 5 demo PPA PDFs
│   └── ingest_contracts.py            ← Chunk + embed + upsert to pgvector
├── .github/
│   └── workflows/
│       ├── ci.yml                     ← Lint + Test + RAGAS eval gate
│       └── cd.yml                     ← Build → Push ACR → Deploy AKS
├── docker-compose.yml                 ← Local dev: PostgreSQL+pgvector+Timescale
└── pyproject.toml
```

---

## 7. Architecture Decision Records (ADRs)

### ADR-001: LangGraph over CrewAI
**Decision:** LangGraph for agent orchestration.  
**Rationale:** LangGraph's explicit StateGraph model gives precise control over multi-agent routing, retry logic, and partial failure handling. In a financial/legal context, deterministic agent behaviour and full trace auditability (via LangSmith) are non-negotiable. CrewAI's implicit orchestration sacrifices this control for ease-of-use.

### ADR-002: pgvector over Qdrant
**Decision:** pgvector (PostgreSQL extension) as the vector store.  
**Rationale:** All structured market data (EPEX prices, GOPACS congestion, contract metadata) already lives in PostgreSQL/TimescaleDB. Co-locating vector search in the same database cluster enables SQL JOINs between semantic retrieval results and structured data — e.g., "retrieve clauses tagged price_risk AND filter to contracts with delivery points in active GOPACS congestion zones." A separate Qdrant cluster cannot perform this cross-store join.

### ADR-003: TimescaleDB for Market Time-Series
**Decision:** TimescaleDB (PostgreSQL extension) for EPEX, ENTSO-E, GOPACS time-series.  
**Rationale:** TimescaleDB's hypertable partitioning handles high-frequency time-series (hourly EPEX data, 15-min GOPACS updates) with native time-bucket aggregation functions. Keeps the entire data layer in one PostgreSQL cluster — same database, two extensions (pgvector + TimescaleDB), zero additional infra.

### ADR-004: Synthetic PPA Contracts
**Decision:** Use synthetically generated but structurally realistic PPA PDFs.  
**Rationale:** Real PPA contracts are commercially sensitive and not publicly available. Synthetic contracts based on EIB, DLA Piper, and ACM templates faithfully represent the risk clause structures that matter for this system. This is explicitly noted in the README and is standard practice in legal NLP research.

### ADR-005: Self-Hosted Embeddings via Ollama
**Decision:** `nomic-embed-text` via Ollama as the fallback embedding model for contract content.  
**Rationale:** PPA contracts may contain commercially sensitive pricing terms. Self-hosted embeddings ensure no contract content is sent to third-party embedding APIs. Primary path uses OpenAI `text-embedding-3-small` with data classified as non-sensitive; sensitive contract ingestion routes through Ollama. This mirrors the Enjins pattern deployed for Next Sense on AKS.

---

## 8. Observability & LLMOps Stack

| Concern | Tool | What's Tracked |
|---|---|---|
| LLM traces | LangSmith | Every prompt, completion, tool call, agent routing decision, latency |
| RAG quality | RAGAS | Faithfulness, answer relevancy, context precision per query |
| Model drift | Evidently AI | Embedding drift, retrieval quality degradation over time |
| Infrastructure | Prometheus + Grafana | API latency P50/P95/P99, pod health, DB query times |
| Data freshness | Custom Grafana dashboard | Age of latest EPEX, ENTSO-E, GOPACS data per source |
| CI quality gate | GitHub Actions | RAGAS faithfulness > 0.85 required to merge to main |
| Feedback loop | PostgreSQL table | User thumbs up/down + optional comment per response |

---

## 9. Evaluation Targets

### RAG Quality (RAGAS Framework)
| Metric | Target | Meaning |
|---|---|---|
| **Faithfulness** | > 0.88 | Agent answers are grounded in retrieved contract clauses |
| **Answer Relevancy** | > 0.85 | Retrieved clauses are relevant to the query |
| **Context Precision** | > 0.80 | No hallucinated clause references |

### System Performance
| Metric | Target |
|---|---|
| End-to-end query latency (P95) | < 20 seconds |
| EPEX data freshness | < 15 minutes lag |
| GOPACS congestion update lag | < 30 minutes |
| Contract ingestion throughput | ≥ 50 pages/min |

---

## 10. Demo Scenarios for GitHub README

| Scenario | User Query | System Response |
|---|---|---|
| **Negative price alert** | "EPEX is negative right now. What's our exposure on the Zeeland PPA?" | Risk flag (HIGH), €5,760 exposure, Article 8.4 reference, PDF report |
| **Contract clause lookup** | "Does our vPPA have a zero-floor price protection?" | Verbatim clause extraction with page ref + plain-language interpretation |
| **Portfolio scan** | "Which of our 3 PPAs has the most curtailment risk given today's GOPACS data?" | Ranked risk table across all contracts |
| **Regulatory check** | "Has the Dutch ACM issued any new guidelines that affect our balancing obligation?" | RAG over regulatory docs + contract change-in-law clause + gap analysis |
| **Report generation** | "Generate a risk report for Q2 2026 for all contracts" | Quarterly risk summary PDF covering all 6 risk categories |

---

## 11. Skills Demonstrated

| Enjins Requirement | How RenewIQ Demonstrates It |
|---|---|
| **LLM & Agentic systems** | LangGraph multi-agent: Orchestrator + 4 specialist agents with parallel fan-out |
| **RAG pipelines** | pgvector + cross-encoder reranking + self-correcting retrieval (3-retry loop) |
| **Production mindset** | RAGAS CI gate, LangSmith traces, Grafana dashboards, feedback loops |
| **MLOps / LLMOps** | LangSmith + Evidently + GitHub Actions CI/CD pipeline |
| **Cloud (Azure)** | AKS deployment with full Terraform IaC |
| **Python + SQL** | FastAPI, LangChain, TimescaleDB SQL, pgvector |
| **Docker / Kubernetes** | Full AKS manifest set + docker-compose for local dev |
| **Data engineering background** | Market data ingestion pipelines (EPEX, ENTSO-E, GOPACS) — mirrors SAS/Databricks migration experience |
| **Climate / Energy domain** | Directly relevant to Enjins' energy client portfolio (Groendus) |
| **Client communication** | README written for both technical engineers AND non-technical energy managers |

---

## 12. Getting Started

```bash
# Clone
git clone https://github.com/venkatesanmariappan/renewiq
cd renewiq

# Start local data stack (PostgreSQL + pgvector + TimescaleDB + Redis)
docker compose up -d

# Run DB migrations
alembic upgrade head

# Seed 90 days of EPEX NL + ENTSO-E NL market data
python scripts/seed_market_data.py --days 90 --market NL

# Generate 5 synthetic PPA contracts (physical, virtual, sleeved)
python scripts/generate_synthetic_ppas.py

# Ingest contracts into pgvector
python scripts/ingest_contracts.py --source data/synthetic_ppa/

# Start the API
uvicorn src.api.main:app --reload --port 8000

# Try a query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our exposure if EPEX goes negative today?", 
       "contracts": ["zeeland-wind-ppa-v1"]}'

# Open Swagger UI
open http://localhost:8000/docs
```

---

## Appendix A: Full Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Orchestration | LangGraph 0.3 | Stateful multi-agent StateGraph |
| LLM Backend | GPT-4o / Ollama (llama3.1) | Reasoning + clause interpretation |
| Embeddings | text-embedding-3-small / nomic-embed-text | Document + query embedding |
| Vector Store | pgvector (PostgreSQL extension) | Semantic contract clause retrieval |
| Time-Series DB | TimescaleDB (PostgreSQL extension) | EPEX, ENTSO-E, GOPACS market data |
| Re-ranking | FlashRank (cross-encoder, local) | Improve RAG precision |
| PDF Parsing | PyMuPDF | Section-aware contract chunking |
| API Gateway | FastAPI | REST + WebSocket interface |
| Task Queue | Celery + Redis | Async agent workers |
| Containerisation | Docker + Docker Compose | Local dev stack |
| Orchestration | Azure Kubernetes Service (AKS) | Production deployment |
| IaC | Terraform | Azure resource provisioning |
| CI/CD | GitHub Actions | Test → Build → Deploy pipeline |
| LLM Observability | LangSmith + Langfuse | Trace every agent call in production |
| RAG Evaluation | RAGAS | Faithfulness, relevancy, precision |
| ML/Data Drift | Evidently AI | Embedding + retrieval quality monitoring |
| Infrastructure Monitoring | Prometheus + Grafana | Latency, throughput, data freshness |
| PDF Report Export | WeasyPrint | Markdown → styled PDF |
| Object Storage | Azure Blob Storage | Raw PDFs, model artefacts |
| Container Registry | Azure Container Registry | Docker image storage |
| Secrets Management | Azure Key Vault | API keys, DB credentials |

---

*RenewIQ is designed to be fully implemented as a working GitHub repository. Every data source is public, every tool is open-source, and the architecture demonstrates exactly the production-grade agentic AI engineering Enjins builds for their climate and energy clients.*
