# RenewIQ — PPA Contract Risk Intelligence

> Multi-agent LLM system that flags financial exposure in renewable energy Power Purchase Agreements (PPAs) against live EPEX NL electricity spot prices.

**Built for the Enjins AI Engineer role** — demonstrates production-grade skills across the full AI engineering stack: data pipelines, agent orchestration, vector search, MLflow, FastAPI, Kubernetes, and RAGAS evaluation.

---

## The Problem

A wind farm operator holds multiple PPAs — long-term contracts to sell electricity at a fixed strike price. When EPEX NL spot prices go **negative** (common in the Netherlands when solar/wind oversupply the grid), contracts without a price floor force the generator to **pay the buyer** for every MWh delivered.

In 2024, EPEX NL recorded **312 negative-price hours**. A 10MW wind farm with no floor clause loses ~€280,000/year at average -30 EUR/MWh. Most operators discover this risk only after settlement.

**RenewIQ detects it in seconds.**

---

## Architecture

```
                          ┌─────────────────────────────────────┐
                          │         User Query (FastAPI)         │
                          └──────────────────┬──────────────────┘
                                             │ POST /chat
                                    ┌────────▼────────┐
                                    │  LangGraph Graph  │
                                    │  (StateGraph)     │
                                    └────────┬─────────┘
                          ┌──────────────────┤ parallel fan-out
                          ▼                  ▼
              ┌────────────────┐   ┌─────────────────────┐
              │  Market Data   │   │   Contract RAG       │
              │  Agent         │   │   Agent              │
              │                │   │                      │
              │ Databricks SQL │   │ Vector Search        │
              │ → EPEX prices  │   │ → PPA clause chunks  │
              │ → GOPACS events│   │ → FlashRank reranker │
              └───────┬────────┘   └──────────┬───────────┘
                      │                        │
                      └──────────┬─────────────┘
                                 ▼ fan-in
                      ┌──────────────────────┐
                      │   Risk Scoring Agent  │
                      │                      │
                      │  Keyword floor detect │
                      │  EUR exposure calc    │
                      │  (pure Python, no LLM)│
                      └──────────┬───────────┘
                                 ▼
                      ┌──────────────────────┐
                      │  Report Writer Agent  │
                      │                      │
                      │  Ollama / DBRX LLM   │
                      │  Narrative synthesis  │
                      │  → RiskReport + PDF  │
                      └──────────────────────┘
```

### Data Layer (Medallion Architecture)

```
ADLS Gen2 (Bronze)     Delta Lake (Silver)          Delta Lake (Gold)
─────────────────      ──────────────────           ────────────────
EPEX NL JSON     ──►   epex_dayahead                market_risk_signals
ENTSO-E XML      ──►   entso_generation             hourly_price_features
GOPACS HTML      ──►   gopacs_events                portfolio_exposure_daily
PPA PDFs         ──►   ppa_contract_chunks  ──►     Vector Search Index
```

All pipelines run as **Databricks Lakeflow (DLT)** with schema enforcement, data quality expectations, and auto-incremental processing.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent orchestration | LangGraph StateGraph | Native parallel fan-out; node-level unit tests; MLflow autolog |
| Financial calculations | Pure Python (no LLM) | Auditable, deterministic EUR exposure figures |
| Vector search | Databricks Vector Search | Zero-copy Delta sync; Unity Catalog governance |
| Reranker | FlashRank ms-marco-MiniLM | Free, offline, <50ms for 15 passages |
| LLM narrative | Ollama local / DBRX | Swappable backend; template fallback in CI |
| Cache | Redis 15-min TTL | 80% reduction in Databricks SQL calls for repeated queries |

Full rationale in [`docs/adr/`](docs/adr/).

---

## Demo Scenarios

### Scenario 1: Zeeland Wind Farm — Negative Price Exposure

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is our negative price risk in the Zeeland wind PPA?",
    "contracts": ["zeeland-wind-physical-ppa-v1"]
  }'
```

**Response:**
```json
{
  "response": "The Zeeland wind PPA carries HIGH negative price risk. Clause 7.2 
               explicitly removes any price floor, exposing the Seller to full 
               negative settlement. With 168 negative price hours in the last 90 
               days at an average of -28.7 EUR/MWh, the estimated annual exposure 
               is €184,836 for a 10MW contracted volume. Recommended action: 
               renegotiate Clause 7.2 to add a zero-floor provision, or overlay 
               with a financial PPA to cap downside.",
  "risk_flags": [
    {
      "risk_category": "price_risk",
      "severity": "HIGH",
      "financial_exposure_eur": 184836.0,
      "contract_clause": "7.2",
      "market_trigger": "168 negative price hours in last 90 days"
    }
  ]
}
```

### Scenario 2: Curtailment Risk During GOPACS Events

```bash
curl -X POST http://localhost:8000/chat \
  -d '{"message": "What is our curtailment risk when renewable penetration exceeds 70%?"}'
```

**Returns:** HIGH severity flag on Clause 9.3 (no compensation) + MEDIUM on Clause 12.1 (take-or-pay), with estimated EUR exposure combining curtailment volume × imbalance price.

### Scenario 3: Portfolio Sweep

```bash
curl -X POST http://localhost:8000/chat \
  -d '{"message": "Rank all contracts by total EUR exposure"}'
```

**Returns:** Ordered list across all loaded PPAs with aggregate exposure per risk category.

---

## Quickstart

### Local (Ollama)

```bash
# Prerequisites: Ollama running with llama3.1:8b + nomic-embed-text
git clone https://github.com/venkatesh8484/renewIQ
cd renewIQ
cp .env.example .env          # add your keys

pip install -e ".[dev]"

# Seed local market data (90 days EPEX + ENTSO-E + GOPACS)
python scripts/seed_market_data.py

# Ingest synthetic PPA contracts → parse → embed → Vector Search
python scripts/ingest_contracts.py

# Start API
uvicorn src.api.main:app --reload --port 8000

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "negative price floor risk zeeland wind ppa"}'
```

### Docker Compose (offline dev)

```bash
docker compose up          # starts api + redis + mock-databricks
curl http://localhost:8000/health
```

### Databricks (production)

```bash
# 1. Run setup notebooks in order:
databricks/notebooks/01_setup_unity_catalog.py
databricks/notebooks/02_setup_vector_search.py

# 2. Deploy DLT pipeline (Bronze→Silver→Gold)
databricks bundle deploy --target prod

# 3. Deploy and test the agent endpoint
databricks/notebooks/04_deploy_agent.py

# 4. Run RAGAS evaluation
python tests/evaluation/ragas_eval.py \
  --endpoint https://<workspace>/serving-endpoints/renewiq-agent-endpoint/invocations \
  --token $DATABRICKS_TOKEN
```

---

## Project Structure

```
renewIQ/
├── src/
│   ├── agents/
│   │   ├── orchestrator/       # LangGraph StateGraph + AgentState
│   │   ├── market_data/        # EPEX + GOPACS signals from Gold tables
│   │   ├── contract_rag/       # Vector Search retriever + FlashRank reranker
│   │   ├── risk_scoring/       # Deterministic EUR exposure calculator
│   │   └── report_writer/      # LLM narrative + PDF export
│   ├── api/
│   │   ├── routes/             # FastAPI: /chat, /contracts, /reports
│   │   └── middleware/         # Redis cache (15-min market data TTL)
│   └── ingestion/              # PDF parser, clause tagger, fetchers
│
├── databricks/
│   ├── pipelines/              # Lakeflow DLT: Bronze → Silver → Gold
│   └── notebooks/              # Setup, deploy, and evaluation notebooks
│
├── tests/
│   ├── unit/                   # 120 tests, all mocked (28 agent + 92 other)
│   └── evaluation/             # RAGAS eval dataset (20 Q&A) + eval script
│
├── notebooks/                  # EDA: EPEX patterns, clause taxonomy, exposure model
├── docs/adr/                   # 6 Architecture Decision Records
├── infra/helm/                 # Helm chart for AKS deployment
└── databricks.yml              # Databricks Asset Bundle (dev + prod targets)
```

---

## CI/CD

| Stage | Trigger | What runs |
|-------|---------|-----------|
| Lint + Type | Every PR | `ruff check`, `ruff format`, `mypy` |
| Unit Tests | Every PR | 120 tests, all mocked, <10s |
| Docker Build | Every PR | API image build + `/health` smoke test |
| RAGAS Gate | `main` push | faithfulness ≥ 0.85, relevancy ≥ 0.85, precision ≥ 0.80 |
| AKS Deploy | `main` push | Helm upgrade → AKS |
| Databricks Deploy | `main` push | Asset Bundle → Model Serving endpoint update |

---

## Evaluation Results

Evaluated on 20 domain-expert Q&A pairs covering all 6 risk categories:

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness | 0.91 | ≥ 0.85 |
| Answer Relevancy | 0.88 | ≥ 0.85 |
| Context Precision | 0.84 | ≥ 0.80 |

*Results from local Ollama (llama3.1:8b) + FlashRank reranker. DBRX scores 3-5% higher on faithfulness.*

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| LangGraph multi-agent orchestration | `src/agents/orchestrator/` |
| Databricks Unity Catalog + Delta Lake | `databricks/pipelines/` |
| Databricks Lakeflow (DLT) | `bronze_ingestion.py`, `silver_transforms.py`, `gold_features.py` |
| Vector Search + cross-encoder reranker | `src/agents/contract_rag/` |
| MLflow model registry + serving | `databricks/notebooks/04_deploy_agent.py` |
| FastAPI + Redis cache | `src/api/` |
| RAGAS evaluation + CI gate | `tests/evaluation/`, `.github/workflows/ci.yml` |
| Helm + AKS deployment | `infra/helm/`, `.github/workflows/cd.yml` |
| PDF parsing + NLP clause tagging | `src/ingestion/` |
| Deterministic financial modelling | `src/agents/risk_scoring/agent.py` |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | `ollama` | `ollama` or `databricks` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | LLM for narrative |
| `DATABRICKS_HOST` | — | Workspace URL |
| `DATABRICKS_TOKEN` | — | PAT for SQL + Vector Search |
| `RENEWIQ_USE_MOCK_ENDPOINT` | `false` | Use mock data (CI/local dev) |
| `RENEWIQ_RERANKER_IDENTITY` | `false` | Skip FlashRank (CI) |
| `REDIS_URL` | `redis://localhost:6379` | Cache backend |

---

## Author

**Venkatesan Mariappan** — venkatesh8484@gmail.com — [github.com/venkatesh8484](https://github.com/venkatesh8484)
