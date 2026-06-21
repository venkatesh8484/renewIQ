# RenewIQ — Build Plan

> **Goal:** Ship a working, demo-able GitHub repo that proves production-grade AI engineering skills  
> aligned with the Enjins AI Engineer role. Every phase ends with something runnable.

---

## Guiding Principles

- **Always working.** Each phase ends with a runnable state — no "half-built" commits on main.
- **Prove > claim.** No placeholder comments. Real code, real API calls, real data.
- **README-first.** The README is the product page. Update it as you build.
- **Deterministic finance.** Financial calculations never go through LLM reasoning — always pure Python.

---

## Phase 1 — Foundation (Days 1–3)

**Goal:** Repo exists, runs locally, and the Docker story is airtight.

| Task | Deliverable | Why |
|------|-------------|-----|
| Initialize repo structure | All folders from system design v2 Section 10 | Clean scaffold signals engineering discipline |
| Write `pyproject.toml` | All dependencies pinned with version ranges | Reproducible installs |
| Write `docker/api/Dockerfile` | Multi-stage build for FastAPI service | Demonstrates Docker skills |
| Write `docker-compose.yml` | api + redis + mock-databricks (offline profile) | Local dev without spending Databricks credits |
| Write `src/api/main.py` | FastAPI app: `/health`, `/chat`, `/contracts`, `/reports` routes stubbed | API contract defined up front |
| Write `infra/helm/renewiq-api/` | Helm chart (values.yaml, templates) for AKS | Kubernetes demonstrated |
| Write `.github/workflows/ci.yml` | Lint (ruff) + type check (mypy) + unit tests on every PR | CI from day one |
| Write `.github/workflows/cd.yml` | Docker build → ACR push → Helm deploy to AKS | CD pipeline skeleton |
| Write `README.md` skeleton | Problem statement + architecture diagram + quickstart | First commit impression |

**Phase 1 exit criterion:** `docker compose up` starts the API on port 8000, `/health` returns 200.

---

## Phase 2 — Data Layer: Bronze → Silver → Gold (Days 4–8)

**Goal:** Real market data flowing through the Medallion pipeline. All tables exist in Delta Lake.

### 2a — External Data Fetchers

| Task | File | Notes |
|------|------|-------|
| EPEX NL fetcher | `src/ingestion/epex_fetcher.py` | Stekker API — free, no auth required |
| ENTSO-E fetcher | `src/ingestion/entso_fetcher.py` | REST API, free registration; returns XML, parse to DataFrame |
| GOPACS fetcher | `src/ingestion/gopacs_fetcher.py` | Scrape market announcements page |
| Fetcher tests | `tests/unit/test_fetchers.py` | Mock HTTP responses; test schema, types, edge cases (empty response, negative prices) |
| Seed script | `scripts/seed_market_data.py` | Pull 90 days EPEX + ENTSO-E + GOPACS, write to ADLS Bronze container |

### 2b — Lakeflow DLT Pipelines

| Task | File | Notes |
|------|------|-------|
| Bronze ingestion | `databricks/pipelines/bronze_ingestion.py` | Auto Loader for all sources; append-only; no transforms |
| Silver transforms | `databricks/pipelines/silver_transforms.py` | Parse JSON, validate with DLT Expectations, deduplicate |
| Gold feature tables | `databricks/pipelines/gold_features.py` | `market_risk_signals`, `hourly_price_features`, `portfolio_exposure_daily` |
| Unity Catalog setup | `databricks/notebooks/01_setup_unity_catalog.py` | Create catalog `renewiq`, schemas `bronze/silver/gold/agents/models` |
| Pipeline tests | `tests/integration/test_dlt_pipeline.py` | Validate Silver schema + DLT Expectations catch bad rows |

**Phase 2 exit criterion:** Lakeflow pipeline runs end-to-end. `SELECT * FROM renewiq.gold.market_risk_signals` returns rows with correct negative price flags.

---

## Phase 3 — Contract Layer: PDF → Vector Search (Days 9–11)

**Goal:** PPA contracts ingested, chunked, embedded, and queryable via semantic search.

| Task | File | Notes |
|------|------|-------|
| Synthetic PPA generator | `scripts/generate_synthetic_ppas.py` | Generate 5 PDFs: 2x physical PPA, 2x virtual PPA, 1x sleeved PPA. Use `reportlab` or `fpdf2`. Each PDF 30–60 pages with realistic clause numbering |
| PDF parser | `src/ingestion/pdf_parser.py` | PyMuPDF section-aware extraction; detect clause numbers ("8.4", "Article 12"); chunk to 512 tokens, 64-token overlap |
| Risk category tagger | `src/ingestion/clause_tagger.py` | Lightweight classifier: keyword match → `price_risk`, `volume_risk`, `curtailment_risk`, `basis_risk`, `counterparty_risk`, `legal_regulatory` |
| Silver DLT table | `databricks/pipelines/silver_transforms.py` | Add `ppa_contract_chunks` table using `parse_pdf_udf` |
| Vector Search setup | `databricks/notebooks/02_setup_vector_search.py` | Create endpoint `renewiq-contract-search`; create Delta Sync index over `silver.ppa_contract_chunks` |
| Reranker | `src/agents/contract_rag/reranker.py` | FlashRank cross-encoder; top-8 → top-3 |
| Ingestion script | `scripts/ingest_contracts.py` | End-to-end: generate PDFs → parse → upsert to Silver → trigger Vector Search sync |
| RAG eval dataset | `tests/evaluation/agent_eval_dataset.json` | 20 Q&A pairs: clause lookup questions with expected clause references and risk categories |

**Phase 3 exit criterion:** `query_contract_clauses("negative price floor", "zeeland-wind-ppa-v1")` returns Article 8.4 as top result with `risk_category = "price_risk"`.

---

## Phase 4 — Agent Layer: LangGraph Orchestrator (Days 12–16)

**Goal:** All four agents working. Full query flow from user question to risk flag to report.

### 4a — Unity Catalog Tool Registration

| Task | File | Notes |
|------|------|-------|
| Market signals SQL tool | `src/uc_tools/market_signals.sql` | UC function over `gold.market_risk_signals` |
| Exposure calculator UC tool | `src/uc_tools/exposure_calculator.py` | Deterministic Python function registered in UC |
| GOPACS congestion tool | `src/uc_tools/gopacs_tool.sql` | UC function over `gold.gopacs_congestion_events` (via silver) |
| Tool registration script | `src/uc_tools/register_tools.py` | Register all tools in `renewiq.agents` schema |
| UC tools notebook | `databricks/notebooks/03_register_uc_tools.py` | Wrapper notebook to run registration |

### 4b — Agent Implementations

| Task | File | Notes |
|------|------|-------|
| State schema | `src/agents/orchestrator/state.py` | `RenewIQState` TypedDict |
| Intent router | `src/agents/orchestrator/router.py` | Classify query → list of agents to invoke |
| Prompt templates | `src/agents/orchestrator/prompts.py` | System prompts for each agent node |
| LangGraph graph | `src/agents/orchestrator/graph.py` | StateGraph: START → classify → [fan-out] → aggregate → report → END |
| Contract RAG agent | `src/agents/contract_rag/agent.py` | Calls `query_contract_clauses` UC tool + reranker |
| Market data agent | `src/agents/market_data/agent.py` | Calls `get_market_signals` + `get_gopacs_congestion` UC tools |
| Risk scoring agent | `src/agents/risk_scoring/agent.py` | Calls `calculate_exposure` UC tool; all 6 risk checks |
| Risk calculators | `src/agents/risk_scoring/calculators.py` | Deterministic Python: negative price, take-or-pay, curtailment, imbalance, basis, regulatory |
| Report agent | `src/agents/report_generation/agent.py` | Jinja2 → Markdown → WeasyPrint PDF |
| Report templates | `src/agents/report_generation/templates/` | `executive_report.md.jinja` + `legal_report.md.jinja` |
| Agent integration test | `tests/integration/test_agent_orchestration.py` | End-to-end: mock UC tools + Vector Search → assert correct RiskFlag output |

**Phase 4 exit criterion:** Running the full query flow locally (with mock Databricks stub) returns a `RiskFlag(severity="HIGH", exposure=5760.00)` for the Zeeland negative price scenario.

---

## Phase 5 — MLflow, Model Serving & API (Days 17–19)

**Goal:** Agent deployed to Databricks Model Serving. FastAPI wrapper live. Fully curl-able.

| Task | File | Notes |
|------|------|-------|
| MLflow autolog setup | `src/agents/orchestrator/graph.py` | Add `mlflow.langchain.autolog()` — zero extra code |
| Agent MLflow logging | `databricks/notebooks/04_deploy_agent.py` | `mlflow.langchain.log_model(...)` + register to UC model registry |
| Model Serving deploy | `databricks/notebooks/04_deploy_agent.py` | Create serving endpoint `renewiq-agent-endpoint`, scale-to-zero enabled |
| FastAPI chat route | `src/api/routes/chat.py` | POST `/chat` → proxy to Databricks endpoint → return response |
| FastAPI contracts route | `src/api/routes/contracts.py` | POST `/contracts/upload` → trigger ingestion pipeline |
| FastAPI reports route | `src/api/routes/reports.py` | GET `/reports/{id}/pdf` → fetch from Delta table |
| Redis caching middleware | `src/api/middleware/cache.py` | Cache market data responses for 15 min |
| Update CI/CD | `.github/workflows/cd.yml` | Add Docker build + Helm deploy + Databricks Asset Bundle deploy steps |

**Phase 5 exit criterion:** `curl -X POST https://<workspace>/serving-endpoints/renewiq-agent-endpoint/invocations` returns a structured risk report with clause references.

---

## Phase 6 — Evaluation, CI Quality Gate & README (Days 20–22)

**Goal:** RAGAS CI gate passing. README is a showcase document someone wants to read.

| Task | File | Notes |
|------|------|-------|
| RAGAS evaluation suite | `tests/evaluation/ragas_eval.py` | Faithfulness > 0.85, Answer Relevancy > 0.85, Context Precision > 0.80 |
| Databricks Agent Evaluation | `databricks/notebooks/05_evaluate_agent.py` | Run eval dataset against deployed endpoint; log results to MLflow |
| CI RAGAS gate | `.github/workflows/ci.yml` | Block merge to main if RAGAS faithfulness < 0.85 |
| EDA notebooks | `notebooks/01_epex_eda.ipynb` through `04_risk_exposure_calc.ipynb` | Show your thinking on negative price patterns, clause taxonomy, RAG baseline |
| Data dictionary | `docs/data-dictionary.md` | PPA, EPEX, GOPACS, ENTSO-E glossary — shows domain knowledge |
| ADR documents | `docs/adr/ADR-001` through `ADR-006` | Write up each architecture decision record |
| Demo GIF / screenshots | `docs/demo/` | Screen-record the Zeeland negative price scenario; embed in README |
| Final README | `README.md` | Problem → Architecture diagram → 5 demo scenarios → Quickstart → Skills table |

**Phase 6 exit criterion:** CI passes on main. README stands alone — a recruiter with no context understands the problem, the solution, and the engineering depth within 3 minutes.

---

## Build Order Summary

```
Phase 1: Foundation        → Repo + Docker + CI skeleton
    ↓
Phase 2: Data Layer        → Medallion pipeline (Bronze→Silver→Gold)
    ↓
Phase 3: Contract Layer    → PDF → chunks → Vector Search
    ↓
Phase 4: Agent Layer       → LangGraph + 4 agents + UC tools
    ↓
Phase 5: Serving & API     → MLflow + Model Serving + FastAPI + Helm
    ↓
Phase 6: Quality & README  → RAGAS gate + EDA + ADRs + demo GIF
```

---

## Key Technical Decisions to Confirm Before Starting

1. **Databricks workspace access** — Do you have an Azure Databricks workspace (or trial) available? Phases 2–6 require it. Phase 1 and early Phase 3 (PDF generation) can run fully locally.

2. **LLM backend** — The design uses DBRX via Databricks AI Gateway. Do you have access to DBRX, or should we use GPT-4o-mini via OpenAI (cheaper for dev) and switch to DBRX for the README demo?

3. **Embedding model** — Design uses `nomic-embed-text` via Databricks Model Serving. For local dev with the mock stub, we'll use the same model via Ollama. Do you have Ollama installed, or should we start with OpenAI `text-embedding-3-small` and swap later?

4. **Starting point** — Recommend starting with **Phase 1** (Foundation) since it's fully local, no cloud costs, and produces the Docker/K8s proof that addresses the JD gap immediately.

---

## Cost Estimate (Azure Databricks)

| Component | Estimated Cost |
|---|---|
| Databricks (dev cluster, 8h/day × 22 days) | ~$40–60 |
| ADLS Gen2 storage | ~$2 |
| OpenAI API (dev + RAGAS evals) | ~$10–15 |
| Total | **~$55–80** |

Scale-to-zero on Model Serving keeps idle costs near zero.
