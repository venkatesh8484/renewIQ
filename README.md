# RenewIQ — Renewable Energy PPA Intelligence Copilot

> **Multi-agent LLM system** that cross-references live energy market data with PPA contract clauses to surface financial risk in real time — built on Azure Databricks Lakehouse.

---

## The Problem

A corporate energy manager has signed a 15-year physical PPA at **€68/MWh**. On a Sunday morning, EPEX NL spot prices go **negative (−€42/MWh)** for 6 hours.

**Questions that need answering — manually, today, by hunting through a 120-page PDF:**
- Does our contract have a negative price floor clause?
- Are we obligated to take-or-pay during negative price hours?
- Is the wind farm under a curtailment order from TenneT right now?
- What is our total financial exposure over this window?

This takes **2–4 hours per PPA**. For a portfolio of 5 contracts, it's a full-time job.

**RenewIQ answers all of this in under 30 seconds.**

---

## Architecture

```
User Query
    │
    ▼
FastAPI Gateway (Docker / AKS)
    │
    ▼
Databricks Model Serving Endpoint
    │
    ▼
LangGraph Orchestrator (StateGraph)
    │
    ├──► Contract RAG Agent     → Databricks Vector Search over Silver Delta table
    ├──► Market Data Agent      → Gold Delta tables (EPEX, GOPACS, ENTSO-E)
    ├──► Risk Scoring Agent     → Deterministic Python calculators (no LLM for €)
    └──► Report Generation      → Markdown + WeasyPrint PDF
    │
    ▼
Medallion Data Pipeline (Lakeflow DLT)
    Bronze (raw) → Silver (validated) → Gold (agent-ready features)
```

Full system design: [`docs/system-design.md`](docs/system-design.md)

---

## Demo Scenarios

| Query | Response |
|-------|----------|
| *"EPEX is negative right now. What's our Zeeland exposure?"* | HIGH risk flag, €5,760 exposure, Article 8.4 reference, PDF report |
| *"Does our vPPA have a zero-floor price protection?"* | Verbatim clause extraction with page ref + plain-language interpretation |
| *"Which of our 3 PPAs has the most curtailment risk today?"* | Ranked risk table across all contracts vs live GOPACS congestion |
| *"Generate a Q2 risk report for all contracts"* | Quarterly PDF covering all 6 risk categories |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 0.3 (StateGraph, parallel fan-out) |
| LLM | DBRX via Databricks AI Gateway / Ollama llama3.1:8b (local) |
| Embeddings | nomic-embed-text via Ollama / Databricks Model Serving |
| Vector Search | Databricks Vector Search (Delta Sync index, Unity Catalog) |
| Data Pipeline | Databricks Lakeflow DLT — Bronze → Silver → Gold |
| Governance | Unity Catalog (lineage, access control, tool registry) |
| LLMOps | MLflow 3 (LangGraph autolog), RAGAS eval CI gate |
| API | FastAPI + Redis cache |
| Containers | Docker (multi-stage), Docker Compose (local dev) |
| Kubernetes | Helm chart for AKS (FastAPI gateway) |
| CI/CD | GitHub Actions — lint → test → Docker build → Helm deploy |
| Data Sources | EPEX SPOT NL, ENTSO-E, GOPACS (all public APIs) |

---

## Quickstart — Local Development

**Prerequisites:** Python 3.11+, Docker, [Ollama](https://ollama.com) with `nomic-embed-text` and `llama3.1:8b` pulled.

```bash
# 1. Clone
git clone https://github.com/venkatesanmariappan/renewiq
cd renewiq

# 2. Configure environment
cp .env.example .env
# Edit .env — add your Databricks credentials (optional for local dev)

# 3. Start local stack
docker compose up

# 4. Verify
curl http://localhost:8000/health
# → {"status": "ok", "llm_backend": "ollama", "mock_mode": false}

# 5. Open Swagger UI
open http://localhost:8000/docs

# 6. Send a test query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our exposure if EPEX goes negative tonight?", "contracts": ["zeeland-wind-ppa-v1"]}'
```

**Offline mode (no Databricks, no Ollama):**
```bash
USE_MOCK_ENDPOINT=true docker compose --profile offline up
```

---

## Quickstart — Databricks (Production)

```bash
# Install Databricks CLI
pip install databricks-cli
databricks configure --token

# Deploy workspace resources
databricks bundle deploy --target dev

# Seed 90 days of market data into Bronze
python scripts/seed_market_data.py --days 90 --market NL

# Generate 5 synthetic PPA contracts
python scripts/generate_synthetic_ppas.py

# Run Lakeflow pipeline (Bronze → Silver → Gold)
databricks pipelines start --pipeline-id <pipeline-id>

# Deploy agent to Model Serving
databricks notebooks run databricks/notebooks/04_deploy_agent.py
```

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 — Foundation | ✅ Complete | Repo scaffold, Docker, FastAPI, CI/CD |
| Phase 2 — Data Layer | 🔄 In progress | Medallion pipeline (Bronze→Silver→Gold) |
| Phase 3 — Contract Layer | ⏳ Pending | PDF ingestion, Vector Search |
| Phase 4 — Agent Layer | ⏳ Pending | LangGraph orchestrator + 4 agents |
| Phase 5 — Serving & API | ⏳ Pending | MLflow, Model Serving, FastAPI wiring |
| Phase 6 — Evaluation | ⏳ Pending | RAGAS gate, EDA notebooks, demo GIF |

---

## Data Sources

All data sources are public and free to access:

| Source | Data | Access |
|--------|------|--------|
| [Stekker API](https://stekker.app) | EPEX NL day-ahead prices | Free, no auth |
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | Generation mix, cross-border flows | Free registration |
| [GOPACS](https://www.gopacs.eu) | Grid congestion announcements | Public |
| Synthetic PPA PDFs | Structurally realistic contracts | Generated (EIB/DLA Piper templates) |

PPA contracts are **synthetically generated** based on EIB Advisory Services, DLA Piper, and ACM Netherlands templates. Real contracts are commercially sensitive and not publicly available. This is standard practice in legal NLP research.

---

## Running Tests

```bash
# Unit tests (no external dependencies — runs in CI)
pytest tests/unit/ -m "not integration" -v

# Integration tests (requires Ollama running)
pytest tests/integration/ -v

# RAGAS evaluation (requires Databricks)
python tests/evaluation/ragas_eval.py
```

---

## Repository Structure

```
renewiq/
├── src/
│   ├── api/              ← FastAPI gateway (routes, config, middleware)
│   ├── agents/           ← LangGraph orchestrator + 4 specialist agents
│   └── ingestion/        ← EPEX, ENTSO-E, GOPACS fetchers + PDF parser
├── databricks/
│   ├── pipelines/        ← Lakeflow DLT (Bronze → Silver → Gold)
│   ├── notebooks/        ← Setup, deploy, evaluate
│   └── bundle/           ← Databricks Asset Bundle (IaC)
├── infra/helm/           ← Helm chart for FastAPI on AKS
├── tests/                ← Unit, integration, evaluation (RAGAS)
├── scripts/              ← Seed market data, generate PPAs, ingest contracts
├── docker/               ← Dockerfile (API + mock Databricks stub)
├── docs/adr/             ← Architecture Decision Records
└── .github/workflows/    ← CI (lint + test + Docker build) + CD (AKS deploy)
```

---

*Built as a portfolio project to demonstrate production-grade AI engineering: multi-agent systems, RAG pipelines, LLMOps, and end-to-end MLOps on Azure Databricks — applied to the Climate Tech / renewable energy domain.*
