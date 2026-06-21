"""
POST /chat — conversational interface to the RenewIQ agent.

Routes:
  - mock mode       → local mock endpoint (docker compose offline profile)
  - databricks mode → Databricks Model Serving endpoint proxy
  - local mode      → runs LangGraph graph in-process (Ollama / CI mock)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Shared report store — populated by chat, read by /reports
# In production this would be a Delta table lookup; here it's an in-process dict
from src.api.routes import _report_store


class ChatRequest(BaseModel):
    message: str
    contracts: list[str] = []
    session_id: Optional[str] = None


class RiskFlag(BaseModel):
    risk_category: str
    severity: str
    financial_exposure_eur: Optional[float] = None
    contract_clause: Optional[str] = None
    market_trigger: Optional[str] = None
    recommendation: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    risk_flags: list[RiskFlag] = []
    contracts_in_scope: list[str] = []
    session_id: Optional[str] = None
    report_id: Optional[str] = None


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural-language query about PPA contracts and live market data.

    Examples:
    - "What is our exposure if EPEX NL goes negative tonight?"
    - "Does our Zeeland PPA have a negative price floor clause?"
    - "Which of our contracts has the highest curtailment risk right now?"
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[/chat] session={session_id} contracts={request.contracts} msg={request.message[:80]!r}")

    if settings.use_mock_endpoint:
        return await _call_mock_endpoint(request, session_id)

    if settings.llm_backend == "databricks":
        return await _call_databricks_endpoint(request, session_id)

    # Local path — invoke the LangGraph graph in a thread pool to avoid blocking the event loop
    return await _invoke_local_graph(request, session_id)


# ── Local graph invocation ─────────────────────────────────────────────────────

async def _invoke_local_graph(request: ChatRequest, session_id: str) -> ChatResponse:
    """Run the LangGraph graph in-process (Ollama / CI mock mode)."""
    from src.agents.orchestrator.graph import get_graph
    from src.agents.orchestrator.state import make_initial_state

    state = make_initial_state(
        query=request.message,
        session_id=session_id,
        contract_ids=request.contracts or None,
        llm_backend=settings.llm_backend,
    )

    loop = asyncio.get_event_loop()
    graph = get_graph()
    result = await loop.run_in_executor(None, lambda: graph.invoke(state))

    report = result.get("report") or {}
    report_id = report.get("session_id", session_id)

    # Persist to shared report store so /reports/{id} can serve it
    if report:
        _report_store[report_id] = report

    risk_flags = _coerce_risk_flags(report.get("risk_flags", []))

    return ChatResponse(
        response=report.get("narrative", "Analysis complete."),
        risk_flags=risk_flags,
        contracts_in_scope=report.get("contract_ids", request.contracts),
        session_id=session_id,
        report_id=report_id,
    )


# ── Remote endpoint proxies ────────────────────────────────────────────────────

async def _call_databricks_endpoint(request: ChatRequest, session_id: str) -> ChatResponse:
    """Proxy to Databricks Model Serving endpoint."""
    url = (
        f"{settings.databricks_host}/serving-endpoints/"
        f"{settings.databricks_serving_endpoint}/invocations"
    )
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": request.message}],
        "contracts": request.contracts,
        "session_id": session_id,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.error(f"Databricks endpoint error: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=502, detail="Databricks endpoint error")
        data = resp.json()

    report_id = data.get("session_id", session_id)
    if data.get("report"):
        _report_store[report_id] = data["report"]

    return ChatResponse(
        response=data.get("response", ""),
        risk_flags=[RiskFlag(**f) for f in data.get("risk_flags", [])],
        contracts_in_scope=data.get("contracts_in_scope", request.contracts),
        session_id=session_id,
        report_id=report_id,
    )


async def _call_mock_endpoint(request: ChatRequest, session_id: str) -> ChatResponse:
    """Proxy to local mock Databricks stub (offline dev via docker compose)."""
    url = "http://mock-databricks:8001/serving-endpoints/renewiq-agent-endpoint/invocations"
    payload = {
        "messages": [{"role": "user", "content": request.message}],
        "contracts": request.contracts,
        "session_id": session_id,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        # Mock endpoint not running — fall back to local graph
        logger.warning("[/chat] Mock endpoint unreachable, falling back to local graph")
        return await _invoke_local_graph(request, session_id)

    report_id = data.get("session_id", session_id)
    return ChatResponse(
        response=data.get("response", ""),
        risk_flags=[RiskFlag(**f) for f in data.get("risk_flags", [])],
        contracts_in_scope=data.get("contracts_in_scope", request.contracts),
        session_id=session_id,
        report_id=report_id,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _coerce_risk_flags(flags: list[dict]) -> list[RiskFlag]:
    result = []
    for f in flags:
        result.append(RiskFlag(
            risk_category=f.get("risk_category", "unknown"),
            severity=f.get("severity", "UNKNOWN"),
            financial_exposure_eur=f.get("exposure_eur"),
            contract_clause=f.get("clause_id"),
            market_trigger=f.get("description"),
            recommendation=None,
        ))
    return result
