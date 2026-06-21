"""
POST /chat — conversational interface to the RenewIQ agent.

In Phase 1 this is stubbed. Phase 4 wires in the real LangGraph orchestrator.
"""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a natural-language query about PPA contracts and live market data.

    Examples:
    - "What is our exposure if EPEX NL goes negative tonight?"
    - "Does our Zeeland PPA have a negative price floor clause?"
    - "Which of our contracts has the highest curtailment risk right now?"
    """
    logger.info(f"Chat request: contracts={request.contracts}, msg={request.message[:80]}")

    # Phase 1 stub — returns a structured placeholder response.
    # Phase 4 replaces this with: response = await orchestrator.invoke(request)
    if settings.use_mock_endpoint:
        return await _call_mock_endpoint(request)

    if settings.llm_backend == "databricks":
        return await _call_databricks_endpoint(request)

    # Local Ollama path — Phase 4 wires the full LangGraph graph here
    return ChatResponse(
        response=(
            "[Phase 1 stub] Agent not yet wired. "
            f"Received: '{request.message}'. "
            "Full LangGraph orchestrator connects in Phase 4."
        ),
        contracts_in_scope=request.contracts,
        session_id=request.session_id,
    )


async def _call_databricks_endpoint(request: ChatRequest) -> ChatResponse:
    """Proxy to Databricks Model Serving endpoint."""
    url = f"{settings.databricks_host}/serving-endpoints/{settings.databricks_serving_endpoint}/invocations"
    headers = {
        "Authorization": f"Bearer {settings.databricks_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": request.message}],
        "contracts": request.contracts,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.error(f"Databricks endpoint error: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=502, detail="Databricks endpoint error")
        data = resp.json()

    return ChatResponse(
        response=data.get("response", ""),
        risk_flags=[RiskFlag(**f) for f in data.get("risk_flags", [])],
        contracts_in_scope=data.get("contracts_in_scope", request.contracts),
        session_id=request.session_id,
    )


async def _call_mock_endpoint(request: ChatRequest) -> ChatResponse:
    """Proxy to local mock Databricks stub (offline dev)."""
    url = f"http://mock-databricks:8001/serving-endpoints/renewiq-agent-endpoint/invocations"
    payload = {
        "messages": [{"role": "user", "content": request.message}],
        "contracts": request.contracts,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return ChatResponse(
        response=data.get("response", ""),
        risk_flags=[RiskFlag(**f) for f in data.get("risk_flags", [])],
        contracts_in_scope=data.get("contracts_in_scope", request.contracts),
        session_id=request.session_id,
    )
