"""
AgentState — Shared LangGraph State
-------------------------------------
Single TypedDict passed between every node in the RenewIQ StateGraph.
All agents READ from state and WRITE to their own output keys.
The orchestrator merges outputs before passing to downstream nodes.

State lifecycle:
  1. User query arrives at /chat → routed to orchestrator
  2. Orchestrator fan-out: MarketDataAgent ‖ ContractRAGAgent (parallel)
  3. RiskScoringAgent consumes both outputs
  4. ReportWriterAgent produces final RiskReport
  5. FastAPI serialises RiskReport → ChatResponse
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


# ── Sub-structures ─────────────────────────────────────────────────────────────

class MarketSignal(TypedDict):
    delivery_date:  str
    hour:           int
    price_eur_mwh:  float
    is_negative:    bool
    signal_type:    str     # "negative_price" | "oversupply" | "normal"
    severity:       str     # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE"
    renewable_pct:  Optional[float]
    details:        str     # JSON blob with wind/solar


class ContractClause(TypedDict):
    chunk_id:       str
    contract_id:    str
    clause_id:      str
    section_title:  str
    risk_category:  Optional[str]
    chunk_text:     str
    rerank_score:   float


class RiskFlag(TypedDict):
    flag_id:        str
    contract_id:    str
    clause_id:      str
    risk_category:  str
    severity:       str     # "HIGH" | "MEDIUM" | "LOW"
    description:    str
    exposure_eur:   Optional[float]


class RiskReport(TypedDict):
    session_id:         str
    query:              str
    contract_ids:       list[str]
    risk_flags:         list[RiskFlag]
    total_exposure_eur: Optional[float]
    negative_hours_90d: int
    avg_negative_price: Optional[float]
    narrative:          str     # LLM-generated summary paragraph
    sources:            list[str]   # chunk_ids cited


# ── Main AgentState ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    session_id:     str
    query:          str
    contract_ids:   list[str]       # contracts to analyse (empty = all)

    # ── Market Data Agent output ───────────────────────────────────────────────
    market_signals:     Annotated[list[MarketSignal], operator.add]
    negative_hours_90d: int
    avg_negative_price: Optional[float]
    market_context:     str     # human-readable market summary for LLM prompt

    # ── Contract RAG Agent output ──────────────────────────────────────────────
    retrieved_clauses:  Annotated[list[ContractClause], operator.add]
    rag_context:        str     # formatted clause snippets for LLM prompt

    # ── Risk Scoring Agent output ──────────────────────────────────────────────
    risk_flags:         Annotated[list[RiskFlag], operator.add]
    total_exposure_eur: Optional[float]

    # ── Report Writer output ───────────────────────────────────────────────────
    report:             Optional[RiskReport]

    # ── Routing / control ─────────────────────────────────────────────────────
    errors:             Annotated[list[str], operator.add]   # non-fatal errors per agent
    llm_backend:        str     # "ollama" | "databricks" — set from config at entry


def make_initial_state(
    query: str,
    session_id: str,
    contract_ids: Optional[list[str]] = None,
    llm_backend: str = "ollama",
) -> AgentState:
    """Build the initial state dict for a new graph invocation."""
    return AgentState(
        session_id          = session_id,
        query               = query,
        contract_ids        = contract_ids or [],
        market_signals      = [],
        negative_hours_90d  = 0,
        avg_negative_price  = None,
        market_context      = "",
        retrieved_clauses   = [],
        rag_context         = "",
        risk_flags          = [],
        total_exposure_eur  = None,
        report              = None,
        errors              = [],
        llm_backend         = llm_backend,
    )
