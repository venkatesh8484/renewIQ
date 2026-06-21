"""
RenewIQ LangGraph Orchestrator
--------------------------------
Builds and compiles the multi-agent StateGraph.

Topology:
  START
    │
    ▼
  [market_data] ──┐
                  ├──► [risk_scoring] ──► [report_writer] ──► END
  [contract_rag] ─┘

market_data and contract_rag run in PARALLEL (LangGraph fan-out).
risk_scoring waits for both (LangGraph fan-in via Send).
report_writer is the final synthesis step.

Usage:
    from src.agents.orchestrator.graph import build_graph

    graph = build_graph()
    result = graph.invoke(make_initial_state(query="...", session_id="..."))
    report = result["report"]
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.agents.orchestrator.state import AgentState
import src.agents.market_data.agent     as market_data_agent
import src.agents.contract_rag.agent    as contract_rag_agent
import src.agents.risk_scoring.agent    as risk_scoring_agent
import src.agents.report_writer.agent   as report_writer_agent

logger = logging.getLogger(__name__)


# ── Node wrappers ──────────────────────────────────────────────────────────────
# LangGraph nodes must be callables: (state) → dict

def _market_data_node(state: AgentState) -> dict:
    logger.debug("[Graph] market_data_node start")
    result = market_data_agent.run(state)
    logger.debug("[Graph] market_data_node done")
    return result


def _contract_rag_node(state: AgentState) -> dict:
    logger.debug("[Graph] contract_rag_node start")
    result = contract_rag_agent.run(state)
    logger.debug("[Graph] contract_rag_node done")
    return result


def _risk_scoring_node(state: AgentState) -> dict:
    logger.debug("[Graph] risk_scoring_node start")
    result = risk_scoring_agent.run(state)
    logger.debug("[Graph] risk_scoring_node done")
    return result


def _report_writer_node(state: AgentState) -> dict:
    logger.debug("[Graph] report_writer_node start")
    result = report_writer_agent.run(state)
    logger.debug("[Graph] report_writer_node done")
    return result


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> "CompiledGraph":
    """
    Construct and compile the RenewIQ StateGraph.

    Returns a compiled graph ready for .invoke() or .stream() calls.
    The compiled graph is thread-safe and can be reused across requests.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("market_data",   _market_data_node)
    builder.add_node("contract_rag",  _contract_rag_node)
    builder.add_node("risk_scoring",  _risk_scoring_node)
    builder.add_node("report_writer", _report_writer_node)

    # Edges — fan-out from START to parallel agents
    builder.add_edge(START,          "market_data")
    builder.add_edge(START,          "contract_rag")

    # Fan-in — both parallel agents must complete before risk_scoring
    builder.add_edge("market_data",  "risk_scoring")
    builder.add_edge("contract_rag", "risk_scoring")

    # Linear tail
    builder.add_edge("risk_scoring",  "report_writer")
    builder.add_edge("report_writer", END)

    graph = builder.compile()
    logger.info("[Graph] RenewIQ StateGraph compiled successfully")
    return graph


# ── Module-level singleton (lazy) ──────────────────────────────────────────────
# Avoids rebuilding the graph on every request.

_GRAPH = None

def get_graph():
    """Return the singleton compiled graph, building it on first call."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def reset_graph():
    """Force rebuild on next get_graph() call — useful in tests."""
    global _GRAPH
    _GRAPH = None
