"""
Report Writer Agent
--------------------
Final node in the LangGraph graph.
"""
from __future__ import annotations
import logging
import uuid
from typing import Optional

try:
    from src.agents.orchestrator.llm import get_llm   # module-level for patch() targeting
except Exception:
    get_llm = None  # LLM not available in this environment (CI / no Ollama)

logger = logging.getLogger(__name__)

_NARRATIVE_PROMPT = """\
You are a PPA risk analyst at a renewable energy company.
Based on the market data and contract analysis below, write a concise risk narrative
(2-3 paragraphs, professional tone, no bullet points) for a senior energy trader.

## User Query
{query}

{market_context}

{rag_context}

## Risk Flags Identified
{risk_flags_text}

## Instructions
- Summarise the key risks in plain English
- Quantify exposure in EUR where available
- Reference specific clause numbers (e.g., "Clause 7.2")
- End with a recommended action (hedge, renegotiate, monitor)
- Do NOT invent information not present in the context above
"""


def run(state: dict) -> dict:
    query          = state.get("query", "")
    session_id     = state.get("session_id", str(uuid.uuid4()))
    risk_flags     = state.get("risk_flags", [])
    clauses        = state.get("retrieved_clauses", [])
    market_context = state.get("market_context", "")
    rag_context    = state.get("rag_context", "")
    neg_hours      = state.get("negative_hours_90d", 0)
    avg_neg_price  = state.get("avg_negative_price")
    total_exposure = state.get("total_exposure_eur")
    contract_ids   = state.get("contract_ids") or _unique_contracts(clauses)
    llm_backend    = state.get("llm_backend", "ollama")

    narrative = _generate_narrative(
        query=query,
        risk_flags=risk_flags,
        market_context=market_context,
        rag_context=rag_context,
        llm_backend=llm_backend,
    )

    report = {
        "session_id":         session_id,
        "query":              query,
        "contract_ids":       contract_ids,
        "risk_flags":         risk_flags,
        "total_exposure_eur": total_exposure,
        "negative_hours_90d": neg_hours,
        "avg_negative_price": avg_neg_price,
        "narrative":          narrative,
        "sources":            [c["chunk_id"] for c in clauses],
    }

    return {"report": report}


def _generate_narrative(query, risk_flags, market_context, rag_context, llm_backend):
    risk_flags_text = _format_flags(risk_flags)
    prompt = _NARRATIVE_PROMPT.format(
        query=query,
        market_context=market_context,
        rag_context=rag_context[:2000],
        risk_flags_text=risk_flags_text,
    )
    try:
        if get_llm is None:
            raise ImportError("LLM not available in this environment")
        llm = get_llm()
        response = llm.invoke(prompt)
        narrative = response.content if hasattr(response, "content") else str(response)
        return narrative.strip()
    except Exception as exc:
        logger.warning(f"[ReportWriterAgent] LLM call failed ({exc}), using template")
        return _template_narrative(risk_flags)


def _format_flags(risk_flags: list) -> str:
    if not risk_flags:
        return "No significant risk flags identified."
    lines = []
    for f in risk_flags:
        exposure_str = (
            f" | Estimated exposure: EUR {f['exposure_eur']:,.0f}/year"
            if f.get("exposure_eur") else ""
        )
        lines.append(
            f"[{f['severity']}] {f['contract_id']} - {f['clause_id']} "
            f"({f['risk_category']}){exposure_str}\n  {f['description']}"
        )
    return "\n".join(lines)


def _template_narrative(risk_flags: list) -> str:
    high = [f for f in risk_flags if f["severity"] == "HIGH"]
    med  = [f for f in risk_flags if f["severity"] == "MEDIUM"]

    if not risk_flags:
        return (
            "Analysis complete. No material risk flags were identified in the reviewed "
            "contract clauses given current market conditions. "
            "Standard monitoring is recommended."
        )

    parts = []

    if high:
        contracts = list({f["contract_id"] for f in high})
        exposures = [f["exposure_eur"] for f in high if f.get("exposure_eur")]
        total_exp = sum(exposures) if exposures else None
        exp_str = f" with estimated annual exposure of EUR {total_exp:,.0f}" if total_exp else ""
        parts.append(
            f"The analysis identified {len(high)} HIGH severity risk flag(s) "
            f"across {len(contracts)} contract(s){exp_str}. "
            + " ".join(
                f"In {f['contract_id']}, {f['clause_id']} was flagged: {f['description']}"
                for f in high[:2]
            )
        )

    if med:
        parts.append(
            f"Additionally, {len(med)} MEDIUM severity flag(s) were noted requiring "
            "attention during the next contract review cycle."
        )

    parts.append(
        "Recommended action: review flagged clauses with legal counsel and consider "
        "hedging negative price exposure via financial PPA overlay or price floor amendment."
    )

    return " ".join(parts)


def _unique_contracts(clauses: list) -> list:
    seen = []
    for c in clauses:
        cid = c.get("contract_id", "")
        if cid and cid not in seen:
            seen.append(cid)
    return seen
