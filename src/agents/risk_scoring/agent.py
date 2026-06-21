"""
Risk Scoring Agent
"""
from __future__ import annotations
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONTRACTED_MW = 10.0


def run(state: dict) -> dict:
    clauses        = state.get("retrieved_clauses", [])
    neg_hours      = state.get("negative_hours_90d", 0)
    avg_neg_price  = state.get("avg_negative_price") or 0.0
    market_signals = state.get("market_signals", [])

    logger.info(f"[RiskScoringAgent] scoring {len(clauses)} clauses | {neg_hours} neg hours")

    has_negative_market = neg_hours > 0
    has_congestion = any(s.get("signal_type") == "oversupply" for s in market_signals)

    risk_flags: list[dict] = []
    seen_flags: set[str] = set()

    for clause in clauses:
        flag = _score_clause(
            clause=clause,
            has_negative_market=has_negative_market,
            avg_neg_price=avg_neg_price,
            neg_hours=neg_hours,
            has_congestion=has_congestion,
        )
        if flag is None:
            continue
        dedup_key = f"{flag['contract_id']}:{flag['risk_category']}"
        if dedup_key in seen_flags:
            continue
        seen_flags.add(dedup_key)
        risk_flags.append(flag)

    _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    risk_flags.sort(key=lambda f: _order.get(f["severity"], 9))

    total_exposure = sum(f["exposure_eur"] for f in risk_flags if f.get("exposure_eur"))

    return {
        "risk_flags": risk_flags,
        "total_exposure_eur": total_exposure if total_exposure > 0 else None,
    }


def _score_clause(clause, has_negative_market, avg_neg_price, neg_hours, has_congestion):
    risk_cat  = clause.get("risk_category")
    text      = clause.get("chunk_text", "").lower()
    clause_id = clause.get("clause_id", "")
    contract  = clause.get("contract_id", "")
    if not risk_cat:
        return None
    severity, description, exposure = _evaluate(
        risk_cat, text, has_negative_market, avg_neg_price, neg_hours, has_congestion
    )
    if severity == "INFO":
        return None
    flag_id = hashlib.md5(f"{contract}:{clause_id}:{risk_cat}".encode()).hexdigest()[:12]
    return {
        "flag_id": flag_id,
        "contract_id": contract,
        "clause_id": clause_id,
        "risk_category": risk_cat,
        "severity": severity,
        "description": description,
        "exposure_eur": exposure,
    }


def _evaluate(risk_cat, text, has_negative_market, avg_neg_price, neg_hours, has_congestion):
    if risk_cat == "price_risk":
        _positive_floor = any(kw in text for kw in [
            "floor shall apply", "floor applies", "not fall below zero",
            "deemed to be zero", "minimum price", "price floor of", "floor price of",
        ])
        _negated_floor = any(kw in text for kw in [
            "no price floor", "no floor", "floor shall not", "floor does not",
            "without a price floor", "no minimum price",
        ])
        has_floor = _positive_floor and not _negated_floor
        if has_negative_market and not has_floor:
            exposure = _calc_exposure(neg_hours, avg_neg_price)
            return (
                "HIGH",
                f"No negative price floor in contract. Market had {neg_hours} negative "
                f"price hours (avg {avg_neg_price:.1f} EUR/MWh) in last 90 days.",
                exposure,
            )
        elif has_negative_market and has_floor:
            return (
                "LOW",
                f"Contract has a price floor. Market had {neg_hours} negative hours but exposure is mitigated.",
                None,
            )
        elif not has_negative_market and not has_floor:
            return (
                "MEDIUM",
                "No price floor detected. Current market is stable but negative price periods make this a latent risk.",
                None,
            )
        else:
            return ("INFO", "Price floor present, market stable.", None)

    elif risk_cat == "curtailment_risk":
        no_compensation = any(kw in text for kw in [
            "no compensation", "without compensation", "no deemed output",
            "shall not be payable", "accepts full curtailment risk",
        ])
        if no_compensation and has_congestion:
            exposure = _calc_exposure(int(neg_hours * 0.3), abs(avg_neg_price) * 0.5)
            return (
                "HIGH",
                "No curtailment compensation in contract. GOPACS grid congestion events detected.",
                exposure,
            )
        elif no_compensation:
            return (
                "MEDIUM",
                "No curtailment compensation clause. Grid congestion risk latent but no active GOPACS events.",
                None,
            )
        else:
            return ("LOW", "Curtailment compensation provisions present.", None)

    elif risk_cat == "volume_risk":
        return (
            "MEDIUM",
            "Volume/output guarantee clause identified. Review minimum quantity obligations against P90 forecasts.",
            None,
        )

    elif risk_cat == "counterparty_risk":
        return (
            "MEDIUM",
            "Credit support / collateral clause detected. Verify current credit thresholds and counterparty rating.",
            None,
        )

    elif risk_cat in ("basis_risk", "legal_regulatory"):
        return (
            "LOW",
            f"{risk_cat.replace('_', ' ').title()} clause identified for review.",
            None,
        )

    return ("INFO", "No significant risk identified.", None)


def _calc_exposure(neg_hours: int, avg_neg_price_abs: float) -> float:
    annualised_hours = neg_hours * (365 / 90)
    return round(annualised_hours * _DEFAULT_CONTRACTED_MW * abs(avg_neg_price_abs), 0)
