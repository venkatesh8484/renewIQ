"""
Market Data Agent
------------------
Queries renewiq.gold.* tables via Databricks SQL (or a local Parquet fallback
when running offline) and returns:
  - Top-N market stress signals (negative price / oversupply hours)
  - 90-day negative price statistics
  - A human-readable market context string for downstream LLM prompting

The agent is a pure function: AgentState → dict of updates.
LangGraph merges the returned dict into the shared state.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── SQL queries ────────────────────────────────────────────────────────────────

_SQL_SIGNALS = """
SELECT
    delivery_date,
    hour,
    ROUND(price_eur_mwh, 2)  AS price_eur_mwh,
    is_negative,
    signal_type,
    severity,
    ROUND(COALESCE(renewable_pct, 0), 3) AS renewable_pct,
    details
FROM renewiq.gold.market_risk_signals
WHERE signal_type != 'normal'
ORDER BY delivery_date DESC, hour ASC
LIMIT 50
"""

_SQL_STATS = """
SELECT
    COUNT(*)                         AS negative_hours_90d,
    ROUND(AVG(price_eur_mwh), 2)    AS avg_negative_price,
    ROUND(MIN(price_eur_mwh), 2)    AS min_price,
    ROUND(MAX(ABS(price_eur_mwh)) * COUNT(*), 0) AS estimated_exposure_eur
FROM renewiq.gold.market_risk_signals
WHERE is_negative = true
"""

_SQL_PORTFOLIO = """
SELECT
    COUNT(DISTINCT delivery_date)   AS negative_days,
    SUM(negative_hours)             AS total_negative_hours,
    ROUND(AVG(avg_negative_price_eur), 2) AS avg_neg_price
FROM renewiq.gold.portfolio_exposure_daily
"""


def run(state: dict) -> dict:
    """
    LangGraph node function.

    Reads market data from Databricks Gold tables and returns state updates.
    Falls back to mock data when RENEWIQ_USE_MOCK_ENDPOINT=true or no
    Databricks connection is available.
    """
    query = state.get("query", "")
    logger.info(f"[MarketDataAgent] query={query[:80]!r}")

    try:
        signals, stats, portfolio = _fetch_from_databricks()
    except Exception as exc:
        logger.warning(f"[MarketDataAgent] Databricks unavailable ({exc}), using mock data")
        signals, stats, portfolio = _mock_data()

    market_context = _build_context(signals, stats, portfolio)

    return {
        "market_signals":     signals,
        "negative_hours_90d": stats.get("negative_hours_90d", 0),
        "avg_negative_price": stats.get("avg_negative_price"),
        "market_context":     market_context,
    }


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_from_databricks() -> tuple[list[dict], dict, dict]:
    """Query Gold tables via Databricks SQL connector."""
    if os.getenv("RENEWIQ_USE_MOCK_ENDPOINT", "false").lower() == "true":
        raise RuntimeError("Mock mode forced via env var")

    from databricks import sql as dbsql  # type: ignore[import]

    host  = os.environ["DATABRICKS_HOST"]
    token = os.environ["DATABRICKS_TOKEN"]
    http_path = os.environ.get(
        "DATABRICKS_SQL_HTTP_PATH",
        os.environ.get("DATABRICKS_HTTP_PATH", ""),
    )

    def _run_query(conn, sql: str) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    with dbsql.connect(
        server_hostname = host.replace("https://", ""),
        http_path       = http_path,
        access_token    = token,
    ) as conn:
        signals_rows   = _run_query(conn, _SQL_SIGNALS)
        stats_rows     = _run_query(conn, _SQL_STATS)
        portfolio_rows = _run_query(conn, _SQL_PORTFOLIO)

    signals   = [_row_to_signal(r) for r in signals_rows]
    stats     = stats_rows[0] if stats_rows else {}
    portfolio = portfolio_rows[0] if portfolio_rows else {}
    return signals, stats, portfolio


def _row_to_signal(row: dict) -> dict:
    return {
        "delivery_date": str(row.get("delivery_date", "")),
        "hour":          int(row.get("hour", 0)),
        "price_eur_mwh": float(row.get("price_eur_mwh", 0.0)),
        "is_negative":   bool(row.get("is_negative", False)),
        "signal_type":   str(row.get("signal_type", "normal")),
        "severity":      str(row.get("severity", "NONE")),
        "renewable_pct": float(row.get("renewable_pct") or 0.0),
        "details":       str(row.get("details", "{}")),
    }


def _mock_data() -> tuple[list[dict], dict, dict]:
    """
    Deterministic mock — mirrors the synthetic EPEX data produced by EPEXFetcher.
    Used in local dev (LLM_BACKEND=ollama) and CI.
    """
    today = datetime.utcnow().date()
    signals = []
    for d in range(5):
        date_str = str(today - timedelta(days=d))
        for hour in [10, 11, 12, 13, 14]:
            price = -35.0 - (hour - 10) * 5
            signals.append({
                "delivery_date": date_str,
                "hour":          hour,
                "price_eur_mwh": price,
                "is_negative":   True,
                "signal_type":   "negative_price",
                "severity":      "HIGH" if price < -30 else "MEDIUM",
                "renewable_pct": 0.72,
                "details":       json.dumps({
                    "price_eur_mwh": price,
                    "renewable_pct": 0.72,
                    "wind_onshore_mw": 4200.0,
                    "solar_mw": 3800.0,
                }),
            })

    stats = {
        "negative_hours_90d":      168,
        "avg_negative_price":      -28.7,
        "min_price":               -82.4,
        "estimated_exposure_eur":  5760.0,
    }
    portfolio = {
        "negative_days":       80,
        "total_negative_hours": 168,
        "avg_neg_price":       -28.7,
    }
    return signals, stats, portfolio


# ── Context builder ────────────────────────────────────────────────────────────

def _build_context(signals: list[dict], stats: dict, portfolio: dict) -> str:
    """
    Produce a compact markdown summary for inclusion in LLM prompts.
    Keeps token count low — agents add detail from retrieved_clauses.
    """
    neg_hours = stats.get("negative_hours_90d", 0)
    avg_price = stats.get("avg_negative_price")
    min_price = stats.get("min_price")
    neg_days  = portfolio.get("negative_days", 0)

    # Top-3 worst hours
    worst = sorted(signals, key=lambda s: s["price_eur_mwh"])[:3]
    worst_lines = "\n".join(
        f"  • {s['delivery_date']} H{s['hour']:02d}: {s['price_eur_mwh']:.1f} €/MWh "
        f"({s['severity']})"
        for s in worst
    )

    high_re = [s for s in signals if s.get("renewable_pct", 0) > 0.70]

    ctx = f"""## Market Context (last 90 days — NL day-ahead)

- **Negative price hours**: {neg_hours} hrs across {neg_days} days
- **Average negative price**: {avg_price:.1f} €/MWh
- **Worst recorded price**: {min_price:.1f} €/MWh
- **High-renewable overlap**: {len(high_re)} negative hours had >70% renewable share

**Worst hours (sample):**
{worst_lines or "  (none in current window)"}

*Negative price exposure is concentrated 10:00–15:00 CET on high-wind, high-solar days.*
"""
    return ctx
