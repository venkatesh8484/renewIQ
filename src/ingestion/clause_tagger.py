"""
Clause Risk Tagger
------------------
Lightweight keyword-based classifier that assigns a `risk_category` to
each PPAChunk. Designed to run fast (no LLM call) during ingestion;
the LangGraph agent layer refines assessments during query time.

Risk categories (aligned with PPA industry taxonomy):
  - price_risk         : negative price floors, price caps, settlement terms
  - volume_risk        : curtailment, output guarantees, capacity obligations
  - curtailment_risk   : grid curtailment, DSO orders, redispatch
  - basis_risk         : location marginal pricing, delivery point, basis differential
  - counterparty_risk  : credit support, collateral, termination, insolvency
  - legal_regulatory   : change in law, permitting, regulatory change, force majeure
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Keyword taxonomy — each category has PRIMARY (high-weight) and
# SECONDARY (lower-weight) terms.  A match on any PRIMARY = tagged.
# Two or more SECONDARY matches = tagged.
# ---------------------------------------------------------------------------

_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "price_risk": {
        "primary": [
            "negative price",
            "price floor",
            "price cap",
            "settlement price",
            "reference price",
            "market price",
            "day-ahead price",
            "balancing price",
        ],
        "secondary": [
            "eur/mwh",
            "strike price",
            "contract price",
            "indexed price",
            "fixed price",
            "floating price",
        ],
    },
    "volume_risk": {
        "primary": [
            "minimum volume",
            "guaranteed output",
            "take-or-pay",
            "annual quantity",
            "capacity obligation",
            "output guarantee",
        ],
        "secondary": [
            "mwh per year",
            "annual generation",
            "contracted volume",
            "forecast output",
            "production shortfall",
            "p50",
            "p90",
        ],
    },
    "curtailment_risk": {
        "primary": [
            "curtailment",
            "curtail",
            "grid curtailment",
            "dso instruction",
            "redispatch",
            "constrained off",
            "dispatch instruction",
        ],
        "secondary": [
            "grid operator",
            "system operator",
            "congestion",
            "balancing responsibility",
            "output reduction",
            "deemed output",
            "compensation for curtailment",
            "no compensation",
        ],
    },
    "basis_risk": {
        "primary": [
            "delivery point",
            "location marginal price",
            "lmp",
            "basis differential",
            "interconnection point",
            "grid connection point",
        ],
        "secondary": [
            "transmission constraint",
            "grid losses",
            "settlement location",
            "delivery node",
            "hub price",
            "zonal price",
        ],
    },
    "counterparty_risk": {
        "primary": [
            "credit support",
            "collateral",
            "letter of credit",
            "parent guarantee",
            "termination event",
            "event of default",
            "insolvency",
        ],
        "secondary": [
            "credit rating",
            "performance bond",
            "security deposit",
            "credit threshold",
            "material adverse change",
            "early termination",
            "close-out",
        ],
    },
    "legal_regulatory": {
        "primary": [
            "change in law",
            "force majeure",
            "permitting",
            "regulatory change",
            "government authority",
            "legislative change",
        ],
        "secondary": [
            "applicable law",
            "environmental permit",
            "grid code",
            "subsidy",
            "renewable obligation",
            "ets",
            "emissions trading",
            "planning permission",
        ],
    },
}

# Pre-compile all patterns (case-insensitive word-boundary match)
_COMPILED: dict[str, dict[str, list[re.Pattern]]] = {
    category: {
        weight: [
            re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for term in terms
        ]
        for weight, terms in weights.items()
    }
    for category, weights in _TAXONOMY.items()
}

# Ordered by priority — first match wins for `tag_chunk`; `tag_all` returns all matches.
_CATEGORY_ORDER = [
    "curtailment_risk",   # most specific — check before volume_risk
    "price_risk",
    "volume_risk",
    "basis_risk",
    "counterparty_risk",
    "legal_regulatory",
]


def tag_chunk(text: str) -> Optional[str]:
    """
    Return the single highest-priority risk category for `text`, or None.

    Uses priority order: curtailment > price > volume > basis > counterparty > legal.
    Call `tag_all` if you need every matched category.
    """
    for category in _CATEGORY_ORDER:
        if _matches(text, category):
            return category
    return None


def tag_all(text: str) -> list[str]:
    """
    Return all matching risk categories (may be empty).
    Useful for multi-label annotation during evaluation.
    """
    return [cat for cat in _CATEGORY_ORDER if _matches(text, cat)]


def _matches(text: str, category: str) -> bool:
    """
    Return True if text matches the category's keyword rules:
      - Any PRIMARY keyword match → True
      - Two or more SECONDARY keyword matches → True
    """
    patterns = _COMPILED[category]

    # Primary: single hit is enough
    for pat in patterns["primary"]:
        if pat.search(text):
            return True

    # Secondary: need 2+ hits
    secondary_hits = sum(1 for pat in patterns["secondary"] if pat.search(text))
    return secondary_hits >= 2


def tag_chunks_batch(chunks) -> list:
    """
    Tag a list of PPAChunk objects in-place (sets risk_category field).
    Returns the same list for chaining.

    Works with any object that has `chunk_text` and `risk_category` attributes.
    """
    for chunk in chunks:
        if chunk.risk_category is None:
            chunk.risk_category = tag_chunk(chunk.chunk_text)
    return chunks


# ---------------------------------------------------------------------------
# Convenience: tag raw text strings (for Databricks UDF wrapper)
# ---------------------------------------------------------------------------

def tag_text(text: str) -> Optional[str]:
    """Alias for tag_chunk — used as the Spark UDF entry point."""
    return tag_chunk(text)
