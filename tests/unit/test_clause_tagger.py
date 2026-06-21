"""
Unit tests for src/ingestion/clause_tagger.py
----------------------------------------------
Verifies keyword taxonomy, single-label tagging, multi-label tagging,
batch tagging, and priority ordering.
"""

import pytest
from src.ingestion.clause_tagger import tag_chunk, tag_all, tag_chunks_batch, tag_text


# ── Fixtures — representative clause text per category ────────────────────

PRICE_RISK_TEXT = """
7.2 Negative Price Provisions
If the day-ahead price falls below zero EUR/MWH for any delivery hour,
the Seller shall apply the settlement price as the reference price
for that hour. No price floor shall apply unless separately agreed.
"""

VOLUME_RISK_TEXT = """
6.1 Guaranteed Output
The Seller guarantees a minimum volume of 45,000 MWh per year.
If actual generation falls below 90% of the contracted volume,
the Seller shall pay the Buyer a shortfall payment at the contract price.
"""

CURTAILMENT_TEXT = """
9.3 Grid Curtailment Compensation
Where the DSO instruction results in curtailment of the Facility,
the Seller shall be entitled to deemed output compensation equal to
the estimated generation at the prevailing settlement price.
No compensation shall apply if the curtailment is due to force majeure.
"""

BASIS_RISK_TEXT = """
4.2 Delivery Point and Location Marginal Price
Settlement shall be based on the location marginal price at the
agreed grid connection point. Any basis differential between the
hub price and the delivery node price shall be for the Buyer's account.
"""

COUNTERPARTY_RISK_TEXT = """
15.1 Credit Support Requirements
Each party shall provide credit support in the form of a letter of credit
from an acceptable financial institution. An event of default shall
include insolvency, material adverse change, or failure to maintain
collateral at the required credit threshold.
"""

LEGAL_REGULATORY_TEXT = """
18.2 Change in Law
If a change in law, legislative change, or regulatory change materially
affects either party's ability to perform, the affected party may notify
the other and both parties shall negotiate in good faith. Environmental
permit withdrawal shall constitute a change in law event.
"""

FORCE_MAJEURE_TEXT = """
19.1 Force Majeure
Neither party shall be liable for failure to perform caused by
force majeure, government authority action, or acts beyond
reasonable control. The affected party shall provide written notice
within five business days of the force majeure event.
"""

PLAIN_TEXT = """
The parties have read and understood the terms of this agreement
and have executed this document as of the date first written above.
Signatures appear on Schedule 1 (Signature Page).
"""


# ── Tests: tag_chunk (single label) ───────────────────────────────────────

class TestTagChunk:
    def test_price_risk_detected(self):
        result = tag_chunk(PRICE_RISK_TEXT)
        assert result == "price_risk"

    def test_volume_risk_detected(self):
        result = tag_chunk(VOLUME_RISK_TEXT)
        assert result == "volume_risk"

    def test_curtailment_risk_detected(self):
        result = tag_chunk(CURTAILMENT_TEXT)
        assert result == "curtailment_risk"

    def test_basis_risk_detected(self):
        result = tag_chunk(BASIS_RISK_TEXT)
        assert result == "basis_risk"

    def test_counterparty_risk_detected(self):
        result = tag_chunk(COUNTERPARTY_RISK_TEXT)
        assert result == "counterparty_risk"

    def test_legal_regulatory_detected(self):
        result = tag_chunk(LEGAL_REGULATORY_TEXT)
        assert result == "legal_regulatory"

    def test_force_majeure_is_legal_regulatory(self):
        result = tag_chunk(FORCE_MAJEURE_TEXT)
        assert result == "legal_regulatory"

    def test_plain_text_returns_none(self):
        result = tag_chunk(PLAIN_TEXT)
        assert result is None

    def test_empty_string_returns_none(self):
        result = tag_chunk("")
        assert result is None

    def test_primary_keyword_alone_matches(self):
        result = tag_chunk("The negative price provision applies from midnight.")
        assert result == "price_risk"

    def test_single_secondary_keyword_does_not_match(self):
        # "eur/mwh" alone = 1 secondary hit — not enough
        result = tag_chunk("The rate is 50 eur/mwh for all hours.")
        # Should not match price_risk on one secondary hit
        assert result != "price_risk" or result is None

    def test_two_secondary_keywords_match(self):
        # "eur/mwh" + "fixed price" = 2 secondary hits → price_risk
        result = tag_chunk("The fixed price of 55 eur/mwh applies throughout the term.")
        assert result == "price_risk"


# ── Tests: tag_all (multi-label) ─────────────────────────────────────────

class TestTagAll:
    def test_curtailment_text_may_also_match_legal(self):
        # Curtailment text mentions force majeure — could match legal too
        result = tag_all(CURTAILMENT_TEXT)
        assert "curtailment_risk" in result

    def test_plain_text_returns_empty_list(self):
        result = tag_all(PLAIN_TEXT)
        assert result == []

    def test_returns_list(self):
        result = tag_all(PRICE_RISK_TEXT)
        assert isinstance(result, list)

    def test_all_detected_categories_are_valid(self):
        valid = {
            "price_risk", "volume_risk", "curtailment_risk",
            "basis_risk", "counterparty_risk", "legal_regulatory",
        }
        for text in [PRICE_RISK_TEXT, VOLUME_RISK_TEXT, CURTAILMENT_TEXT,
                     BASIS_RISK_TEXT, COUNTERPARTY_RISK_TEXT, LEGAL_REGULATORY_TEXT]:
            result = tag_all(text)
            for cat in result:
                assert cat in valid


# ── Tests: priority ordering ───────────────────────────────────────────────

class TestPriorityOrdering:
    def test_curtailment_wins_over_volume(self):
        """
        Text mentions both curtailment + contracted volume.
        curtailment_risk has higher priority → wins.
        """
        mixed = """
        The curtailment instruction reduced contracted volume output significantly.
        The Seller's annual quantity was affected by DSO redispatch orders.
        """
        result = tag_chunk(mixed)
        assert result == "curtailment_risk"

    def test_curtailment_wins_over_legal(self):
        """
        Curtailment text mentions force majeure — curtailment_risk still wins.
        """
        mixed = CURTAILMENT_TEXT  # contains "force majeure" but curtailment wins
        result = tag_chunk(mixed)
        assert result == "curtailment_risk"


# ── Tests: tag_text alias ─────────────────────────────────────────────────

class TestTagText:
    def test_tag_text_same_as_tag_chunk(self):
        for text in [PRICE_RISK_TEXT, PLAIN_TEXT, CURTAILMENT_TEXT]:
            assert tag_text(text) == tag_chunk(text)


# ── Tests: tag_chunks_batch ───────────────────────────────────────────────

class TestTagChunksBatch:
    def _make_chunk(self, text: str, risk_cat=None):
        class _C:
            def __init__(self, t, r):
                self.chunk_text = t
                self.risk_category = r
        return _C(text, risk_cat)

    def test_batch_tags_untagged_chunks(self):
        chunks = [
            self._make_chunk(PRICE_RISK_TEXT),
            self._make_chunk(CURTAILMENT_TEXT),
            self._make_chunk(PLAIN_TEXT),
        ]
        result = tag_chunks_batch(chunks)
        assert result[0].risk_category == "price_risk"
        assert result[1].risk_category == "curtailment_risk"
        assert result[2].risk_category is None

    def test_batch_does_not_overwrite_existing_tags(self):
        chunk = self._make_chunk(PRICE_RISK_TEXT, risk_cat="manual_tag")
        tag_chunks_batch([chunk])
        assert chunk.risk_category == "manual_tag"

    def test_batch_returns_same_list(self):
        chunks = [self._make_chunk("text")]
        result = tag_chunks_batch(chunks)
        assert result is chunks

    def test_empty_batch_returns_empty_list(self):
        result = tag_chunks_batch([])
        assert result == []

    def test_batch_processes_all_elements(self):
        chunks = [self._make_chunk(PRICE_RISK_TEXT) for _ in range(20)]
        tag_chunks_batch(chunks)
        assert all(c.risk_category == "price_risk" for c in chunks)
