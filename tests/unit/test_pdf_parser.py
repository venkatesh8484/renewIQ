"""
Unit tests for src/ingestion/pdf_parser.py
-------------------------------------------
Tests run without PyMuPDF (fitz) by patching the import.
This allows CI to run these tests without installing the full PyMuPDF binary.

For real PDF parsing tests, see tests/integration/test_pdf_parser_integration.py
(requires PyMuPDF + synthetic PDFs to be generated first).
"""

import hashlib
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_block(text: str, page: int = 1) -> dict:
    return {"text": text, "page": page}


# ── Import module under test ───────────────────────────────────────────────

from src.ingestion.pdf_parser import (
    PPAChunk,
    _detect_clause,
    _split_into_sections,
    _chunk_sections,
    TOKENS_PER_CHUNK,
    OVERLAP_TOKENS,
)


# ── Tests: _detect_clause ─────────────────────────────────────────────────

class TestDetectClause:
    def test_article_header_uppercase(self):
        result = _detect_clause("ARTICLE 7\nSome content here")
        assert result is not None
        clause_id, title = result
        assert clause_id == "ARTICLE 7"

    def test_article_header_mixed_case(self):
        result = _detect_clause("Article 12\nMore content")
        assert result is not None
        assert result[0] == "Article 12"

    def test_numbered_clause_with_title(self):
        result = _detect_clause("7.2 Negative Price Provisions")
        assert result is not None
        clause_id, title = result
        assert clause_id == "7.2"
        assert "Negative Price Provisions" in title

    def test_nested_numbered_clause(self):
        result = _detect_clause("8.4.1 Curtailment Compensation")
        assert result is not None
        assert result[0] == "8.4.1"

    def test_schedule_header(self):
        result = _detect_clause("SCHEDULE 1\nDelivery Terms")
        assert result is not None
        assert result[0] == "SCHEDULE 1"

    def test_annex_header(self):
        result = _detect_clause("ANNEX A\nCredit Support Provisions")
        assert result is not None
        assert result[0] == "ANNEX A"

    def test_sub_clause_roman(self):
        result = _detect_clause("(i) first item")
        assert result is not None
        assert result[0] == "i"

    def test_plain_paragraph_not_detected(self):
        result = _detect_clause("The seller shall deliver electricity to the buyer.")
        assert result is None

    def test_numbered_list_item_not_detected(self):
        # "7.2" without uppercase title should not match
        result = _detect_clause("7.2 something in lowercase follows")
        assert result is None

    def test_empty_text_not_detected(self):
        result = _detect_clause("")
        assert result is None

    def test_section_title_stripped_of_separators(self):
        result = _detect_clause("ARTICLE 7 — Negative Price")
        assert result is not None
        clause_id, title = result
        assert clause_id == "ARTICLE 7"
        # Title should not start with — or -
        assert not title.startswith("—")
        assert not title.startswith("-")


# ── Tests: _split_into_sections ────────────────────────────────────────────

class TestSplitIntoSections:
    def test_preamble_section_always_created(self):
        blocks = [_make_block("This agreement is entered into by...")]
        sections = _split_into_sections(blocks)
        assert len(sections) == 1
        assert sections[0]["clause_id"] == "PREAMBLE"

    def test_article_starts_new_section(self):
        blocks = [
            _make_block("Preamble content here"),
            _make_block("ARTICLE 7\nNegative Price Provisions", page=2),
            _make_block("If the market price falls below zero...", page=2),
        ]
        sections = _split_into_sections(blocks)
        assert len(sections) == 2
        assert sections[0]["clause_id"] == "PREAMBLE"
        assert sections[1]["clause_id"] == "ARTICLE 7"

    def test_multiple_articles_produce_multiple_sections(self):
        blocks = [
            _make_block("Preamble"),
            _make_block("ARTICLE 1\nDefinitions"),
            _make_block("Definition content"),
            _make_block("ARTICLE 2\nDelivery"),
            _make_block("Delivery content"),
        ]
        sections = _split_into_sections(blocks)
        assert len(sections) == 3

    def test_section_text_joins_multiple_blocks(self):
        blocks = [
            _make_block("First paragraph of article content."),
            _make_block("Second paragraph of article content."),
        ]
        sections = _split_into_sections(blocks)
        assert "First paragraph" in sections[0]["text"]
        assert "Second paragraph" in sections[0]["text"]

    def test_page_number_tracked_from_first_block(self):
        blocks = [
            _make_block("Preamble", page=1),
            _make_block("ARTICLE 7\nPricing", page=5),
        ]
        sections = _split_into_sections(blocks)
        assert sections[1]["page"] == 5

    def test_empty_section_not_added(self):
        # Two consecutive clause headers — first should not produce empty section
        blocks = [
            _make_block("ARTICLE 7\nPricing"),
            _make_block("ARTICLE 8\nDelivery"),
            _make_block("Delivery content"),
        ]
        sections = _split_into_sections(blocks)
        # ARTICLE 7 has no body text, so it produces one section with just the header text
        # ARTICLE 8 has content
        assert all(s["text"].strip() for s in sections)


# ── Tests: _chunk_sections ─────────────────────────────────────────────────

class TestChunkSections:
    def _make_section(self, text: str, clause_id: str = "7.2", page: int = 1) -> dict:
        return {
            "clause_id":     clause_id,
            "section_title": "Test Section",
            "page":          page,
            "text":          text,
        }

    def test_short_section_produces_one_chunk(self):
        section = self._make_section("This is a short clause.")
        chunks = _chunk_sections([section], "contract-001")
        assert len(chunks) == 1

    def test_chunk_inherits_section_metadata(self):
        section = self._make_section("Short text.", clause_id="8.4")
        chunks = _chunk_sections([section], "contract-001")
        chunk = chunks[0]
        assert chunk.contract_id == "contract-001"
        assert chunk.clause_id == "8.4"
        assert chunk.section_title == "Test Section"

    def test_long_section_produces_multiple_overlapping_chunks(self):
        # Create text with more words than TOKENS_PER_CHUNK / 1.3
        words_per_chunk = int(TOKENS_PER_CHUNK / 1.3)
        long_text = " ".join([f"word{i}" for i in range(words_per_chunk * 3)])
        section = self._make_section(long_text)
        chunks = _chunk_sections([section], "contract-001")
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        words_per_chunk = int(TOKENS_PER_CHUNK / 1.3)
        long_text = " ".join([f"word{i}" for i in range(words_per_chunk * 3)])
        section = self._make_section(long_text)
        chunks = _chunk_sections([section], "contract-001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_is_16_char_hex(self):
        section = self._make_section("Some text here.")
        chunks = _chunk_sections([section], "contract-001")
        for chunk in chunks:
            assert len(chunk.chunk_id) == 16
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_token_count_matches_word_count(self):
        section = self._make_section("one two three four five")
        chunks = _chunk_sections([section], "contract-001")
        assert chunks[0].token_count == 5

    def test_chunks_are_ppa_chunk_instances(self):
        section = self._make_section("Some contract clause text.")
        chunks = _chunk_sections([section], "contract-001")
        for chunk in chunks:
            assert isinstance(chunk, PPAChunk)

    def test_risk_category_none_by_default(self):
        section = self._make_section("Some clause.")
        chunks = _chunk_sections([section], "contract-001")
        for chunk in chunks:
            assert chunk.risk_category is None

    def test_different_contracts_produce_different_chunk_ids(self):
        section = self._make_section("Same text.")
        chunks_a = _chunk_sections([section], "contract-A")
        chunks_b = _chunk_sections([section], "contract-B")
        assert chunks_a[0].chunk_id != chunks_b[0].chunk_id


# ── Tests: PPAChunk dataclass ──────────────────────────────────────────────

class TestPPAChunk:
    def test_chunk_construction(self):
        chunk = PPAChunk(
            chunk_id="abc123",
            contract_id="zeeland-wind-ppa",
            chunk_text="No price floor applies.",
            clause_id="7.2",
            section_title="Negative Price Provisions",
            page_number=4,
            token_count=5,
        )
        assert chunk.risk_category is None  # default
        assert chunk.char_start == 0        # default

    def test_chunk_with_risk_category(self):
        chunk = PPAChunk(
            chunk_id="abc123",
            contract_id="test",
            chunk_text="Curtailment compensation provisions.",
            clause_id="9.1",
            section_title="Curtailment",
            page_number=6,
            token_count=4,
            risk_category="curtailment_risk",
        )
        assert chunk.risk_category == "curtailment_risk"


# ── Tests: parse_ppa_pdf (integration, skipped if fitz absent) ─────────────

fitz_available = pytest.importorskip("fitz", reason="PyMuPDF not installed") if False else None
try:
    import fitz as _fitz_check
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


@pytest.mark.skipif(not _FITZ_AVAILABLE, reason="PyMuPDF not installed (CI skip)")
class TestParsePpaPdf:
    """
    Tests for parse_ppa_pdf() — skipped in CI where PyMuPDF binary isn't installed.
    Run locally after: pip install pymupdf
    """

    def test_parse_ppa_pdf_uses_filename_as_contract_id(self, tmp_path):
        """contract_id defaults to pdf stem when no override is given."""
        import fitz
        from src.ingestion.pdf_parser import parse_ppa_pdf

        # Create a minimal 1-page PDF with fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "ARTICLE 1\nThis is a test clause.")
        pdf_path = tmp_path / "zeeland-wind-ppa.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = parse_ppa_pdf(pdf_path)
        assert all(c.contract_id == "zeeland-wind-ppa" for c in chunks)

    def test_parse_ppa_pdf_respects_explicit_contract_id(self, tmp_path):
        import fitz
        from src.ingestion.pdf_parser import parse_ppa_pdf

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "ARTICLE 1\nClause text here.")
        pdf_path = tmp_path / "dummy.pdf"
        doc.save(str(pdf_path))
        doc.close()

        chunks = parse_ppa_pdf(pdf_path, contract_id="my-contract-id")
        assert all(c.contract_id == "my-contract-id" for c in chunks)
