"""
PPA PDF Parser — Section-Aware Chunking
-----------------------------------------
Extracts text from PPA PDF contracts using PyMuPDF (fitz),
preserving clause structure (Article 7.2, Clause 8.4, etc.)
and chunking into 512-token windows with 64-token overlap.

Each chunk carries metadata:
  - contract_id, clause_id, section_title, page_number, token_count
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Clause number patterns we detect in PPA documents
CLAUSE_PATTERNS = [
    re.compile(r"^(ARTICLE\s+\d+)", re.IGNORECASE),           # ARTICLE 7
    re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(?=[A-Z])"),         # 7.2 Negative Price
    re.compile(r"^(SCHEDULE\s+\d+)", re.IGNORECASE),           # SCHEDULE 1
    re.compile(r"^(ANNEX\s+[A-Z\d]+)", re.IGNORECASE),        # ANNEX A
    re.compile(r"^\(([ivxlcdmIVXLCDM]+)\)\s"),                 # (i) sub-clauses
]

TOKENS_PER_CHUNK = 512
OVERLAP_TOKENS   = 64


@dataclass
class PPAChunk:
    chunk_id:      str
    contract_id:   str
    chunk_text:    str
    clause_id:     str           # e.g. "7.2", "ARTICLE 7", "SCHEDULE 1"
    section_title: str           # e.g. "Negative Price Provisions"
    page_number:   int
    token_count:   int
    risk_category: Optional[str] = None   # set by RiskTagger after chunking
    char_start:    int = 0


def parse_ppa_pdf(pdf_path: Path, contract_id: Optional[str] = None) -> list[PPAChunk]:
    """
    Parse a PPA PDF into section-aware chunks.

    Args:
        pdf_path:    Path to the PDF file
        contract_id: Override; defaults to pdf stem (filename without extension)

    Returns:
        List of PPAChunk objects ready for embedding + upsert to pgvector/Vector Search
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    if contract_id is None:
        contract_id = pdf_path.stem

    logger.info(f"Parsing PDF: {pdf_path} (contract_id={contract_id})")

    doc = fitz.open(str(pdf_path))
    all_blocks: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")   # [(x0, y0, x1, y1, text, block_no, block_type)]
        for block in blocks:
            text = block[4].strip()
            if len(text) < 5:   # skip near-empty blocks
                continue
            all_blocks.append({"text": text, "page": page_num})

    doc.close()

    # Detect clause boundaries and build sections
    sections = _split_into_sections(all_blocks)
    chunks   = _chunk_sections(sections, contract_id)

    logger.info(f"Parsed {len(chunks)} chunks from {len(doc)} pages ({contract_id})")
    return chunks


def _split_into_sections(blocks: list[dict]) -> list[dict]:
    """
    Group blocks into logical sections based on clause headings.
    Returns list of {clause_id, section_title, page, text}.
    """
    sections = []
    current  = {"clause_id": "PREAMBLE", "section_title": "Preamble", "page": 1, "texts": []}

    for block in blocks:
        text     = block["text"]
        page     = block["page"]
        detected = _detect_clause(text)

        if detected:
            # Save current section if it has content
            if current["texts"]:
                sections.append({
                    "clause_id":     current["clause_id"],
                    "section_title": current["section_title"],
                    "page":          current["page"],
                    "text":          " ".join(current["texts"]),
                })
            # Start new section
            clause_id, section_title = detected
            current = {
                "clause_id":     clause_id,
                "section_title": section_title,
                "page":          page,
                "texts":         [text],
            }
        else:
            current["texts"].append(text)

    # Flush last section
    if current["texts"]:
        sections.append({
            "clause_id":     current["clause_id"],
            "section_title": current["section_title"],
            "page":          current["page"],
            "text":          " ".join(current["texts"]),
        })

    return sections


def _detect_clause(text: str) -> Optional[tuple[str, str]]:
    """
    Return (clause_id, section_title) if text is a clause heading, else None.
    """
    first_line = text.split("\n")[0].strip()

    for pattern in CLAUSE_PATTERNS:
        m = pattern.match(first_line)
        if m:
            clause_id = m.group(1).strip()
            # Section title = everything after the clause id on the first line
            rest = first_line[m.end():].strip(" —-:")
            section_title = rest if rest else clause_id
            return clause_id, section_title

    return None


def _chunk_sections(sections: list[dict], contract_id: str) -> list[PPAChunk]:
    """
    Chunk each section into TOKENS_PER_CHUNK-word windows with OVERLAP_TOKENS overlap.
    (Using word count as a proxy for tokens — ~1.3 words per token average.)
    """
    words_per_chunk   = int(TOKENS_PER_CHUNK / 1.3)
    words_per_overlap = int(OVERLAP_TOKENS / 1.3)

    chunks: list[PPAChunk] = []

    for section in sections:
        words     = section["text"].split()
        step      = words_per_chunk - words_per_overlap
        positions = range(0, max(1, len(words) - words_per_overlap), step)

        for pos in positions:
            window_words = words[pos: pos + words_per_chunk]
            if not window_words:
                continue

            chunk_text  = " ".join(window_words)
            token_count = len(window_words)   # approximate

            # Stable chunk_id based on content hash
            chunk_id = hashlib.md5(
                f"{contract_id}:{section['clause_id']}:{pos}".encode()
            ).hexdigest()[:16]

            chunks.append(PPAChunk(
                chunk_id      = chunk_id,
                contract_id   = contract_id,
                chunk_text    = chunk_text,
                clause_id     = section["clause_id"],
                section_title = section["section_title"],
                page_number   = section["page"],
                token_count   = token_count,
                char_start    = pos * 5,   # rough character offset
            ))

    return chunks
