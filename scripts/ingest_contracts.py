#!/usr/bin/env python3
"""
ingest_contracts.py — End-to-End PPA Contract Ingestion
---------------------------------------------------------
Generates synthetic PPA PDFs (or uses existing ones), parses them into
clause-aware chunks, tags risk categories, and writes to a local Parquet
file for local dev OR to renewiq.silver.ppa_contract_chunks on Databricks.

Usage:
    # Generate fresh PDFs + ingest to local Parquet (default)
    python scripts/ingest_contracts.py

    # Use existing PDFs in a directory
    python scripts/ingest_contracts.py --pdf-dir data/contracts/

    # Write to Databricks (requires DATABRICKS_HOST + DATABRICKS_TOKEN)
    python scripts/ingest_contracts.py --target databricks

    # Full run: generate + parse + write to both
    python scripts/ingest_contracts.py --target local,databricks

Options:
    --pdf-dir PATH       Directory with existing PDFs (default: generate fresh)
    --output-dir PATH    Local output dir for Parquet (default: data/silver/)
    --target STR         Output target: local, databricks, or both (default: local)
    --contract-ids LIST  Comma-separated contract IDs to process (default: all 5)
    --dry-run            Parse only; print chunk stats, don't write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_contracts")

# Add repo root to path so src.ingestion.* imports work when run as a script
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest PPA contracts into Silver layer")
    p.add_argument("--pdf-dir",     type=Path, default=None,
                   help="Directory of existing PDFs (skip generation if set)")
    p.add_argument("--output-dir",  type=Path, default=Path("data/silver"),
                   help="Local output directory for Parquet (default: data/silver)")
    p.add_argument("--target",      default="local",
                   help="Output target(s): local, databricks, or local,databricks")
    p.add_argument("--contract-ids", default=None,
                   help="Comma-separated contract IDs to process (all if omitted)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Parse + tag only; print stats, don't write")
    return p.parse_args()


def generate_pdfs(output_dir: Path) -> Path:
    """Run generate_synthetic_ppas.py and return the output directory."""
    from scripts.generate_synthetic_ppas import PPAGenerator, CONTRACTS  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating {len(CONTRACTS)} synthetic PPA PDFs → {output_dir}")

    for spec in CONTRACTS:
        out_path = output_dir / f"{spec.contract_id}.pdf"
        gen = PPAGenerator(spec)
        gen.generate(out_path)
        logger.info(f"  ✓ {spec.contract_id}.pdf ({out_path.stat().st_size // 1024} KB)")

    return output_dir


def ingest_pdfs(
    pdf_dir: Path,
    contract_ids: Optional[list[str]] = None,
) -> list[dict]:
    """
    Parse all PDFs in pdf_dir, tag risk categories, return list of chunk dicts.
    """
    from src.ingestion.pdf_parser import parse_ppa_pdf
    from src.ingestion.clause_tagger import tag_chunks_batch

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    if contract_ids:
        pdf_files = [f for f in pdf_files if f.stem in contract_ids]
        logger.info(f"Filtered to {len(pdf_files)} PDFs: {[f.stem for f in pdf_files]}")

    all_chunks: list[dict] = []

    for pdf_path in pdf_files:
        contract_id = pdf_path.stem
        logger.info(f"Parsing {pdf_path.name}...")

        chunks = parse_ppa_pdf(pdf_path, contract_id=contract_id)
        tag_chunks_batch(chunks)

        chunk_dicts = [
            {
                "chunk_id":      c.chunk_id,
                "contract_id":   c.contract_id,
                "chunk_text":    c.chunk_text,
                "clause_id":     c.clause_id,
                "section_title": c.section_title,
                "page_number":   c.page_number,
                "token_count":   c.token_count,
                "risk_category": c.risk_category,
                "char_start":    c.char_start,
            }
            for c in chunks
        ]
        all_chunks.extend(chunk_dicts)

        # Per-contract summary
        tagged = sum(1 for c in chunks if c.risk_category is not None)
        by_cat: dict[str, int] = {}
        for c in chunks:
            if c.risk_category:
                by_cat[c.risk_category] = by_cat.get(c.risk_category, 0) + 1

        logger.info(
            f"  → {len(chunks)} chunks | {tagged} tagged | "
            + ", ".join(f"{k}:{v}" for k, v in sorted(by_cat.items()))
        )

    return all_chunks


def write_local(chunks: list[dict], output_dir: Path) -> Path:
    """Write chunks to a Parquet file in output_dir."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install pandas --break-system-packages")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ppa_contract_chunks.parquet"

    df = pd.DataFrame(chunks)
    df.to_parquet(out_path, index=False)
    logger.info(f"✓ Wrote {len(df)} chunks → {out_path}")
    return out_path


def write_databricks(chunks: list[dict]) -> None:
    """
    Write chunks to renewiq.silver.ppa_contract_chunks on Databricks.
    Requires DATABRICKS_HOST and DATABRICKS_TOKEN in env / .env.
    Uses delta-spark to merge (upsert) by chunk_id.
    """
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip
    except ImportError:
        raise ImportError(
            "PySpark + delta-spark required. "
            "Run: pip install pyspark delta-spark --break-system-packages"
        )

    import pandas as pd

    logger.info("Connecting to Databricks SparkSession (remote)...")
    spark = (
        SparkSession.builder
        .appName("renewiq-ingest-contracts")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    df_pandas = pd.DataFrame(chunks)
    df_spark  = spark.createDataFrame(df_pandas)
    TARGET    = "renewiq.silver.ppa_contract_chunks"

    df_spark.createOrReplaceTempView("_chunks_to_merge")
    spark.sql(f"""
        MERGE INTO {TARGET} AS t
        USING _chunks_to_merge AS s ON t.chunk_id = s.chunk_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    count = spark.sql(f"SELECT COUNT(*) AS n FROM {TARGET}").collect()[0]["n"]
    logger.info(f"✓ Upserted {len(chunks)} chunks → {TARGET} (total rows: {count})")


def print_stats(chunks: list[dict]) -> None:
    """Print a summary table of chunk counts by contract and risk category."""
    from collections import Counter

    total = len(chunks)
    by_contract: Counter = Counter(c["contract_id"] for c in chunks)
    by_category: Counter = Counter(c["risk_category"] for c in chunks if c["risk_category"])
    untagged = sum(1 for c in chunks if not c["risk_category"])

    print(f"\n{'='*60}")
    print(f"  Total chunks: {total}")
    print(f"\n  By contract:")
    for cid, n in sorted(by_contract.items()):
        print(f"    {cid:<45} {n:>4} chunks")
    print(f"\n  By risk category:")
    for cat, n in sorted(by_category.items()):
        pct = n / total * 100
        print(f"    {cat:<25} {n:>4} ({pct:.1f}%)")
    print(f"    {'(untagged)':<25} {untagged:>4} ({untagged/total*100:.1f}%)")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    targets = {t.strip() for t in args.target.split(",")}
    contract_ids = (
        [c.strip() for c in args.contract_ids.split(",")]
        if args.contract_ids else None
    )

    # ── Step 1: Get PDFs ──────────────────────────────────────────────────────
    if args.pdf_dir:
        pdf_dir = args.pdf_dir
        logger.info(f"Using existing PDFs from {pdf_dir}")
    else:
        pdf_dir = Path("data/contracts")
        generate_pdfs(pdf_dir)

    # ── Step 2: Parse + tag ───────────────────────────────────────────────────
    chunks = ingest_pdfs(pdf_dir, contract_ids=contract_ids)
    print_stats(chunks)

    if args.dry_run:
        logger.info("Dry run — skipping write")
        return

    # ── Step 3: Write outputs ─────────────────────────────────────────────────
    if "local" in targets:
        parquet_path = write_local(chunks, args.output_dir)
        logger.info(f"→ Open with: pd.read_parquet('{parquet_path}')")

    if "databricks" in targets:
        write_databricks(chunks)

    logger.info("✓ Contract ingestion complete")


if __name__ == "__main__":
    main()
