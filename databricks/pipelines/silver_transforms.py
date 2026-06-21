"""
Silver Layer — Cleaned, Validated, Conformed Data
---------------------------------------------------
Parses Bronze raw JSON payloads into typed, validated schemas.
DLT Expectations enforce data quality — bad rows go to quarantine, not main table.

Run as part of the same Lakeflow pipeline as bronze_ingestion.py
  Catalog: renewiq | Schema: silver  (schema-qualified per table)
"""

from __future__ import annotations

from typing import Iterator

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, DateType, IntegerType, BooleanType,
)

# ── Schema definitions ────────────────────────────────────────────────────────

EPEX_ROW_SCHEMA = StructType([
    StructField("hour", IntegerType()),
    StructField("delivery_ts", StringType()),
    StructField("price_eur_mwh", DoubleType()),
    StructField("is_negative", BooleanType()),
    StructField("market", StringType()),
])

ENTSO_ROW_SCHEMA = StructType([
    StructField("delivery_ts", StringType()),
    StructField("hour", IntegerType()),
    StructField("wind_onshore_mw", DoubleType()),
    StructField("wind_offshore_mw", DoubleType()),
    StructField("solar_mw", DoubleType()),
    StructField("total_renewable_mw", DoubleType()),
    StructField("total_mw", DoubleType()),
    StructField("renewable_pct", DoubleType()),
    StructField("oversupply_flag", BooleanType()),
    StructField("country", StringType()),
])

GOPACS_ROW_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("dso_zone", StringType()),
    StructField("direction", StringType()),
    StructField("start_time", StringType()),
    StructField("end_time", StringType()),
    StructField("mw_needed", DoubleType()),
    StructField("price_eur_mwh", DoubleType()),
    StructField("status", StringType()),
])


# ── EPEX Silver ───────────────────────────────────────────────────────────────

@dlt.table(
    name="silver.epex_dayahead",
    comment="Parsed, typed EPEX NL day-ahead prices with negative price flag",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("valid_price_range", "price_eur_mwh BETWEEN -500 AND 4000")
@dlt.expect_or_drop("valid_hour", "hour BETWEEN 0 AND 23")
@dlt.expect_or_drop("non_null_timestamp", "delivery_ts IS NOT NULL")
@dlt.expect_or_drop("non_null_market", "market IS NOT NULL")
def transform_epex():
    raw = dlt.read_stream("bronze.epex_dayahead_raw")
    exploded = (
        raw
        .withColumn("rows", F.from_json(F.col("raw_payload"), f"array<{_schema_ddl(EPEX_ROW_SCHEMA)}>"))
        .select(
            F.col("market"),
            F.col("fetch_date").cast(DateType()).alias("fetch_date"),
            F.col("ingestion_ts"),
            F.explode("rows").alias("row"),
        )
        .select(
            F.col("row.hour").alias("hour"),
            F.to_timestamp(F.col("row.delivery_ts")).alias("delivery_ts"),
            F.col("row.price_eur_mwh").cast(DoubleType()).alias("price_eur_mwh"),
            (F.col("row.price_eur_mwh") < 0).alias("is_negative"),
            F.col("row.market").alias("market"),
            F.col("fetch_date"),
            F.col("ingestion_ts"),
        )
    )
    return exploded.withColumn("delivery_date", F.to_date("delivery_ts"))


# ── ENTSO-E Silver ────────────────────────────────────────────────────────────

@dlt.table(
    name="silver.entso_generation",
    comment="Parsed ENTSO-E actual generation per production type — NL hourly",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("non_null_timestamp", "delivery_ts IS NOT NULL")
@dlt.expect_or_drop("valid_total_mw", "total_mw >= 0")
@dlt.expect_or_drop("valid_renewable_pct", "renewable_pct BETWEEN 0 AND 1")
def transform_entso():
    raw = dlt.read_stream("bronze.entso_generation_raw")
    return (
        raw
        .withColumn("rows", F.from_json(F.col("raw_payload"), f"array<{_schema_ddl(ENTSO_ROW_SCHEMA)}>"))
        .select(F.explode("rows").alias("row"), F.col("ingestion_ts"))
        .select(
            F.to_timestamp(F.col("row.delivery_ts")).alias("delivery_ts"),
            F.to_date(F.col("row.delivery_ts")).alias("delivery_date"),
            F.col("row.hour").alias("hour"),
            F.col("row.wind_onshore_mw").cast(DoubleType()).alias("wind_onshore_mw"),
            F.col("row.wind_offshore_mw").cast(DoubleType()).alias("wind_offshore_mw"),
            F.col("row.solar_mw").cast(DoubleType()).alias("solar_mw"),
            F.col("row.total_renewable_mw").cast(DoubleType()).alias("total_renewable_mw"),
            F.col("row.total_mw").cast(DoubleType()).alias("total_mw"),
            F.col("row.renewable_pct").cast(DoubleType()).alias("renewable_pct"),
            F.col("row.oversupply_flag").alias("oversupply_flag"),
            F.col("row.country").alias("country"),
            F.col("ingestion_ts"),
        )
    )


# ── GOPACS Silver ─────────────────────────────────────────────────────────────

@dlt.table(
    name="silver.gopacs_congestion_events",
    comment="Parsed GOPACS grid congestion events with typed timestamps",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("non_null_zone", "dso_zone IS NOT NULL")
@dlt.expect_or_drop("valid_direction", "direction IN ('upward', 'downward')")
@dlt.expect_or_drop("positive_mw", "mw_needed > 0")
@dlt.expect_or_drop("valid_time_window", "end_time > start_time")
def transform_gopacs():
    raw = dlt.read_stream("bronze.gopacs_announcements_raw")
    return (
        raw
        .withColumn("rows", F.from_json(F.col("raw_payload"), f"array<{_schema_ddl(GOPACS_ROW_SCHEMA)}>"))
        .select(F.explode("rows").alias("row"), F.col("ingestion_ts"))
        .select(
            F.col("row.event_id").alias("event_id"),
            F.col("row.dso_zone").alias("dso_zone"),
            F.col("row.direction").alias("direction"),
            F.to_timestamp(F.col("row.start_time")).alias("start_time"),
            F.to_timestamp(F.col("row.end_time")).alias("end_time"),
            F.col("row.mw_needed").cast(DoubleType()).alias("mw_needed"),
            F.col("row.price_eur_mwh").cast(DoubleType()).alias("price_eur_mwh"),
            F.col("row.status").alias("status"),
            F.col("ingestion_ts"),
        )
        .dropDuplicates(["event_id"])
    )


# ── PPA Contract Chunks ───────────────────────────────────────────────────────

@dlt.table(
    name="silver.ppa_contract_chunks",
    comment="Section-aware text chunks from PPA PDFs with risk category tags",
    table_properties={"quality": "silver"},
)
@dlt.expect("non_null_chunk_text", "chunk_text IS NOT NULL")
@dlt.expect("non_null_contract_id", "contract_id IS NOT NULL")
def transform_ppa_chunks():
    import sys
    REPO_PATH = "/Workspace/Repos/venkatesh8484/renewIQ"
    if REPO_PATH not in sys.path:
        sys.path.insert(0, REPO_PATH)

    from pyspark.sql.types import (
        StructType, StructField, StringType as ST, IntegerType as IT,
    )
    import pandas as pd
    from pathlib import Path

    CHUNK_SCHEMA = StructType([
        StructField("chunk_id",      ST()),
        StructField("contract_id",   ST()),
        StructField("chunk_text",    ST()),
        StructField("clause_id",     ST()),
        StructField("section_title", ST()),
        StructField("page_number",   IT()),
        StructField("token_count",   IT()),
        StructField("risk_category", ST()),
        StructField("char_start",    IT()),
    ])

    def _extract_chunks(pdf_rows):
        import sys
        if REPO_PATH not in sys.path:
            sys.path.insert(0, REPO_PATH)

        from src.ingestion.pdf_parser import parse_ppa_pdf
        from src.ingestion.clause_tagger import tag_chunks_batch

        for batch in pdf_rows:
            all_chunks = []
            for _, row in batch.iterrows():
                pdf_path = Path(row["path"])
                contract_id = row["contract_id"]
                try:
                    chunks = parse_ppa_pdf(pdf_path, contract_id=contract_id)
                    tag_chunks_batch(chunks)
                    for c in chunks:
                        all_chunks.append({
                            "chunk_id":      c.chunk_id,
                            "contract_id":   c.contract_id,
                            "chunk_text":    c.chunk_text,
                            "clause_id":     c.clause_id,
                            "section_title": c.section_title,
                            "page_number":   c.page_number,
                            "token_count":   c.token_count,
                            "risk_category": c.risk_category,
                            "char_start":    c.char_start,
                        })
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Failed to parse {pdf_path}: {e}")

            import pandas as _pd
            if all_chunks:
                yield _pd.DataFrame(all_chunks)
            else:
                yield _pd.DataFrame(columns=[f.name for f in CHUNK_SCHEMA.fields])

    raw = dlt.read_stream("bronze.ppa_documents_raw")
    return (
        raw
        .select("path", "contract_id", "ingestion_ts")
        .mapInPandas(_extract_chunks, schema=CHUNK_SCHEMA)
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _schema_ddl(schema: StructType) -> str:
    """Convert a StructType to a DDL string for from_json."""
    return schema.simpleString()
