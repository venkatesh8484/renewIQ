"""
Silver Layer — Cleaned, Validated, Conformed Data
---------------------------------------------------
Parses Bronze raw JSON payloads into typed, validated schemas.
DLT Expectations enforce data quality — bad rows go to quarantine, not main table.

Run as part of the same Lakeflow pipeline as bronze_ingestion.py
  Catalog: renewiq | Schema: silver
"""

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
    name="epex_dayahead",
    comment="Parsed, typed EPEX NL day-ahead prices with negative price flag",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("valid_price_range", "price_eur_mwh BETWEEN -500 AND 4000")
@dlt.expect_or_drop("valid_hour", "hour BETWEEN 0 AND 23")
@dlt.expect_or_drop("non_null_timestamp", "delivery_ts IS NOT NULL")
@dlt.expect_or_drop("non_null_market", "market IS NOT NULL")
def transform_epex():
    raw = dlt.read_stream("epex_dayahead_raw")
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
    # Derive delivery_date from timestamp
    return exploded.withColumn("delivery_date", F.to_date("delivery_ts"))


# ── ENTSO-E Silver ────────────────────────────────────────────────────────────

@dlt.table(
    name="entso_generation",
    comment="Parsed ENTSO-E actual generation per production type — NL hourly",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("non_null_timestamp", "delivery_ts IS NOT NULL")
@dlt.expect_or_drop("valid_total_mw", "total_mw >= 0")
@dlt.expect_or_drop("valid_renewable_pct", "renewable_pct BETWEEN 0 AND 1")
def transform_entso():
    raw = dlt.read_stream("entso_generation_raw")
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
    name="gopacs_congestion_events",
    comment="Parsed GOPACS grid congestion events with typed timestamps",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("non_null_zone", "dso_zone IS NOT NULL")
@dlt.expect_or_drop("valid_direction", "direction IN ('upward', 'downward')")
@dlt.expect_or_drop("positive_mw", "mw_needed > 0")
@dlt.expect_or_drop("valid_time_window", "end_time > start_time")
def transform_gopacs():
    raw = dlt.read_stream("gopacs_announcements_raw")
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
        # Deduplicate by event_id (GOPACS may resend same event in updates)
        .dropDuplicates(["event_id"])
    )


# ── PPA Contract Chunks ───────────────────────────────────────────────────────
# (Fully implemented in Phase 3 — PDF parsing + chunk extraction)
# Stub table registered here so Gold can reference it by name from Phase 2.

@dlt.table(
    name="ppa_contract_chunks",
    comment="Section-aware text chunks from PPA PDFs with risk category tags — populated in Phase 3",
    table_properties={"quality": "silver"},
)
def transform_ppa_chunks():
    """
    Phase 3 implements the full PyMuPDF extraction + risk tagging UDF here.
    For now returns the raw metadata so the table exists in Unity Catalog.
    """
    return (
        dlt.read_stream("ppa_documents_raw")
        .select(
            F.col("contract_id"),
            F.col("path").alias("source_path"),
            F.col("file_size_bytes"),
            F.col("ingestion_ts"),
            # Placeholders — Phase 3 populates real chunk columns
            F.lit(None).cast(StringType()).alias("chunk_text"),
            F.lit(None).cast(StringType()).alias("clause_id"),
            F.lit(None).cast(StringType()).alias("section_title"),
            F.lit(None).cast(IntegerType()).alias("page_number"),
            F.lit(None).cast(StringType()).alias("risk_category"),
            F.lit(None).cast(IntegerType()).alias("token_count"),
        )
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _schema_ddl(schema: StructType) -> str:
    """Convert a StructType to a DDL string for from_json."""
    return schema.simpleString()
