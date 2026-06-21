"""
Bronze Layer — Raw Ingestion via Databricks Lakeflow Declarative Pipelines (DLT)
----------------------------------------------------------------------------------
All sources ingested as-is. No transformations. Append-only Delta tables.

Run as a Databricks Lakeflow pipeline:
  Catalog: renewiq | Schema: bronze
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, TimestampType, DateType, LongType

# ── Storage paths ─────────────────────────────────────────────────────────────
# Dev: Unity Catalog Volume (works when DBFS root is disabled — UC-enforced workspaces)
#   /Volumes/<catalog>/<schema>/<volume>/
# Prod override: set renewiq.storage_root in pipeline config to your ADLS path
#   e.g. abfss://raw@<storage_account>.dfs.core.windows.net
STORAGE_ROOT = spark.conf.get("renewiq.storage_root", "/Volumes/renewiq/bronze/raw_data")


# ── EPEX Day-Ahead Prices ─────────────────────────────────────────────────────

@dlt.table(
    name="epex_dayahead_raw",
    comment="Raw EPEX NL day-ahead hourly prices from ENTSO-E Transparency — append-only",
    table_properties={
        "quality": "bronze",
        "pipelines.reset.allowed": "true",
    },
)
def ingest_epex_raw():
    """
    Auto Loader picks up JSON files dropped by seed_market_data.py
    or the daily scheduler job.
    Schema: {ingestion_ts, source_api, market, fetch_date, raw_payload}
    """
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{STORAGE_ROOT}/_schemas/epex")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{STORAGE_ROOT}/epex/")
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# ── ENTSO-E Generation Mix ────────────────────────────────────────────────────

@dlt.table(
    name="entso_generation_raw",
    comment="Raw ENTSO-E actual generation per production type — append-only",
    table_properties={"quality": "bronze"},
)
def ingest_entso_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{STORAGE_ROOT}/_schemas/entso")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{STORAGE_ROOT}/entso_generation/")
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# ── GOPACS Congestion Announcements ───────────────────────────────────────────

@dlt.table(
    name="gopacs_announcements_raw",
    comment="Raw GOPACS grid congestion market announcements — append-only",
    table_properties={"quality": "bronze"},
)
def ingest_gopacs_raw():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{STORAGE_ROOT}/_schemas/gopacs")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{STORAGE_ROOT}/gopacs/")
        .withColumn("ingestion_ts", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


# ── PPA Contract Documents ────────────────────────────────────────────────────

@dlt.table(
    name="ppa_documents_raw",
    comment="Raw PPA PDF binary metadata — content extracted in Silver layer",
    table_properties={"quality": "bronze"},
)
def ingest_ppa_docs():
    """
    Auto Loader monitors the ADLS contracts container for new PDF uploads.
    Stores file metadata only — actual content extracted by Silver pipeline.
    """
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("cloudFiles.schemaLocation", f"{STORAGE_ROOT}/_schemas/ppa_docs")
        .load(
            spark.conf.get(
                "renewiq.contracts_root",
                "/Volumes/renewiq/bronze/raw_data/contracts"
            )
        )
        .select(
            F.col("path"),
            F.col("modificationTime").alias("last_modified"),
            F.col("length").alias("file_size_bytes"),
            F.regexp_extract(F.col("path"), r"([^/]+)\.pdf$", 1).alias("contract_id"),
            F.current_timestamp().alias("ingestion_ts"),
        )
    )
