# Databricks notebook source
# MAGIC %md
# MAGIC # RenewIQ — Unity Catalog Setup
# MAGIC
# MAGIC Run this notebook **once** to create the catalog, schemas, volume, and pipeline.
# MAGIC Requires: Unity Catalog enabled workspace + CREATE CATALOG privilege.

# COMMAND ----------

# MAGIC %md ## 1. Create Catalog and Schemas

# COMMAND ----------

# DBTITLE 1,Create renewiq catalog
spark.sql("CREATE CATALOG IF NOT EXISTS renewiq COMMENT 'RenewIQ PPA Intelligence Copilot'")
spark.sql("USE CATALOG renewiq")
print("✓ Catalog 'renewiq' ready")

# COMMAND ----------

# DBTITLE 1,Create schemas (Medallion + Agents + Models)
schemas = {
    "bronze":  "Raw ingested data — append-only, no transformations",
    "silver":  "Cleaned, validated, conformed data",
    "gold":    "Agent-ready feature tables and aggregations",
    "agents":  "Unity Catalog tools (SQL + Python functions) for LangGraph agents",
    "models":  "MLflow registered models and agent versions",
}

for schema, comment in schemas.items():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS renewiq.{schema} COMMENT '{comment}'")
    print(f"✓ Schema 'renewiq.{schema}' ready")

# COMMAND ----------

# MAGIC %md ## 2. Create Unity Catalog Volume (replaces DBFS root)
# MAGIC
# MAGIC UC Volumes are the modern replacement for DBFS in Unity Catalog workspaces.
# MAGIC Path format: `/Volumes/<catalog>/<schema>/<volume>/`

# COMMAND ----------

# DBTITLE 1,Create managed volume for raw seed data
spark.sql("""
    CREATE VOLUME IF NOT EXISTS renewiq.bronze.raw_data
    COMMENT 'Raw seed data: EPEX, ENTSO-E, GOPACS JSON files and PPA PDFs'
""")

VOLUME_PATH = "/Volumes/renewiq/bronze/raw_data"

# Create subdirectory structure inside the volume
for subdir in ["epex", "entso_generation", "gopacs", "contracts", "_schemas"]:
    dbutils.fs.mkdirs(f"{VOLUME_PATH}/{subdir}")
    print(f"✓ {VOLUME_PATH}/{subdir}/")

print(f"\n✓ Volume ready at {VOLUME_PATH}")

# COMMAND ----------

# DBTITLE 1,Verify volume contents (after uploading seed data)
try:
    files = dbutils.fs.ls(f"{VOLUME_PATH}/epex/")
    print(f"✓ EPEX files found: {len(files)}")
    for f in files[:3]:
        print(f"    ├── {f.name}")
except Exception as e:
    print(f"⚠ No EPEX files yet — upload seed data first (see instructions below)")

# COMMAND ----------

# MAGIC %md ## 3. Seed Bronze Data (runs fetchers directly in cluster)
# MAGIC
# MAGIC This cell runs the seed script **inside the cluster** — no local upload needed.
# MAGIC Uses mock data if ENTSO_E_API_KEY is not set.

# COMMAND ----------

# DBTITLE 1,Install dependencies and run seed script
%pip install pandas numpy httpx beautifulsoup4 -q

# COMMAND ----------

# DBTITLE 1,Generate and write seed data directly to UC Volume
import sys
import json
import importlib
from datetime import date, timedelta
from pathlib import Path

# Add repo to path (adjust if your Repo path differs)
REPO_PATH = "/Workspace/Repos/venkatesh8484/renewIQ"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Re-import after pip install
import importlib
import src.ingestion.epex_fetcher as epex_mod
import src.ingestion.entso_fetcher as entso_mod
import src.ingestion.gopacs_fetcher as gopacs_mod
importlib.reload(epex_mod)
importlib.reload(entso_mod)
importlib.reload(gopacs_mod)

from src.ingestion.epex_fetcher import EPEXFetcher
from src.ingestion.entso_fetcher import ENTSOFetcher
from src.ingestion.gopacs_fetcher import GOPACSFetcher

DAYS = 90
end_date = date.today() - timedelta(days=1)
start_date = end_date - timedelta(days=DAYS - 1)

print(f"Seeding {DAYS} days: {start_date} → {end_date}")
print(f"Writing to: {VOLUME_PATH}\n")

# ── EPEX ──────────────────────────────────────────────────────────────────────
epex = EPEXFetcher()
df_epex = epex.fetch_day_ahead(start_date, end=end_date, market="NL")

epex_count = 0
for day, group in df_epex.groupby("delivery_date"):
    payload = json.dumps({
        "source_api": "entso-e-transparency",
        "market": "NL",
        "fetch_date": str(day),
        "ingestion_ts": str(date.today()),
        "raw_payload": group.drop(columns=["delivery_date"]).to_json(orient="records", date_format="iso"),
    })
    dbutils.fs.put(f"{VOLUME_PATH}/epex/{day}.json", payload, overwrite=True)
    epex_count += 1

print(f"✓ EPEX: {epex_count} daily files written ({len(df_epex)} rows)")

# ── ENTSO-E ───────────────────────────────────────────────────────────────────
entso = ENTSOFetcher()
df_entso = entso.fetch_generation(start_date, end=end_date)

entso_count = 0
for day, group in df_entso.groupby("delivery_date"):
    payload = json.dumps({
        "source_api": "entso-e-transparency",
        "country": "NL",
        "fetch_date": str(day),
        "ingestion_ts": str(date.today()),
        "raw_payload": group.drop(columns=["delivery_date"]).to_json(orient="records", date_format="iso"),
    })
    dbutils.fs.put(f"{VOLUME_PATH}/entso_generation/{day}.json", payload, overwrite=True)
    entso_count += 1

print(f"✓ ENTSO-E: {entso_count} daily files written ({len(df_entso)} rows)")

# ── GOPACS ────────────────────────────────────────────────────────────────────
gopacs = GOPACSFetcher()
df_gopacs = gopacs.fetch_announcements(lookback_hours=48)

payload = json.dumps({
    "source": "gopacs.eu",
    "ingestion_ts": str(date.today()),
    "raw_payload": df_gopacs.to_json(orient="records", date_format="iso"),
})
dbutils.fs.put(f"{VOLUME_PATH}/gopacs/{date.today()}.json", payload, overwrite=True)

print(f"✓ GOPACS: {len(df_gopacs)} events written")
print(f"\nSeed complete ✓")

# COMMAND ----------

# MAGIC %md ## 4. Create Lakeflow Pipeline

# COMMAND ----------

# DBTITLE 1,Create Lakeflow DLT pipeline via REST API (serverless)
import requests

ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host  = ctx.apiUrl().get()

# Derive repo root robustly — works whether notebook lives under /Repos/ or /Users/
# Path example: /Users/venkatesh8484@gmail.com/renewIQ/databricks/notebooks/01_setup_unity_catalog
# Go up 3 segments from the file: /notebooks/<file> + /databricks → repo root
notebook_path = ctx.notebookPath().get()
repo_root     = "/".join(notebook_path.split("/")[:-3])   # strips /databricks/notebooks/<file>
pipeline_base = f"{repo_root}/databricks/pipelines"

print(f"Notebook path : {notebook_path}")
print(f"Repo root     : {repo_root}")
print(f"Pipeline base : {pipeline_base}")

pipeline_config = {
    "name": "renewiq_data_pipeline",
    "catalog": "renewiq",
    "target": "silver",
    "serverless": True,                        # required by this workspace
    "configuration": {                         # Spark conf for serverless (replaces clusters[].spark_conf)
        "renewiq.storage_root": VOLUME_PATH,
    },
    "libraries": [
        {"file": {"path": f"{pipeline_base}/bronze_ingestion.py"}},
        {"file": {"path": f"{pipeline_base}/silver_transforms.py"}},
        {"file": {"path": f"{pipeline_base}/gold_features.py"}},
    ],
    "continuous": False,
    "development": True,
}

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp    = requests.post(f"{host}/api/2.0/pipelines", json=pipeline_config, headers=headers)

if resp.status_code == 200:
    pipeline_id = resp.json().get("pipeline_id")
    print(f"\n✓ Pipeline created — ID: {pipeline_id}")
    print(f"  Storage : {VOLUME_PATH}")
    print(f"  Mode    : serverless, triggered")
    print(f"\n→ Workflows → Delta Live Tables → renewiq_data_pipeline → Start")
elif resp.status_code == 400 and "already exists" in resp.text.lower():
    print("Pipeline already exists → Workflows → Delta Live Tables")
else:
    print(f"Error {resp.status_code}: {resp.text}")

# COMMAND ----------

# MAGIC %md ## 5. Verify After Pipeline Run

# COMMAND ----------

# DBTITLE 1,Verify Silver tables populated (run after pipeline completes)
queries = {
    "EPEX rows (expect ~2160 for 90 days)":
        "SELECT COUNT(*), MIN(delivery_date), MAX(delivery_date) FROM renewiq.silver.epex_dayahead",
    "Negative price hours":
        "SELECT COUNT(*) as neg_hours, ROUND(AVG(price_eur_mwh),2) as avg_neg_price FROM renewiq.silver.epex_dayahead WHERE is_negative = true",
    "GOPACS events":
        "SELECT COUNT(*), COUNT(DISTINCT dso_zone) as zones FROM renewiq.silver.gopacs_congestion_events",
    "Gold market signals":
        "SELECT signal_type, severity, COUNT(*) as hours FROM renewiq.gold.market_risk_signals GROUP BY 1,2 ORDER BY 1,2",
    "Gold portfolio exposure (negative days)":
        "SELECT COUNT(*) as neg_days, ROUND(AVG(negative_hours),1) as avg_neg_hrs FROM renewiq.gold.portfolio_exposure_daily",
    "Gold price features (volatility)":
        "SELECT COUNT(*) as rows, COUNT(CASE WHEN volatility_flag THEN 1 END) as volatile_hours FROM renewiq.gold.hourly_price_features",
}

for label, sql in queries.items():
    print(f"\n── {label}")
    try:
        spark.sql(sql).show()
    except Exception as e:
        print(f"  Not ready yet: {e}")
