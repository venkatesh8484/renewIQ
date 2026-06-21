# Databricks notebook source
# MAGIC %md
# MAGIC # RenewIQ — Unity Catalog Setup
# MAGIC
# MAGIC Run this notebook **once** to create the catalog, schemas, and grants.
# MAGIC Requires: Unity Catalog enabled workspace + admin or catalog CREATE privilege.

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

# MAGIC %md ## 2. Configure Storage (ADLS Gen2)

# COMMAND ----------

# DBTITLE 1,Verify ADLS Gen2 access
STORAGE_ACCOUNT = "renewiqstorage"  # Change to your storage account name

try:
    files = dbutils.fs.ls(f"abfss://raw@{STORAGE_ACCOUNT}.dfs.core.windows.net/")
    print(f"✓ ADLS Gen2 accessible — {len(files)} items in bronze container")
except Exception as e:
    print(f"⚠ ADLS access issue: {e}")
    print("Set up storage credentials via Databricks Secret Scope or Managed Identity")

# COMMAND ----------

# MAGIC %md ## 3. Create Lakeflow Pipeline

# COMMAND ----------

# DBTITLE 1,Create Lakeflow DLT pipeline via SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineCluster, PipelineLibrary, NotebookLibrary

w = WorkspaceClient()

pipeline_settings = {
    "name": "renewiq_data_pipeline",
    "catalog": "renewiq",
    "target": "silver",   # DLT uses this as the target schema for Silver tables
    "clusters": [
        {
            "label": "default",
            "num_workers": 1,
            "spark_conf": {
                "renewiq.storage_root": f"abfss://raw@{STORAGE_ACCOUNT}.dfs.core.windows.net"
            },
        }
    ],
    "libraries": [
        {"notebook": {"path": "/Repos/renewiq/databricks/pipelines/bronze_ingestion"}},
        {"notebook": {"path": "/Repos/renewiq/databricks/pipelines/silver_transforms"}},
        {"notebook": {"path": "/Repos/renewiq/databricks/pipelines/gold_features"}},
    ],
    "continuous": False,   # Triggered mode — run on schedule or manually
    "development": True,   # Set to False for production
}

try:
    pipeline = w.pipelines.create(**pipeline_settings)
    print(f"✓ Lakeflow pipeline created: {pipeline.pipeline_id}")
    print(f"  → Start it in the Databricks UI: Workflows → Delta Live Tables")
except Exception as e:
    print(f"Pipeline creation error: {e}")
    print("You can also create the pipeline manually in the Databricks UI")

# COMMAND ----------

# MAGIC %md ## 4. Verify Unity Catalog Structure

# COMMAND ----------

# DBTITLE 1,Show catalog structure
print("Unity Catalog structure:")
for schema in ["bronze", "silver", "gold", "agents", "models"]:
    tables = spark.sql(f"SHOW TABLES IN renewiq.{schema}").collect()
    print(f"\n  renewiq.{schema}/ ({len(tables)} tables)")
    for t in tables:
        print(f"    ├── {t.tableName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **Run the Lakeflow pipeline**: Workflows → Delta Live Tables → renewiq_data_pipeline → Start
# MAGIC 2. **Seed Bronze data**: `python scripts/seed_market_data.py --days 90`
# MAGIC 3. **Set up Vector Search**: Run `02_setup_vector_search.py` (Phase 3)
# MAGIC 4. **Register UC tools**: Run `03_register_uc_tools.py` (Phase 4)
