# Databricks notebook source
# MAGIC %md
# MAGIC # RenewIQ — Vector Search Setup
# MAGIC
# MAGIC Run this notebook **after** the Lakeflow pipeline has completed at least one run
# MAGIC and `renewiq.silver.ppa_contract_chunks` is populated.
# MAGIC
# MAGIC Creates:
# MAGIC - A **Databricks Vector Search endpoint** (`renewiq-vs-endpoint`)
# MAGIC - A **Delta Sync index** over `ppa_contract_chunks` using Databricks-managed embeddings
# MAGIC
# MAGIC Requires: `databricks-vectorsearch` SDK

# COMMAND ----------

# MAGIC %md ## 0. Prerequisites

# COMMAND ----------

# DBTITLE 1,Install Vector Search SDK
%pip install databricks-vectorsearch>=0.40 -q
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Verify source table exists and has rows
row_count = spark.sql(
    "SELECT COUNT(*) as n FROM renewiq.silver.ppa_contract_chunks WHERE chunk_text IS NOT NULL"
).collect()[0]["n"]

print(f"✓ ppa_contract_chunks has {row_count} rows with chunk text")

if row_count == 0:
    raise ValueError(
        "No chunks found — run the Lakeflow pipeline first, then re-run this notebook.\n"
        "If you're testing locally, run scripts/ingest_contracts.py to populate the table."
    )

# COMMAND ----------

# MAGIC %md ## 1. Enable Change Data Feed on source table
# MAGIC
# MAGIC Delta Sync index requires CDF to track incremental updates.

# COMMAND ----------

# DBTITLE 1,Enable CDF on ppa_contract_chunks
spark.sql("""
    ALTER TABLE renewiq.silver.ppa_contract_chunks
    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print("✓ Change Data Feed enabled on renewiq.silver.ppa_contract_chunks")

# COMMAND ----------

# MAGIC %md ## 2. Create Vector Search Endpoint

# COMMAND ----------

# DBTITLE 1,Create VS endpoint (idempotent)
from databricks.vector_search.client import VectorSearchClient

VS_ENDPOINT  = "renewiq-vs-endpoint"
VS_INDEX     = "renewiq.silver.ppa_chunks_vs_index"
SOURCE_TABLE = "renewiq.silver.ppa_contract_chunks"
PRIMARY_KEY  = "chunk_id"
EMBED_COLUMN = "chunk_text"

# Embedding model — Databricks-managed BGE (no external API key needed)
EMBED_MODEL  = "databricks-bge-large-en"

vsc = VectorSearchClient(disable_notice=True)

# Check if endpoint already exists
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]

if VS_ENDPOINT in existing:
    print(f"✓ Endpoint '{VS_ENDPOINT}' already exists")
else:
    print(f"Creating endpoint '{VS_ENDPOINT}' — this takes ~5 min...")
    vsc.create_endpoint(
        name=VS_ENDPOINT,
        endpoint_type="STANDARD",
    )
    print(f"✓ Endpoint '{VS_ENDPOINT}' created")

# Wait until endpoint is online
import time

def wait_for_endpoint(vsc, name, timeout_s=600):
    start = time.time()
    while time.time() - start < timeout_s:
        status = vsc.get_endpoint(name)["endpoint_status"]["state"]
        if status == "ONLINE":
            print(f"✓ Endpoint '{name}' is ONLINE")
            return
        print(f"  … endpoint state: {status} ({int(time.time()-start)}s elapsed)")
        time.sleep(20)
    raise TimeoutError(f"Endpoint '{name}' did not come online within {timeout_s}s")

wait_for_endpoint(vsc, VS_ENDPOINT)

# COMMAND ----------

# MAGIC %md ## 3. Create Delta Sync Index

# COMMAND ----------

# DBTITLE 1,Create or sync the Delta Sync index
endpoint = vsc.get_endpoint(VS_ENDPOINT)
index_names = [i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])]

if VS_INDEX in index_names:
    print(f"✓ Index '{VS_INDEX}' already exists — triggering sync...")
    vsc.get_index(VS_ENDPOINT, VS_INDEX).sync()
    print("✓ Sync triggered")
else:
    print(f"Creating Delta Sync index '{VS_INDEX}'...")
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",          # manual sync; change to CONTINUOUS for real-time
        primary_key=PRIMARY_KEY,
        embedding_source_column=EMBED_COLUMN,
        embedding_model_endpoint_name=EMBED_MODEL,
    )
    print(f"✓ Index '{VS_INDEX}' created")

# COMMAND ----------

# DBTITLE 1,Wait for index to be ready
def wait_for_index(vsc, endpoint_name, index_name, timeout_s=900):
    start = time.time()
    while time.time() - start < timeout_s:
        idx = vsc.get_index(endpoint_name, index_name)
        status = idx.describe().get("status", {})
        state  = status.get("detailed_state", status.get("state", "UNKNOWN"))
        if state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"):
            print(f"✓ Index '{index_name}' is {state}")
            return
        if "FAILED" in state:
            raise RuntimeError(f"Index creation failed: {status}")
        print(f"  … index state: {state} ({int(time.time()-start)}s elapsed)")
        time.sleep(30)
    raise TimeoutError(f"Index '{index_name}' did not become ready within {timeout_s}s")

wait_for_index(vsc, VS_ENDPOINT, VS_INDEX)

# COMMAND ----------

# MAGIC %md ## 4. Smoke Test — Similarity Search

# COMMAND ----------

# DBTITLE 1,Run a sample query against the index
index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

results = index.similarity_search(
    query_text="negative price provisions no price floor curtailment compensation",
    columns=["chunk_id", "contract_id", "clause_id", "section_title", "risk_category", "chunk_text"],
    num_results=5,
    filters={}
)

print("Top results for negative price query:")
print("-" * 60)
for hit in results.get("result", {}).get("data_array", []):
    chunk_id, contract_id, clause_id, section_title, risk_cat, text, score = hit
    print(f"  score={score:.3f} | {contract_id} | {clause_id} | {risk_cat}")
    print(f"    {text[:120]}...")
    print()

# COMMAND ----------

# MAGIC %md ## 5. Save Config to Unity Catalog (for agents to discover at runtime)

# COMMAND ----------

# DBTITLE 1,Write VS config to Gold layer for agent discovery
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS renewiq.gold.vector_search_config (
        config_key   STRING NOT NULL,
        config_value STRING NOT NULL,
        updated_ts   TIMESTAMP
    )
    USING DELTA
    COMMENT 'Vector Search endpoint and index configuration for RenewIQ agents'
""")

config_rows = [
    ("vs_endpoint",   VS_ENDPOINT),
    ("vs_index",      VS_INDEX),
    ("embed_model",   EMBED_MODEL),
    ("primary_key",   PRIMARY_KEY),
    ("embed_column",  EMBED_COLUMN),
    ("source_table",  SOURCE_TABLE),
]

from datetime import datetime
now = datetime.utcnow()
for key, val in config_rows:
    spark.sql(f"""
        MERGE INTO renewiq.gold.vector_search_config AS t
        USING (SELECT '{key}' AS config_key, '{val}' AS config_value, CURRENT_TIMESTAMP() AS updated_ts) AS s
        ON t.config_key = s.config_key
        WHEN MATCHED THEN UPDATE SET t.config_value = s.config_value, t.updated_ts = s.updated_ts
        WHEN NOT MATCHED THEN INSERT *
    """)

print("✓ VS config written to renewiq.gold.vector_search_config")
spark.sql("SELECT * FROM renewiq.gold.vector_search_config").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Resource | Value |
# MAGIC |---|---|
# MAGIC | VS Endpoint | `renewiq-vs-endpoint` |
# MAGIC | VS Index | `renewiq.silver.ppa_chunks_vs_index` |
# MAGIC | Embedding model | `databricks-bge-large-en` |
# MAGIC | Source table | `renewiq.silver.ppa_contract_chunks` |
# MAGIC | Sync type | `TRIGGERED` (run this notebook to refresh) |
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Phase 4: `src/agents/contract_rag/retriever.py` queries this index
# MAGIC - To keep index fresh: schedule this notebook or change pipeline_type to `CONTINUOUS`
