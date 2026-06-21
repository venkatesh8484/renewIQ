# Databricks notebook source
# MAGIC %md
# MAGIC # RenewIQ — Phase 5: MLflow Log & Model Serving Deploy
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Runs a representative query through the RenewIQ LangGraph agent
# MAGIC 2. Logs the agent to MLflow using `mlflow.langchain.log_model`
# MAGIC 3. Registers the model to Unity Catalog (`renewiq.models.renewiq_agent`)
# MAGIC 4. Deploys a scale-to-zero Model Serving endpoint

# COMMAND ----------

# MAGIC %pip install -q langchain-community langgraph flashrank mlflow>=2.13

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.langchain

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/renewiq/agent_experiments")

# COMMAND ----------

# MAGIC %md ## 1. Build and smoke-test the graph

# COMMAND ----------

import os
os.environ["RENEWIQ_USE_MOCK_ENDPOINT"] = "true"
os.environ["RENEWIQ_RERANKER_IDENTITY"] = "true"

import sys
sys.path.insert(0, "/Repos/venkatesh8484@gmail.com/renewIQ")

from src.agents.orchestrator.graph import build_graph, reset_graph
from src.agents.orchestrator.state import make_initial_state

reset_graph()
graph = build_graph()

test_state = make_initial_state(
    query="What is the negative price risk in the Zeeland wind PPA?",
    session_id="smoke-test-001",
)
result = graph.invoke(test_state)

print(f"Risk flags: {len(result['risk_flags'])}")
print(f"Narrative length: {len(result['report']['narrative'])} chars")
print(f"Total exposure: €{result['report'].get('total_exposure_eur', 0):,.0f}")
assert result["report"] is not None, "Smoke test failed — no report generated"
print("Smoke test PASSED")

# COMMAND ----------

# MAGIC %md ## 2. Log agent to MLflow

# COMMAND ----------

UC_MODEL_NAME = "renewiq.models.renewiq_agent"

# Wrap the compiled graph so MLflow can log it as a LangChain-compatible model
class RenewIQGraphWrapper:
    """
    Thin wrapper that adapts LangGraph's StateGraph to the MLflow LangChain interface.
    MLflow expects: model.invoke({"messages": [{"role": "user", "content": "..."}]})
    """
    def __init__(self, graph):
        self._graph = graph

    def invoke(self, inputs: dict) -> dict:
        from src.agents.orchestrator.state import make_initial_state
        import uuid

        messages = inputs.get("messages", [])
        query = messages[-1]["content"] if messages else inputs.get("query", "")
        contract_ids = inputs.get("contracts", [])
        session_id = inputs.get("session_id", str(uuid.uuid4()))

        state = make_initial_state(
            query=query,
            session_id=session_id,
            contract_ids=contract_ids or None,
        )
        result = self._graph.invoke(state)
        report = result.get("report", {})

        return {
            "response": report.get("narrative", ""),
            "risk_flags": report.get("risk_flags", []),
            "total_exposure_eur": report.get("total_exposure_eur"),
            "contracts_in_scope": report.get("contract_ids", []),
            "session_id": session_id,
            "report": report,
        }


reset_graph()
wrapped_agent = RenewIQGraphWrapper(build_graph())

with mlflow.start_run(run_name="renewiq_agent_v1") as run:
    mlflow.log_params({
        "agent_type": "langgraph_multi_agent",
        "nodes": "market_data,contract_rag,risk_scoring,report_writer",
        "llm_backend": "ollama_local",
        "reranker": "flashrank_crossencoder",
    })

    # Log smoke test metrics
    mlflow.log_metrics({
        "risk_flags_count": len(result["risk_flags"]),
        "narrative_length": len(result["report"]["narrative"]),
        "total_exposure_eur": result["report"].get("total_exposure_eur") or 0,
    })

    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="renewiq_agent",
        python_model=wrapped_agent,
        registered_model_name=UC_MODEL_NAME,
        pip_requirements=[
            "langchain>=0.2",
            "langchain-community>=0.2",
            "langgraph>=0.2",
            "flashrank>=0.2",
            "mlflow>=2.13",
        ],
    )

    run_id = run.info.run_id
    print(f"MLflow run: {run_id}")
    print(f"Model logged to: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## 3. Register and deploy to Model Serving

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest version just registered
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
model_version = latest.version
print(f"Deploying model version: {model_version}")

# COMMAND ----------

import requests

DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

ENDPOINT_NAME = "renewiq-agent-endpoint"
ENDPOINT_URL = f"https://{DATABRICKS_HOST}/api/2.0/serving-endpoints"

# Check if endpoint already exists
existing = requests.get(
    f"{ENDPOINT_URL}/{ENDPOINT_NAME}",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
).json()

endpoint_exists = "name" in existing

endpoint_config = {
    "name": ENDPOINT_NAME,
    "config": {
        "served_models": [{
            "name": "renewiq-agent-v1",
            "model_name": UC_MODEL_NAME,
            "model_version": str(model_version),
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }],
        "traffic_config": {
            "routes": [{"served_model_name": "renewiq-agent-v1", "traffic_percentage": 100}]
        },
    },
}

if endpoint_exists:
    print(f"Updating existing endpoint: {ENDPOINT_NAME}")
    resp = requests.put(
        f"{ENDPOINT_URL}/{ENDPOINT_NAME}/config",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
        json=endpoint_config["config"],
    )
else:
    print(f"Creating new endpoint: {ENDPOINT_NAME}")
    resp = requests.post(
        ENDPOINT_URL,
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
        json=endpoint_config,
    )

print(f"Status: {resp.status_code}")
assert resp.status_code in (200, 201), f"Endpoint deploy failed: {resp.text}"
print(f"Endpoint '{ENDPOINT_NAME}' deploy initiated successfully")

# COMMAND ----------

# MAGIC %md ## 4. Smoke test the deployed endpoint

# COMMAND ----------

import time

# Wait for endpoint to be ready (scale-to-zero can take ~60s on first call)
print("Waiting for endpoint to be ready...")
for attempt in range(12):
    status_resp = requests.get(
        f"{ENDPOINT_URL}/{ENDPOINT_NAME}",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
    ).json()
    state = status_resp.get("state", {}).get("ready", "NOT_READY")
    print(f"  Attempt {attempt+1}: {state}")
    if state == "READY":
        break
    time.sleep(10)

# Invoke the endpoint
invocation_resp = requests.post(
    f"https://{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
    json={
        "messages": [{"role": "user", "content": "What is the negative price risk in the Zeeland wind PPA?"}],
        "contracts": ["zeeland-wind-physical-ppa-v1"],
    },
)

print(f"Invocation status: {invocation_resp.status_code}")
if invocation_resp.status_code == 200:
    resp_data = invocation_resp.json()
    print(f"Response preview: {str(resp_data.get('response',''))[:200]}")
    print(f"Risk flags: {len(resp_data.get('risk_flags', []))}")
    print("Endpoint smoke test PASSED")
else:
    print(f"Invocation error: {invocation_resp.text}")
