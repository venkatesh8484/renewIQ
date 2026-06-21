FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY tests/fixtures/ ./fixtures/

# Inline mock server — simulates Databricks Model Serving invocation endpoint
RUN cat > mock_server.py << 'EOF'
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock Databricks Endpoint")

FIXTURES_DIR = Path("/app/fixtures")

class InvocationRequest(BaseModel):
    messages: list[dict]
    contracts: list[str] = []

@app.get("/health")
def health():
    return {"status": "ok", "mode": "mock"}

@app.post("/serving-endpoints/{endpoint_name}/invocations")
def invoke(endpoint_name: str, request: InvocationRequest):
    """Return fixture response based on first message content."""
    user_msg = request.messages[-1].get("content", "").lower()

    # Route to fixture based on keywords
    if "negative" in user_msg or "exposure" in user_msg:
        fixture = "negative_price_response.json"
    elif "curtailment" in user_msg:
        fixture = "curtailment_response.json"
    elif "report" in user_msg:
        fixture = "report_response.json"
    else:
        fixture = "default_response.json"

    fixture_path = FIXTURES_DIR / fixture
    if fixture_path.exists():
        return json.loads(fixture_path.read_text())

    # Fallback if fixture not found yet
    return {
        "response": f"[MOCK] Processed query: {request.messages[-1].get('content')}",
        "risk_flags": [],
        "contracts_in_scope": request.contracts,
    }
EOF

EXPOSE 8001
CMD ["uvicorn", "mock_server:app", "--host", "0.0.0.0", "--port", "8001"]
