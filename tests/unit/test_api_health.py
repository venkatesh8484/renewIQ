"""
Unit tests for the FastAPI gateway.
/chat tests mock _invoke_local_graph — no external dependencies required.
End-to-end agent behaviour is covered in tests/unit/test_agents.py.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_MOCK_CHAT_RESPONSE = {
    "response": "HIGH negative price risk. Recommend hedge.",
    "risk_flags": [
        {
            "risk_category": "price_risk",
            "severity": "HIGH",
            "financial_exposure_eur": 23104.0,
            "contract_clause": "7.2",
            "market_trigger": "168 negative price hours in last 90 days",
            "recommendation": "Add price floor amendment",
        }
    ],
    "contracts_in_scope": ["zeeland-wind-physical-ppa-v1"],
    "session_id": "test-session-001",
    "report_id": "test-session-001",
}


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "llm_backend" in data
        assert "mock_mode" in data

    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200


class TestChatRoute:
    def _post_chat(self, msg="test query", contracts=None):
        body = {"message": msg}
        if contracts:
            body["contracts"] = contracts
        with patch("src.api.routes.chat._invoke_local_graph", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = MagicMock(**_MOCK_CHAT_RESPONSE)
            return client.post("/chat", json=body)

    def test_chat_stub_returns_200(self):
        assert self._post_chat().status_code == 200

    def test_chat_response_shape(self):
        resp = self._post_chat("What is our exposure?", ["zeeland-ppa-v1"])
        data = resp.json()
        assert "response" in data
        assert "risk_flags" in data
        assert "contracts_in_scope" in data
        assert isinstance(data["risk_flags"], list)

    def test_chat_empty_message_still_responds(self):
        assert self._post_chat("").status_code == 200

    def test_chat_response_contains_risk_flags(self):
        data = self._post_chat("negative price risk zeeland").json()
        assert len(data["risk_flags"]) > 0
        assert data["risk_flags"][0]["severity"] == "HIGH"


class TestContractsRoute:
    def test_list_contracts_returns_empty_list(self):
        resp = client.get("/contracts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_upload_non_pdf_returns_400(self):
        resp = client.post(
            "/contracts/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert resp.status_code == 400

    def test_get_nonexistent_contract_returns_404(self):
        resp = client.get("/contracts/does-not-exist")
        assert resp.status_code == 404


class TestReportsRoute:
    def test_list_reports_returns_empty_list(self):
        resp = client.get("/reports")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_report_returns_404(self):
        resp = client.get("/reports/does-not-exist")
        assert resp.status_code == 404

    def test_download_nonexistent_report_pdf_returns_404(self):
        resp = client.get("/reports/does-not-exist/pdf")
        assert resp.status_code == 404
