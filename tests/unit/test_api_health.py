"""
Unit tests for the FastAPI gateway — no external dependencies required.
These run in CI on every PR.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


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
    def test_chat_stub_returns_200(self):
        resp = client.post("/chat", json={"message": "test query"})
        assert resp.status_code == 200

    def test_chat_response_shape(self):
        resp = client.post(
            "/chat",
            json={"message": "What is our exposure?", "contracts": ["zeeland-ppa-v1"]},
        )
        data = resp.json()
        assert "response" in data
        assert "risk_flags" in data
        assert "contracts_in_scope" in data
        assert isinstance(data["risk_flags"], list)

    def test_chat_empty_message_still_responds(self):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 200


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
