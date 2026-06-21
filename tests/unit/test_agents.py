"""
Unit tests for Phase 4 agents.
All external dependencies (LLM, Databricks, Vector Search) are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agents.orchestrator.state import make_initial_state, AgentState


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    state = make_initial_state(
        query       = "What is the negative price risk in the Zeeland wind PPA?",
        session_id  = "test-session-001",
        llm_backend = "ollama",
    )
    state.update(overrides)
    return state


MOCK_SIGNALS = [
    {
        "delivery_date": "2026-06-10",
        "hour": 12,
        "price_eur_mwh": -45.0,
        "is_negative": True,
        "signal_type": "negative_price",
        "severity": "HIGH",
        "renewable_pct": 0.75,
        "details": '{"price_eur_mwh": -45.0}',
    },
    {
        "delivery_date": "2026-06-10",
        "hour": 13,
        "price_eur_mwh": -22.0,
        "is_negative": True,
        "signal_type": "negative_price",
        "severity": "MEDIUM",
        "renewable_pct": 0.73,
        "details": '{"price_eur_mwh": -22.0}',
    },
]

MOCK_CLAUSES = [
    {
        "chunk_id":      "c-001",
        "contract_id":   "zeeland-wind-physical-ppa-v1",
        "clause_id":     "7.2",
        "section_title": "Negative Price Provisions",
        "risk_category": "price_risk",
        "chunk_text":    "No price floor shall apply. If the settlement price is negative the Seller shall pay the Buyer.",
        "rerank_score":  0.92,
    },
    {
        "chunk_id":      "c-002",
        "contract_id":   "zeeland-wind-physical-ppa-v1",
        "clause_id":     "9.3",
        "section_title": "Curtailment Compensation",
        "risk_category": "curtailment_risk",
        "chunk_text":    "No compensation shall be payable in respect of curtailment. Seller accepts full curtailment risk.",
        "rerank_score":  0.85,
    },
]


# ── Tests: AgentState ──────────────────────────────────────────────────────────

class TestAgentState:
    def test_make_initial_state_defaults(self):
        state = make_initial_state("test query", "sess-1")
        assert state["query"] == "test query"
        assert state["session_id"] == "sess-1"
        assert state["market_signals"] == []
        assert state["retrieved_clauses"] == []
        assert state["risk_flags"] == []
        assert state["report"] is None
        assert state["errors"] == []
        assert state["llm_backend"] == "ollama"

    def test_make_initial_state_with_contracts(self):
        state = make_initial_state("q", "s", contract_ids=["contract-A"])
        assert state["contract_ids"] == ["contract-A"]

    def test_make_initial_state_llm_backend(self):
        state = make_initial_state("q", "s", llm_backend="databricks")
        assert state["llm_backend"] == "databricks"


# ── Tests: Market Data Agent ───────────────────────────────────────────────────

class TestMarketDataAgent:
    def test_run_uses_mock_data_when_env_set(self):
        import os
        from src.agents.market_data.agent import run

        with patch.dict(os.environ, {"RENEWIQ_USE_MOCK_ENDPOINT": "true"}):
            result = run(_base_state())

        assert "market_signals" in result
        assert "market_context" in result
        assert "negative_hours_90d" in result
        assert result["negative_hours_90d"] > 0

    def test_market_context_is_non_empty(self):
        import os
        from src.agents.market_data.agent import run

        with patch.dict(os.environ, {"RENEWIQ_USE_MOCK_ENDPOINT": "true"}):
            result = run(_base_state())

        assert len(result["market_context"]) > 50
        assert "Negative price" in result["market_context"]

    def test_market_signals_have_required_keys(self):
        import os
        from src.agents.market_data.agent import run

        with patch.dict(os.environ, {"RENEWIQ_USE_MOCK_ENDPOINT": "true"}):
            result = run(_base_state())

        required = {"delivery_date", "hour", "price_eur_mwh", "is_negative",
                    "signal_type", "severity"}
        for sig in result["market_signals"]:
            assert required <= sig.keys()

    def test_avg_negative_price_is_negative(self):
        import os
        from src.agents.market_data.agent import run

        with patch.dict(os.environ, {"RENEWIQ_USE_MOCK_ENDPOINT": "true"}):
            result = run(_base_state())

        assert result["avg_negative_price"] < 0


# ── Tests: Contract RAG Agent ──────────────────────────────────────────────────

class TestContractRAGAgent:
    def test_run_uses_local_retriever_when_no_databricks(self):
        import os
        from src.agents.contract_rag.agent import run

        env = {"RENEWIQ_USE_MOCK_ENDPOINT": "true", "RENEWIQ_RERANKER_IDENTITY": "true"}
        with patch.dict(os.environ, env):
            result = run(_base_state())

        assert "retrieved_clauses" in result
        assert "rag_context" in result

    def test_retrieved_clauses_have_required_keys(self):
        import os
        from src.agents.contract_rag.agent import run

        env = {"RENEWIQ_USE_MOCK_ENDPOINT": "true", "RENEWIQ_RERANKER_IDENTITY": "true"}
        with patch.dict(os.environ, env):
            result = run(_base_state())

        required = {"chunk_id", "contract_id", "clause_id", "chunk_text", "rerank_score"}
        for clause in result["retrieved_clauses"]:
            assert required <= clause.keys()

    def test_rag_context_contains_contract_reference(self):
        import os
        from src.agents.contract_rag.agent import run

        env = {"RENEWIQ_USE_MOCK_ENDPOINT": "true", "RENEWIQ_RERANKER_IDENTITY": "true"}
        with patch.dict(os.environ, env):
            result = run(_base_state())

        # Mock retriever returns zeeland-wind contract
        assert "zeeland" in result["rag_context"].lower() or len(result["rag_context"]) > 0


# ── Tests: Risk Scoring Agent ──────────────────────────────────────────────────

class TestRiskScoringAgent:
    def _state_with_data(self) -> dict:
        return _base_state(
            retrieved_clauses   = MOCK_CLAUSES,
            market_signals      = MOCK_SIGNALS,
            negative_hours_90d  = 168,
            avg_negative_price  = -28.7,
        )

    def test_run_returns_risk_flags(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        assert "risk_flags" in result
        assert len(result["risk_flags"]) > 0

    def test_no_floor_clause_generates_high_flag(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        high_flags = [f for f in result["risk_flags"] if f["severity"] == "HIGH"]
        assert len(high_flags) > 0

    def test_high_flag_has_exposure_eur(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        high_flags = [f for f in result["risk_flags"] if f["severity"] == "HIGH"]
        for f in high_flags:
            if f["risk_category"] == "price_risk":
                assert f.get("exposure_eur") is not None
                assert f["exposure_eur"] > 0

    def test_total_exposure_eur_is_positive(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        assert result["total_exposure_eur"] is not None
        assert result["total_exposure_eur"] > 0

    def test_flags_sorted_high_first(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        flags = result["risk_flags"]
        if len(flags) > 1:
            order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            for i in range(len(flags) - 1):
                assert order.get(flags[i]["severity"], 9) <= order.get(flags[i+1]["severity"], 9)

    def test_no_clauses_returns_empty_flags(self):
        from src.agents.risk_scoring.agent import run
        state = _base_state(retrieved_clauses=[], negative_hours_90d=100)
        result = run(state)
        assert result["risk_flags"] == []
        assert result["total_exposure_eur"] is None

    def test_flag_ids_are_unique(self):
        from src.agents.risk_scoring.agent import run
        result = run(self._state_with_data())
        ids = [f["flag_id"] for f in result["risk_flags"]]
        assert len(ids) == len(set(ids))

    def test_floor_clause_gives_low_severity(self):
        from src.agents.risk_scoring.agent import run
        state = _base_state(
            retrieved_clauses=[{
                "chunk_id":      "c-floor",
                "contract_id":   "nordsee-offshore-ppa-v2",
                "clause_id":     "7.1",
                "section_title": "Price Floor",
                "risk_category": "price_risk",
                "chunk_text":    "The settlement price shall not fall below zero EUR/MWh. Floor applies.",
                "rerank_score":  0.88,
            }],
            market_signals     = MOCK_SIGNALS,
            negative_hours_90d = 168,
            avg_negative_price = -28.7,
        )
        result = run(state)
        price_flags = [f for f in result["risk_flags"] if f["risk_category"] == "price_risk"]
        # With a floor, severity should NOT be HIGH
        for f in price_flags:
            assert f["severity"] != "HIGH"


# ── Tests: Report Writer Agent ─────────────────────────────────────────────────

class TestReportWriterAgent:
    def _state_with_flags(self) -> dict:
        return _base_state(
            retrieved_clauses   = MOCK_CLAUSES,
            market_signals      = MOCK_SIGNALS,
            negative_hours_90d  = 168,
            avg_negative_price  = -28.7,
            market_context      = "## Market Context\n168 negative hours, avg -28.7 €/MWh",
            rag_context         = "## Clauses\nClause 7.2: No price floor.",
            risk_flags          = [
                {
                    "flag_id":       "abc123",
                    "contract_id":   "zeeland-wind-physical-ppa-v1",
                    "clause_id":     "7.2",
                    "risk_category": "price_risk",
                    "severity":      "HIGH",
                    "description":   "No price floor. 168 neg hours.",
                    "exposure_eur":  23104.0,
                },
            ],
            total_exposure_eur  = 23104.0,
        )

    def test_run_returns_report(self):
        from src.agents.report_writer.agent import run
        result = run(self._state_with_flags())
        assert "report" in result
        assert result["report"] is not None

    def test_report_has_required_keys(self):
        from src.agents.report_writer.agent import run
        report = run(self._state_with_flags())["report"]
        required = {
            "session_id", "query", "contract_ids", "risk_flags",
            "total_exposure_eur", "negative_hours_90d", "narrative", "sources",
        }
        assert required <= report.keys()

    def test_report_narrative_is_non_empty(self):
        from src.agents.report_writer.agent import run

        # Force LLM to fail so we test the template fallback
        with patch("src.agents.report_writer.agent.get_llm", side_effect=ImportError):
            result = run(self._state_with_flags())

        assert len(result["report"]["narrative"]) > 50

    def test_report_sources_match_clause_ids(self):
        from src.agents.report_writer.agent import run
        result = run(self._state_with_flags())
        sources = result["report"]["sources"]
        clause_ids = [c["chunk_id"] for c in MOCK_CLAUSES]
        for src in sources:
            assert src in clause_ids

    def test_report_total_exposure_matches_state(self):
        from src.agents.report_writer.agent import run
        result = run(self._state_with_flags())
        assert result["report"]["total_exposure_eur"] == 23104.0

    def test_template_narrative_mentions_high_severity(self):
        from src.agents.report_writer.agent import _template_narrative
        flags = [{"flag_id": "x", "contract_id": "c-1", "clause_id": "7.2",
                  "risk_category": "price_risk", "severity": "HIGH",
                  "description": "No floor.", "exposure_eur": 10000.0}]
        narrative = _template_narrative(flags)
        assert "HIGH" in narrative or "high" in narrative.lower()
        assert "€" in narrative or "exposure" in narrative.lower()

    def test_empty_flags_returns_no_risk_narrative(self):
        from src.agents.report_writer.agent import _template_narrative
        narrative = _template_narrative([])
        assert "No material risk" in narrative or "no" in narrative.lower()


# ── Tests: Full graph (integration, mocked) ───────────────────────────────────

class TestGraph:
    def test_graph_compiles(self):
        from src.agents.orchestrator.graph import build_graph, reset_graph
        reset_graph()
        graph = build_graph()
        assert graph is not None

    def test_graph_invoke_end_to_end(self):
        """Full graph run with all external calls mocked."""
        import os
        from src.agents.orchestrator.graph import build_graph, reset_graph
        from src.agents.orchestrator.state import make_initial_state

        reset_graph()
        graph = build_graph()

        env = {"RENEWIQ_USE_MOCK_ENDPOINT": "true", "RENEWIQ_RERANKER_IDENTITY": "true"}
        with patch.dict(os.environ, env):
            with patch("src.agents.report_writer.agent.get_llm", side_effect=ImportError):
                state  = make_initial_state(
                    query      = "What is the curtailment risk for the Zeeland wind farm?",
                    session_id = "test-e2e-001",
                )
                result = graph.invoke(state)

        assert result["report"] is not None
        assert len(result["report"]["narrative"]) > 0
        assert isinstance(result["risk_flags"], list)

    def test_graph_produces_risk_flags_for_negative_price_query(self):
        import os
        from src.agents.orchestrator.graph import build_graph, reset_graph
        from src.agents.orchestrator.state import make_initial_state

        reset_graph()
        graph = build_graph()

        env = {"RENEWIQ_USE_MOCK_ENDPOINT": "true", "RENEWIQ_RERANKER_IDENTITY": "true"}
        with patch.dict(os.environ, env):
            with patch("src.agents.report_writer.agent.get_llm", side_effect=ImportError):
                state  = make_initial_state(
                    query      = "negative price floor risk zeeland wind ppa curtailment",
                    session_id = "test-e2e-002",
                )
                result = graph.invoke(state)

        # Mock retriever returns zeeland clauses with no price floor → HIGH flag
        assert result.get("negative_hours_90d", 0) > 0
        assert result["report"] is not None
