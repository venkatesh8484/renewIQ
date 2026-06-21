# ADR-002: LangGraph StateGraph for Multi-Agent Orchestration

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

RenewIQ's core analytical pipeline requires coordinating four specialized agents to produce a PPA risk report:

1. **market_data agent** — fetches EPEX spot prices, ENTSO-E forecasts, and GOPACS congestion events from Gold Delta tables; computes market context statistics.
2. **contract_rag agent** — performs semantic search over the uploaded PPA clause chunks using Databricks Vector Search + FlashRank reranker; retrieves the most relevant contract provisions for the risk query.
3. **risk_scoring agent** — applies deterministic Python logic to calculate EUR exposure metrics; generates structured `RiskFlag` objects with severity levels.
4. **report_writer agent** — synthesizes market context, retrieved clauses, and risk flags into a structured Markdown risk report using an LLM.

A key architectural requirement is that `market_data` and `contract_rag` can execute **in parallel** since they are independent — neither depends on the other's output. Only `risk_scoring` requires both their outputs, and `report_writer` requires `risk_scoring`'s output.

Three orchestration approaches were evaluated:

- **LangChain AgentExecutor**: ReAct-style single-agent loop where tools are called sequentially based on LLM reasoning. No native parallelism; routing is non-deterministic (LLM decides tool call order); hard to unit-test individual tool invocations in isolation.
- **Custom async orchestrator**: Hand-written `asyncio` code managing agent coroutines with explicit `asyncio.gather()` for parallel execution. Maximum control but high maintenance burden; no built-in state management, tracing, or retry logic.
- **LangGraph StateGraph**: Graph-based framework where nodes are Python functions (agents) and edges define data flow. Supports parallel fan-out via branching edges to multiple nodes. State is a `TypedDict` that flows through the graph, providing type safety and introspectability.

## Decision

Use **LangGraph StateGraph** as the orchestration layer for all four agents.

The graph topology is:
```
START → [market_data, contract_rag] (parallel fan-out)
        ↓              ↓
        └──── risk_scoring (waits for both) ────→ report_writer → END
```

The shared `AgentState` TypedDict carries all inter-agent data: contract text, market statistics, risk flags, and the final report. Each agent node is a pure Python function with signature `(state: AgentState) -> dict` — returning only the keys it modifies, which LangGraph merges into the state via reducer functions.

Node-level unit tests mock the state dictionary and assert on the returned delta, enabling fast, isolated testing without running the full graph. MLflow autologging captures the full graph execution trace (node entry/exit timestamps, token counts per LLM call, retrieved clause IDs) as a nested run under the parent analysis run.

Conditional edges from `risk_scoring` allow short-circuiting to a minimal report if critical data is missing (e.g., EPEX data unavailable), rather than failing the entire pipeline.

## Consequences

**Positive:**
- Parallel execution of `market_data` and `contract_rag` reduces total wall-clock time by ~40% compared to sequential execution (empirically measured at ~3.2s vs ~5.4s for typical PPA analyses).
- TypedDict state provides compile-time type checking via mypy, catching inter-agent contract mismatches before runtime.
- Each node is independently unit-testable, enabling a comprehensive test suite without expensive end-to-end LLM calls.
- LangGraph's native MLflow integration logs the full execution DAG as structured traces, enabling per-node latency analysis and token cost attribution.
- Deterministic routing (Python conditional functions, not LLM decisions) makes the system behavior reproducible and auditable.

**Negative:**
- LangGraph has a steeper learning curve than simple sequential LangChain pipelines; new contributors must understand graph execution semantics and reducer logic.
- LangGraph is a relatively young library; minor version upgrades have occasionally introduced breaking changes in state reducer behavior.
- Debugging parallel branches requires understanding LangGraph's internal checkpoint mechanism, which adds complexity when tracing race conditions.
