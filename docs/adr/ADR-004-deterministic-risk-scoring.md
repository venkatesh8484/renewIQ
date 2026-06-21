# ADR-004: Deterministic Python for Financial Risk Calculations

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

A central function of RenewIQ is computing quantitative financial risk metrics for PPA portfolios. These metrics include:

- **EUR price exposure:** `(contract_price - spot_price) × contracted_volume_MWh` for each delivery interval
- **Annual value-at-risk:** exposure distribution across simulated price scenarios (historical simulation method)
- **Curtailment-adjusted revenue loss:** `expected_generation_MWh × capture_rate × (contract_price - floor_price)`
- **Basis risk:** spread between delivery point price and reference hub price, multiplied by contracted volume
- **Counterparty credit exposure:** mark-to-market value of the PPA under current market conditions

Two approaches were evaluated for implementing these calculations:

- **LLM-based financial reasoning**: Allow the LLM (GPT-4o or Claude Sonnet) to perform the arithmetic and financial reasoning end-to-end, receiving raw numbers as context and returning computed risk figures as text. This approach is attractive for its flexibility — LLMs can reason about novel clause structures and non-standard risk formulas embedded in contract text. However, LLMs are demonstrably unreliable at multi-step arithmetic, especially with floating-point numbers, unit conversions (MWh vs. MW × hours), and currency calculations. A miscalculation of 1000 MWh × EUR 45.23/MWh produces a six-figure error with no indication anything went wrong.
- **Deterministic Python calculations**: All financial arithmetic is implemented as pure Python functions in `risk_scoring/agent.py` using standard numeric libraries. The LLM is invoked only to generate the narrative explanation of risk findings, never to perform calculations. Inputs and outputs are validated using Pydantic models.

The regulatory context reinforces this decision: under EU REMIT (Regulation on Energy Market Integrity and Transparency) and MiFID II reporting obligations, financial exposure calculations submitted in regulatory reports must be reproducible and auditable. An LLM's non-deterministic output cannot satisfy this requirement.

## Decision

Implement all financial risk calculations as **deterministic Python functions** in `risk_scoring/agent.py`. The LLM is strictly prohibited from performing arithmetic on financial figures.

The division of responsibility is explicit:

**Python handles:**
- EUR/MWh exposure calculations (vectorized via NumPy/Pandas over hourly price arrays)
- Risk flag generation (`RiskFlag` Pydantic objects with `category`, `severity`, `value_eur`, `description`)
- Threshold comparisons (e.g., `if basis_risk_eur > 50000: severity = "HIGH"`)
- Statistical metrics (P5/P50/P95 price scenarios, volatility σ)

**LLM handles:**
- Natural-language narrative explaining what the risk flags mean in the context of this specific PPA
- Recommendations for risk mitigation (hedging strategies, clause renegotiation)
- Synthesizing market context with contract terms in prose

The `risk_scoring` node validates all inputs against `MarketDataOutput` and `ContractRAGOutput` TypedDict schemas before running calculations, and validates all outputs against `RiskScoringOutput` before passing to `report_writer`. MLflow logs every input/output snapshot as an artifact for full audit traceability.

A dedicated test suite in `tests/test_risk_scoring.py` covers 40+ edge cases: zero-generation periods, negative spread scenarios, missing GOPACS event data, multi-currency contracts (EUR vs. GBP), and weekend/holiday delivery periods.

## Consequences

**Positive:**
- Full auditability: every EUR figure in the output can be traced back to a specific formula, input data point, and timestamp.
- Reproducibility: given the same inputs, the calculation always produces identical outputs — satisfying regulatory and client audit requirements.
- Speed: Python arithmetic on hourly price arrays (8760 rows/year) completes in milliseconds; no LLM latency for the calculation step.
- Testability: pure functions with numeric inputs/outputs are straightforward to unit-test with precise assertions.
- Trust: clients and risk managers can inspect the formulas directly; there is no "black box" producing financial figures.

**Negative:**
- The LLM cannot discover novel risk patterns embedded in non-standard contract clauses that require financial reasoning to quantify (e.g., an unusual indexed-price formula). Such cases must be escalated to human analysts.
- Adding new risk calculation types requires Python development and test coverage, rather than prompt engineering.
- The strict separation requires careful coordination between the risk_scoring developer and the LLM prompt author to ensure narrative descriptions accurately reflect calculated figures.
