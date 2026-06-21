# RenewIQ Data Dictionary

**Version:** 1.0.0
**Date:** 2026-06-21
**Maintainer:** Venkatesan Mariappan

---

## Table of Contents

1. [Domain Concepts](#1-domain-concepts)
2. [Source Data Systems](#2-source-data-systems)
3. [Bronze Layer Tables](#3-bronze-layer-tables)
4. [Silver Layer Tables](#4-silver-layer-tables)
5. [Gold Layer Tables](#5-gold-layer-tables)
6. [LangGraph Agent Types](#6-langgraph-agent-types)
7. [Risk Classification](#7-risk-classification)

---

## 1. Domain Concepts

### 1.1 PPA Types

| Term | Definition |
|------|-----------|
| **PPA (Power Purchase Agreement)** | A long-term bilateral contract between an electricity generator (seller) and an offtaker (buyer) specifying volume, price, delivery period, and risk allocation for renewable energy. Typical durations: 10–20 years. |
| **Physical PPA** | The generator physically delivers electricity to the offtaker at a specified delivery point (e.g., a grid connection node). The offtaker takes title to the electrons. Generator bears balancing risk if metered output deviates from nominated schedule. Common in markets with direct access to grid connection (NL, DE, UK). |
| **Virtual PPA (VPPA / Contract for Difference)** | A purely financial instrument with no physical delivery. Generator sells electricity at spot price to the grid; offtaker buys at spot from their own supplier. A financial settlement payment ("contract for difference") passes between parties based on the spread between the agreed strike price and the realized spot price. Eliminates delivery logistics but retains full price risk exposure. |
| **Sleeved PPA** | A physical PPA where a licensed energy supplier ("sleeve provider") intermediates between the generator and corporate offtaker, handling grid balancing, scheduling, and regulatory compliance on behalf of both parties. Adds counterparty credit risk (the sleeve provider) but reduces operational complexity for corporate buyers without supply licenses. |
| **Strike Price** | The fixed price per MWh agreed in the PPA contract. For physical PPAs, this is the price the offtaker pays. For VPPAs, this is the reference price for calculating the CfD settlement. Typically indexed or fixed; may include annual escalation clauses. |
| **Capture Rate** | The ratio of the volume-weighted average price actually received by a renewable generator to the simple average spot price over the same period. Wind and solar generation concentrates during periods of lower demand (and thus lower prices), causing capture rates below 100%. A capture rate of 85% means the generator's MWh are worth only 85% of the average market price. |
| **Balancing Responsible Party (BRP)** | An entity holding a balance responsible agreement with the TSO, responsible for nominating generation/consumption schedules and settling imbalances. In the Netherlands, the BRP is registered with TenneT. |
| **Imbalance Settlement Price (ISP)** | The price at which TenneT settles deviations between a BRP's nominated schedule and actual metered output. Can be highly volatile during grid stress events. Distinct from EPEX spot prices. |

---

### 1.2 Contract Clause Types (used in `silver.ppa_clauses_chunked.clause_type`)

| `clause_type` value | Description |
|--------------------|-------------|
| `price_formula` | Defines how the contract price is calculated, including escalation indices, floor/ceiling prices, and indexation mechanisms. |
| `volume_obligation` | Specifies contracted volume, shape (baseload/profile), delivery schedule, and tolerance bands (e.g., ±10% annual volume flexibility). |
| `curtailment` | Conditions under which the generator may reduce output below schedule without financial penalty. Includes grid curtailment, force majeure, and economic curtailment thresholds. |
| `force_majeure` | Events beyond parties' control (extreme weather, grid failure, regulatory change) that excuse performance obligations. |
| `termination` | Conditions triggering early termination, notice periods, and termination payment calculations (mark-to-market vs. fixed penalty). |
| `change_in_law` | Provisions allocating risk of regulatory changes (feed-in tariff removal, grid code changes, carbon pricing) between parties. |
| `metering` | Measurement point specifications, meter reading procedures, dispute resolution for meter failures. |
| `credit_support` | Collateral arrangements (letters of credit, parent guarantees, cash escrow) and thresholds triggering additional credit support calls. |
| `representations` | Representations and warranties given by each party (corporate authority, absence of litigation, regulatory compliance). |
| `definitions` | Contract-level definitions of capitalized terms. |

---

## 2. Source Data Systems

### 2.1 EPEX SPOT Netherlands (NL)

EPEX SPOT SE operates the Pan-European exchange for short-term power trading. The NL bidding zone corresponds to the Netherlands control area operated by TenneT NL.

| Field | Description |
|-------|-------------|
| **Market area** | `NL` — Netherlands bidding zone |
| **Day-Ahead Market (DAM)** | 24 hourly price products for the following calendar day, cleared via single-price auction at 12:00 CET daily. Prices in EUR/MWh. |
| **Intraday Continuous (IDC)** | Continuous trading for individual hourly and 15-minute products up to 5 minutes before delivery. Multiple trades may clear at different prices within a delivery hour. |
| **Price unit** | EUR/MWh |
| **Volume unit** | MWh per hour (for hourly products) |
| **Negative prices** | Technically possible; occurred 52 hours in NL in 2024 during high wind / low demand periods. Important for curtailment risk analysis. |
| **Data source** | EPEX SPOT Market Data API (authenticated); also available via ENTSO-E Transparency Platform with 24h delay. |
| **Update frequency** | DAM results published daily at ~12:45 CET; IDC prices updated continuously. Bronze ingestion runs hourly for IDC, once daily for final DAM settlement. |

### 2.2 ENTSO-E Transparency Platform

The European Network of Transmission System Operators for Electricity publishes generation, load, and grid data for all EU member states.

| Dataset | Description | Resolution | Update Frequency |
|---------|-------------|-----------|----------------|
| **Actual Generation per Production Type** (B01) | Metered generation in MW broken down by source (wind onshore, wind offshore, solar PV, gas, nuclear, etc.) for the NL bidding zone. | 15-minute | ~1 hour delay |
| **Day-Ahead Generation Forecasts** (B04) | TSO-published day-ahead wind and solar generation forecasts used in market clearing. | Hourly | Published D-1 by 10:00 CET |
| **Load Forecast** (B05) | Total load forecast for the NL bidding zone. | Hourly | Published D-1 |
| **Cross-Border Physical Flows** (B11) | Physical electricity flows on interconnectors (NL-DE, NL-BE, NL-GB). Used to detect congestion-driven price divergence. | Hourly | ~2 hour delay |

### 2.3 GOPACS (Grid Operator Platform for Congestion Solutions)

GOPACS is a collaboration between Dutch DSOs (Liander, Enexis, Stedin, Westland Infra) and TSO TenneT to manage distribution grid congestion through market-based redispatch.

| Field | Description |
|-------|-------------|
| **Congestion event** | A period during which a grid operator identifies that scheduled generation exceeds the hosting capacity of a specific grid node or transformer, requiring curtailment or redispatch. |
| **Event ID** | Unique identifier for each congestion event (format: `GOPACS-{year}-{sequence}`). |
| **Grid operator** | DSO responsible for the congested network element (e.g., `Liander`, `Enexis`). |
| **EAN location code** | 18-digit Energy Address Number identifying the specific grid connection point affected. |
| **Direction** | `CONSUME` (grid needs load reduction) or `PRODUCE` (grid needs generation reduction). |
| **Start / End datetime** | UTC timestamps for the congestion window. Congestion events typically 1–8 hours. |
| **Requested volume (MW)** | Volume of generation reduction or load increase requested from the market. |
| **Settled volume (MW)** | Actual volume dispatched in response to the GOPACS request. |
| **Status** | `ACTIVE`, `RESOLVED`, `CANCELLED` |

---

## 3. Bronze Layer Tables

### `bronze.epex_nl_raw`

| Column | Type | Description |
|--------|------|-------------|
| `_ingested_at` | TIMESTAMP | UTC timestamp when the record was written to Bronze by the ingestion job. |
| `_source_system` | STRING | Always `EPEX_SPOT_API` for this table. |
| `_raw_json` | STRING | Complete raw JSON payload from the EPEX SPOT API, unparsed. |
| `delivery_date` | DATE | Calendar date for which the price applies (extracted from raw JSON for partitioning). |

### `bronze.entso_e_generation_raw`

| Column | Type | Description |
|--------|------|-------------|
| `_ingested_at` | TIMESTAMP | UTC ingestion timestamp. |
| `_document_type` | STRING | ENTSO-E document type code (e.g., `A65` for load, `A71` for actual generation). |
| `_raw_xml` | STRING | Complete raw XML publication from ENTSO-E Transparency Platform. |
| `_period_start` | TIMESTAMP | Start of the data period covered by this document (extracted for partitioning). |

### `bronze.gopacs_congestion_raw`

| Column | Type | Description |
|--------|------|-------------|
| `_ingested_at` | TIMESTAMP | UTC ingestion timestamp. |
| `event_id` | STRING | GOPACS event identifier (extracted for deduplication). |
| `_raw_json` | STRING | Raw GOPACS API response payload. |

---

## 4. Silver Layer Tables

### `silver.epex_prices_hourly`

One row per delivery hour per market product type.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `price_id` | STRING | N | UUID surrogate key. |
| `delivery_start` | TIMESTAMP | N | UTC start of the delivery hour (e.g., `2025-01-15T13:00:00Z` for the 14:00–15:00 CET hour). |
| `delivery_end` | TIMESTAMP | N | UTC end of the delivery hour. Always `delivery_start + 1 hour`. |
| `bidding_zone` | STRING | N | Always `NL` in this deployment. |
| `market_type` | STRING | N | `DAM` (Day-Ahead Market) or `IDC` (Intraday Continuous). |
| `price_eur_mwh` | DOUBLE | N | Cleared price in EUR/MWh. May be negative. |
| `volume_mwh` | DOUBLE | Y | Cleared volume in MWh. NULL for DAM (volume not published by EPEX). |
| `_source_ingested_at` | TIMESTAMP | N | Ingestion timestamp from Bronze. |
| `_validated_at` | TIMESTAMP | N | Timestamp when the DLT Silver pipeline processed this record. |

**Partitioned by:** `delivery_start` (year, month)
**Primary key:** `price_id`
**Unique constraint:** `(delivery_start, bidding_zone, market_type)`

### `silver.ppa_clauses_chunked`

One row per text chunk extracted from a PPA document.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `chunk_id` | STRING | N | UUID surrogate key. Used as the primary key in the Databricks Vector Search index. |
| `contract_id` | STRING | N | Foreign key to `silver.ppa_contracts`. |
| `document_version` | INTEGER | N | Monotonically increasing version number for this contract document. Enables tracking of re-uploaded amended contracts. |
| `clause_type` | STRING | Y | Classified clause category (see Section 1.2). NULL if classification confidence < 0.7. |
| `chunk_text` | STRING | N | Raw text content of this chunk, approximately 400–600 tokens. Overlapping by 100 tokens with adjacent chunks. |
| `chunk_embedding` | ARRAY<FLOAT> | N | 1536-dimensional embedding vector produced by `text-embedding-3-small`. |
| `page_number` | INTEGER | Y | Source PDF page number (1-indexed). NULL for digitally-originated contracts. |
| `chunk_index` | INTEGER | N | Ordinal position of this chunk within the document (0-indexed). |
| `party_a` | STRING | Y | Name of the generator party as extracted by NER. |
| `party_b` | STRING | Y | Name of the offtaker party as extracted by NER. |
| `effective_date` | DATE | Y | Contract effective date as extracted from the document. |
| `_ingested_at` | TIMESTAMP | N | Timestamp when the document was chunked and embedded. |

---

## 5. Gold Layer Tables

### `gold.hourly_price_features`

Pre-computed ML features for each delivery hour in the EPEX NL spot market. Consumed by the `market_data` agent to provide market context to risk scoring.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `feature_id` | STRING | N | UUID surrogate key. |
| `delivery_start` | TIMESTAMP | N | UTC start of the delivery hour. |
| `price_eur_mwh` | DOUBLE | N | DAM clearing price for this hour in EUR/MWh. |
| `rolling_30d_mean` | DOUBLE | Y | Rolling 30-day simple moving average of DAM prices ending at this hour. |
| `rolling_30d_std` | DOUBLE | Y | Rolling 30-day standard deviation of DAM prices. Proxy for near-term price volatility. |
| `rolling_90d_mean` | DOUBLE | Y | Rolling 90-day simple moving average. Used as medium-term baseline for contract-vs-market spread calculation. |
| `rolling_90d_std` | DOUBLE | Y | Rolling 90-day standard deviation. |
| `price_percentile_90d` | DOUBLE | Y | Percentile rank of this hour's price within the preceding 90 days (0.0–1.0). Values near 1.0 indicate unusually high price; values near 0.0 indicate low/negative price events. |
| `hour_of_day` | INTEGER | N | Hour of day in CET (0–23). Used for time-of-day seasonality features. |
| `day_of_week` | INTEGER | N | ISO day of week (1=Monday, 7=Sunday). |
| `is_weekend` | BOOLEAN | N | True if Saturday or Sunday. |
| `solar_generation_mw` | DOUBLE | Y | ENTSO-E actual solar generation in NL bidding zone for this hour. NULL if not yet published. |
| `wind_generation_mw` | DOUBLE | Y | ENTSO-E actual wind (onshore + offshore) generation in NL for this hour. |
| `renewable_share` | DOUBLE | Y | `(solar_generation_mw + wind_generation_mw) / total_load_mw`. Proxy for merit order depression pressure. |
| `has_gopacs_event` | BOOLEAN | N | True if any active GOPACS congestion event overlaps this delivery hour. |
| `gopacs_requested_mw` | DOUBLE | N | Sum of requested curtailment volumes from all GOPACS events overlapping this hour. 0 if none. |
| `_computed_at` | TIMESTAMP | N | Timestamp when this row was computed by the Gold feature pipeline. |

**Partitioned by:** `delivery_start` (year, month)

---

### `gold.market_risk_signals`

One row per calendar day, aggregating market risk indicators for portfolio-level risk monitoring. Consumed by the `market_data` agent.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `signal_date` | DATE | N | Calendar date (CET) for which the signals are computed. |
| `daily_avg_price` | DOUBLE | N | Simple average of all hourly DAM prices for this date in EUR/MWh. |
| `daily_volatility` | DOUBLE | N | Standard deviation of hourly DAM prices within this date. High intra-day volatility indicates balancing risk. |
| `negative_price_hours` | INTEGER | N | Count of hours where DAM price < 0. More than 4 hours in a day is flagged as elevated curtailment risk. |
| `price_spike_hours` | INTEGER | N | Count of hours where DAM price > 200 EUR/MWh. Relevant for virtual PPA settlement. |
| `capture_rate_wind` | DOUBLE | Y | Volume-weighted average price received by wind generation divided by daily average price. Values below 0.75 indicate significant profile risk. |
| `capture_rate_solar` | DOUBLE | Y | Volume-weighted average price received by solar generation divided by daily average price. |
| `gopacs_events_count` | INTEGER | N | Number of distinct GOPACS congestion events active on this date. |
| `gopacs_total_mw` | DOUBLE | N | Sum of requested curtailment volumes across all GOPACS events on this date. |
| `risk_level` | STRING | N | Composite daily risk level: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`. Derived from threshold rules on the above fields. |
| `_computed_at` | TIMESTAMP | N | Pipeline computation timestamp. |

---

### `gold.portfolio_exposure_daily`

One row per (contract, date) combination. Represents the mark-to-market financial exposure of each PPA contract against current market prices. Core input to the `risk_scoring` agent's EUR exposure calculation.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `exposure_id` | STRING | N | UUID surrogate key. |
| `contract_id` | STRING | N | Foreign key identifying the PPA contract. |
| `exposure_date` | DATE | N | Calendar date for which exposure is calculated. |
| `contracted_volume_mwh` | DOUBLE | N | PPA contracted volume scheduled for delivery on this date in MWh. |
| `contract_price_eur_mwh` | DOUBLE | N | PPA strike price applicable on this date in EUR/MWh (after applying any escalation index). |
| `market_price_eur_mwh` | DOUBLE | N | Reference market price on this date (DAM daily average) in EUR/MWh. |
| `spread_eur_mwh` | DOUBLE | N | `contract_price_eur_mwh - market_price_eur_mwh`. Positive = contract above market (offtaker overpaying). Negative = contract below market (generator receiving below market). |
| `daily_exposure_eur` | DOUBLE | N | `spread_eur_mwh × contracted_volume_mwh`. EUR value of the contract's out-of-the-money or in-the-money position on this date. |
| `cumulative_exposure_eur` | DOUBLE | N | Running sum of `daily_exposure_eur` from contract effective date to `exposure_date`. Mark-to-market accumulated value. |
| `ppa_type` | STRING | N | `PHYSICAL`, `VIRTUAL`, or `SLEEVED`. Determines settlement mechanics for exposure calculation. |
| `counterparty_id` | STRING | N | Identifier of the counterparty (offtaker or generator, depending on perspective). Used for credit risk aggregation. |
| `_computed_at` | TIMESTAMP | N | Pipeline computation timestamp. |

---

## 6. LangGraph Agent Types

### `AgentState` (TypedDict)

The shared state dictionary that flows through the LangGraph StateGraph. All agent nodes read from and write to this state.

| Field | Type | Set By | Description |
|-------|------|--------|-------------|
| `contract_id` | `str` | API gateway | Unique identifier of the PPA contract being analyzed. |
| `analysis_query` | `str` | API gateway | Natural language risk analysis request from the user (e.g., "Assess curtailment and price risk for this wind PPA in the NL market"). |
| `ppa_type` | `str` | API gateway | `PHYSICAL`, `VIRTUAL`, or `SLEEVED`. |
| `analysis_date` | `date` | API gateway | Reference date for market data retrieval (typically today). |
| `market_data` | `MarketDataOutput \| None` | `market_data` agent | Structured market statistics and risk indicators fetched from Gold tables. |
| `retrieved_clauses` | `list[ClauseResult] \| None` | `contract_rag` agent | Top-3 reranked clause chunks most relevant to `analysis_query`. |
| `risk_flags` | `list[RiskFlag] \| None` | `risk_scoring` agent | Deterministically calculated risk flags with severity and EUR exposure values. |
| `report_markdown` | `str \| None` | `report_writer` agent | Final synthesized risk report in Markdown format. |
| `errors` | `list[str]` | Any agent | Non-fatal errors encountered during execution. Empty list if no errors. |
| `mlflow_run_id` | `str \| None` | API gateway | Parent MLflow run ID for trace logging. |

### `MarketDataOutput` (TypedDict)

| Field | Type | Description |
|-------|------|-------------|
| `avg_price_30d` | `float` | 30-day rolling average EPEX NL DAM price in EUR/MWh. |
| `price_volatility_30d` | `float` | 30-day rolling standard deviation of DAM prices in EUR/MWh. |
| `negative_price_hours_30d` | `int` | Count of negative-price hours in the past 30 days. |
| `capture_rate_estimate` | `float` | Estimated capture rate for the PPA's generation technology (wind/solar) based on recent actuals. |
| `gopacs_events_30d` | `int` | Number of GOPACS congestion events in the past 30 days. |
| `market_risk_level` | `str` | Composite risk level (`LOW`/`MEDIUM`/`HIGH`/`CRITICAL`) from `gold.market_risk_signals`. |
| `daily_exposure_eur` | `float` | Current daily mark-to-market exposure from `gold.portfolio_exposure_daily`. |
| `data_as_of` | `datetime` | Timestamp of the most recent data point used. Passed to UI for staleness display. |

### `ClauseResult` (TypedDict)

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | UUID of the retrieved clause chunk. |
| `clause_type` | `str \| None` | Classified clause type (see Section 1.2). |
| `chunk_text` | `str` | Text content of the clause chunk. |
| `rerank_score` | `float` | FlashRank relevance score (higher = more relevant). Range approximately 0.0–1.0. |
| `page_number` | `int \| None` | Source page number for citation. |

### `RiskFlag` (TypedDict / Pydantic model)

| Field | Type | Description |
|-------|------|-------------|
| `flag_id` | `str` | UUID for this risk flag instance. |
| `category` | `str` | Risk category code (see Section 7). |
| `severity` | `str` | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. |
| `value_eur` | `float \| None` | Quantified financial exposure in EUR, if calculable. NULL for qualitative flags (e.g., legal/regulatory). |
| `description` | `str` | One-sentence human-readable description of the risk condition (generated by deterministic Python, not LLM). |
| `triggering_formula` | `str` | The Python expression or formula that triggered this flag, for audit trail (e.g., `"spread_eur_mwh=-12.4, volume_mwh=1200, daily_exposure_eur=-14880"`). |
| `data_sources` | `list[str]` | List of Gold table names or external data sources used in calculating this flag. |

---

## 7. Risk Classification

### Risk Categories

| `category` value | Full Name | Description | Primary Data Sources |
|-----------------|-----------|-------------|---------------------|
| `price_risk` | Market Price Risk | Risk that spot market prices diverge materially from the PPA strike price, creating financial exposure for either party. Includes downside risk (prices fall below strike for generator in fixed-price deal) and upside risk (prices rise above strike for offtaker). | `gold.portfolio_exposure_daily`, `gold.hourly_price_features` |
| `volume_risk` | Generation Volume Risk | Risk that actual generation deviates from contracted volume obligations, potentially triggering shortfall penalties or creating excess delivery that must be sold at spot. Includes wind/solar resource variability. | `silver.entso_generation_15min`, PPA volume obligation clauses |
| `curtailment_risk` | Grid Curtailment Risk | Risk that network congestion forces the generator to reduce output below schedule, resulting in lost revenue. Elevated when GOPACS events are frequent or when negative spot prices incentivize voluntary curtailment. | `gold.market_risk_signals.gopacs_events_count`, `gold.hourly_price_features.negative_price_hours` |
| `basis_risk` | Delivery Point Basis Risk | Risk arising from price differences between the contract delivery point and the reference market hub. Relevant for PPAs where the settlement price is indexed to a different market than where the generator physically connects. | `gold.hourly_price_features`, cross-border flow data |
| `counterparty_risk` | Counterparty Credit Risk | Risk that the counterparty (offtaker or generator) defaults on payment or delivery obligations before contract expiry. Assessed from the mark-to-market value of the PPA and counterparty credit information. | `gold.portfolio_exposure_daily.cumulative_exposure_eur`, credit support clauses |
| `legal_regulatory` | Legal and Regulatory Risk | Risk of adverse regulatory changes (e.g., subsidy scheme removal, grid code revision, environmental permitting changes) that affect the economic value or legal enforceability of the PPA. Qualitative assessment based on change-in-law clause analysis. | `silver.ppa_clauses_chunked` (clause_type=change_in_law, force_majeure) |

### Severity Thresholds

| Severity | Criteria |
|----------|---------|
| `LOW` | Financial exposure < EUR 50,000; or qualitative risk with established mitigation in contract |
| `MEDIUM` | Financial exposure EUR 50,000–500,000; or qualitative risk with partial mitigation |
| `HIGH` | Financial exposure EUR 500,000–5,000,000; or significant contractual gap identified |
| `CRITICAL` | Financial exposure > EUR 5,000,000; or risk of contract termination or regulatory breach |

