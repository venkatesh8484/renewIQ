# ADR-006: Redis Cache Layer for Market Data and Report Caching

**Status:** Accepted
**Date:** 2026-06-21
**Deciders:** Venkatesan Mariappan

## Context

The RenewIQ FastAPI gateway handles two types of requests with distinct latency and freshness requirements:

1. **Market data queries** (via the `market_data` agent): Fetch current EPEX NL spot prices, ENTSO-E generation forecasts, and GOPACS congestion events from Gold Delta tables via Databricks SQL. These queries take 2–8 seconds due to Databricks SQL warehouse cold-start latency. EPEX prices are published hourly; ENTSO-E forecasts update every 15 minutes; GOPACS events are near-real-time but typically resolved within hours.
2. **Risk report generation**: End-to-end LangGraph pipeline execution takes 15–45 seconds (LLM calls dominate). For a given PPA + market snapshot combination, the report is deterministic — the same inputs always produce the same risk assessment.

Under expected production load (50–200 concurrent portfolio analysts), the same EPEX price query for "NL Day-Ahead prices, past 30 days" will be issued hundreds of times per hour by different users analyzing different PPAs. Without caching, each query hits the Databricks SQL warehouse independently, driving unnecessary DBU costs and increasing latency for all users.

Three caching strategies were evaluated:

- **No cache**: Every request hits Databricks SQL. Simplest to implement. Unacceptable latency (2–8s for market data that changes hourly) and DBU costs scale linearly with user count.
- **CDN cache (Azure Front Door)**: Appropriate for static assets; poorly suited for authenticated API responses that vary by user context and query parameters. Cannot cache POST request bodies (LangGraph execution payloads are POST). No fine-grained TTL control per data type.
- **Redis cache (Azure Cache for Redis)**: In-memory key-value store with per-key TTL. Sub-millisecond read latency. Supports complex cache key construction from query parameters. Integrates cleanly with FastAPI via Starlette middleware or explicit cache decorators. Widely used in production LLM application stacks.

## Decision

Deploy **Azure Cache for Redis (Basic C1 tier, 1GB)** and implement a two-tier caching strategy:

**Tier 1 — Market Data Cache (TTL: 15 minutes)**
Cache key: `market_data:{query_hash}` where `query_hash` is an MD5 of the normalized query parameters (date range, bidding zone, resolution). The 15-minute TTL aligns with ENTSO-E update frequency (the fastest-changing data source). EPEX prices (hourly) and GOPACS events (hours to resolve) tolerate 15-minute staleness without impact on risk assessments.

**Tier 2 — Report Cache (TTL: 1 hour)**
Cache key: `report:{contract_id}:{market_snapshot_date}`. Risk reports are cached for 1 hour. The cache is explicitly invalidated when a new PPA document is uploaded for a contract or when the market snapshot advances to a new day.

A **Starlette middleware** (`CacheMiddleware`) intercepts requests at the gateway layer: on cache hit, the middleware returns the cached response immediately without invoking the LangGraph pipeline. Cache miss proceeds to the pipeline; the response is serialized to JSON and stored in Redis before being returned to the client. The middleware is transparent to the LangGraph agents — they are unaware of caching.

Cache keys are namespaced by environment (`dev:`, `staging:`, `prod:`) to prevent cross-environment pollution on shared Redis instances during development.

Cache warming: A Databricks Workflow job runs at :05 past each hour to pre-populate the most common market data queries (NL Day-Ahead, NL Intraday, last 30 days) immediately after EPEX data lands in the Gold layer.

## Consequences

**Positive:**
- Measured 80% reduction in Databricks SQL warehouse invocations during load testing (50 concurrent users), reducing DBU spend proportionally.
- Market data responses drop from 2–8 seconds to <5ms on cache hit, eliminating the Databricks SQL cold-start penalty for the majority of requests.
- Report cache eliminates redundant LLM calls when the same PPA is re-analyzed within the same market day.
- Starlette middleware approach requires zero changes to LangGraph agent code — caching is a pure infrastructure concern.
- Redis also serves as the session store for FastAPI's rate limiter (100 requests/minute per API key), consolidating two infrastructure needs into one service.

**Negative:**
- Users may see market data up to 15 minutes stale. In volatile markets (price spikes during grid stress events), a 15-minute-old EPEX price could materially affect a risk assessment. The UI displays a "data as of [timestamp]" indicator derived from the cache entry's creation time to make staleness visible.
- Redis adds an operational dependency: if the Azure Cache for Redis instance is unavailable, the gateway falls back to direct Databricks SQL queries (cache-aside pattern with graceful degradation), but latency degrades significantly.
- The Basic C1 tier (1GB) must be monitored for memory pressure as the cached report payloads can be 10–50KB each; eviction of warm market data entries under memory pressure could degrade hit rates unexpectedly.
- Cache invalidation logic for report entries (on document upload or day rollover) introduces complexity and potential for stale reports if invalidation events are missed.
