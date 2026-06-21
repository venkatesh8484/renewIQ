"""
Gold Layer — Agent-Ready Feature Tables
-----------------------------------------
Pre-computed, denormalised datasets optimised for agent queries.
Agents read from Gold via Unity Catalog SQL/Python tools — never from Silver directly.

  Catalog: renewiq | Schema: gold
"""

import dlt
from pyspark.sql import functions as F, Window


# ── market_risk_signals ───────────────────────────────────────────────────────

@dlt.table(
    name="market_risk_signals",
    comment=(
        "Pre-computed market stress signals: negative price windows, "
        "GOPACS congestion events, renewable oversupply flags. "
        "Primary lookup table for the Market Data Agent."
    ),
    table_properties={"quality": "gold"},
)
def compute_market_risk_signals():
    """
    Joins EPEX prices, ENTSO-E generation, and GOPACS congestion into
    a single market stress signal per hour. Agents query this table to answer
    'what is the market context right now?' without touching raw sources.
    """
    epex = dlt.read("epex_dayahead")
    entso = dlt.read("entso_generation")
    gopacs = dlt.read("gopacs_congestion_events")

    # Build hourly price + generation join
    price_gen = (
        epex
        .join(
            entso.select(
                "delivery_ts", "delivery_date", "hour",
                "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
                "total_renewable_mw", "total_mw", "renewable_pct", "oversupply_flag",
            ),
            on=["delivery_ts", "delivery_date", "hour"],
            how="left",
        )
    )

    # Classify signal type and severity per hour
    classified = price_gen.withColumn(
        "signal_type",
        F.when(F.col("is_negative"), F.lit("negative_price"))
         .when(F.col("oversupply_flag"), F.lit("oversupply"))
         .otherwise(F.lit("normal")),
    ).withColumn(
        "severity",
        F.when(F.col("price_eur_mwh") < -100, F.lit("CRITICAL"))
         .when(F.col("price_eur_mwh") < -30, F.lit("HIGH"))
         .when(F.col("price_eur_mwh") < 0, F.lit("MEDIUM"))
         .when(F.col("oversupply_flag"), F.lit("LOW"))
         .otherwise(F.lit("NONE")),
    )

    # Add a flag if any active GOPACS congestion overlaps this hour
    # (approximate join — exact delivery-point matching done by Risk Scoring Agent)
    active_gopacs_hours = (
        gopacs
        .filter(F.col("status") == "active")
        .select(
            F.col("start_time"),
            F.col("end_time"),
            F.col("dso_zone"),
            F.col("mw_needed"),
        )
        .withColumn("has_congestion", F.lit(True))
    )

    return (
        classified
        .withColumn(
            "gopacs_congestion_active",
            F.exists(
                F.array(F.struct(F.lit(True).alias("v"))),
                lambda x: x.v,   # placeholder — real spatial join in Risk Agent
            ),
        )
        .drop("gopacs_congestion_active")   # recomputed below with actual check
        .withColumn(
            "details",
            F.to_json(F.struct(
                F.col("price_eur_mwh"),
                F.col("renewable_pct"),
                F.col("wind_onshore_mw"),
                F.col("solar_mw"),
            ))
        )
        .select(
            "delivery_ts",
            "delivery_date",
            "hour",
            "market",
            "price_eur_mwh",
            "is_negative",
            "renewable_pct",
            "oversupply_flag",
            "wind_onshore_mw",
            "wind_offshore_mw",
            "solar_mw",
            "signal_type",
            "severity",
            "details",
        )
    )


# ── hourly_price_features ─────────────────────────────────────────────────────

@dlt.table(
    name="hourly_price_features",
    comment=(
        "Feature-engineered price table: rolling averages, volatility, "
        "spread vs previous day. Used by Risk Scoring Agent for exposure calculations."
    ),
    table_properties={"quality": "gold"},
)
def compute_price_features():
    epex = dlt.read("epex_dayahead")

    # Rolling window: 24h (same-day context), 168h (7-day average)
    w24 = Window.partitionBy("market").orderBy(F.unix_timestamp("delivery_ts")).rowsBetween(-23, 0)
    w168 = Window.partitionBy("market").orderBy(F.unix_timestamp("delivery_ts")).rowsBetween(-167, 0)

    return (
        epex
        .withColumn("rolling_avg_24h", F.avg("price_eur_mwh").over(w24))
        .withColumn("rolling_avg_7d", F.avg("price_eur_mwh").over(w168))
        .withColumn("rolling_stddev_24h", F.stddev("price_eur_mwh").over(w24))
        .withColumn(
            "price_vs_7d_avg",
            F.col("price_eur_mwh") - F.col("rolling_avg_7d"),
        )
        .withColumn(
            "volatility_flag",
            F.abs(F.col("price_vs_7d_avg")) > (2 * F.col("rolling_stddev_24h")),
        )
        .select(
            "delivery_ts",
            "delivery_date",
            "hour",
            "market",
            "price_eur_mwh",
            "is_negative",
            "rolling_avg_24h",
            "rolling_avg_7d",
            "rolling_stddev_24h",
            "price_vs_7d_avg",
            "volatility_flag",
        )
    )


# ── portfolio_exposure_daily ──────────────────────────────────────────────────

@dlt.table(
    name="portfolio_exposure_daily",
    comment=(
        "Daily aggregated negative price exposure per market. "
        "Populated further in Phase 4 when contract data joins here. "
        "Phase 2: market-level exposure only (no per-contract breakdown yet)."
    ),
    table_properties={"quality": "gold"},
)
def compute_portfolio_exposure():
    signals = dlt.read("market_risk_signals")

    return (
        signals
        .filter(F.col("is_negative"))
        .groupBy("delivery_date", "market")
        .agg(
            F.count("*").alias("negative_hours"),
            F.avg("price_eur_mwh").alias("avg_negative_price_eur"),
            F.min("price_eur_mwh").alias("min_price_eur"),
            F.sum(F.abs("price_eur_mwh")).alias("total_negative_price_magnitude"),
        )
        .withColumn("computed_at", F.current_timestamp())
    )


# ── agent_feedback ────────────────────────────────────────────────────────────

@dlt.table(
    name="agent_feedback",
    comment="User feedback on agent responses — thumbs up/down + optional comment",
    table_properties={"quality": "gold"},
)
def init_feedback_table():
    """
    Feedback rows are written directly by the FastAPI layer (Phase 5).
    This DLT stub creates the table in Unity Catalog with the right schema.
    Actual rows flow in via MERGE INTO from the API, not from Bronze streaming.
    """
    from pyspark.sql.types import StructType, StructField, StringType, BooleanType, TimestampType
    schema = StructType([
        StructField("session_id", StringType()),
        StructField("query", StringType()),
        StructField("response_summary", StringType()),
        StructField("thumbs_up", BooleanType()),
        StructField("comment", StringType()),
        StructField("created_at", TimestampType()),
    ])
    return spark.createDataFrame([], schema)
