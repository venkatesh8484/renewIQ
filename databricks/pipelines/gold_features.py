"""
Gold Layer — Agent-Ready Feature Tables
-----------------------------------------
Pre-computed, denormalised datasets optimised for agent queries.
Agents read from Gold via Unity Catalog SQL/Python tools — never from Silver directly.

  Catalog: renewiq | Schema: gold  (schema-qualified per table)
"""

import dlt
from pyspark.sql import functions as F, Window


# ── market_risk_signals ───────────────────────────────────────────────────────

@dlt.table(
    name="gold.market_risk_signals",
    comment=(
        "Pre-computed market stress signals: negative price windows, "
        "GOPACS congestion events, renewable oversupply flags. "
        "Primary lookup table for the Market Data Agent."
    ),
    table_properties={"quality": "gold"},
)
def compute_market_risk_signals():
    epex   = dlt.read("silver.epex_dayahead")
    entso  = dlt.read("silver.entso_generation")
    gopacs = dlt.read("silver.gopacs_congestion_events")   # registered for lineage

    _ = gopacs   # Phase 4 Risk Agent does per-contract delivery-point matching

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

    classified = price_gen.withColumn(
        "signal_type",
        F.when(F.col("is_negative"), F.lit("negative_price"))
         .when(F.col("oversupply_flag"), F.lit("oversupply"))
         .otherwise(F.lit("normal")),
    ).withColumn(
        "severity",
        F.when(F.col("price_eur_mwh") < -100, F.lit("CRITICAL"))
         .when(F.col("price_eur_mwh") < -30,  F.lit("HIGH"))
         .when(F.col("price_eur_mwh") < 0,    F.lit("MEDIUM"))
         .when(F.col("oversupply_flag"),       F.lit("LOW"))
         .otherwise(F.lit("NONE")),
    )

    return (
        classified
        # Phase 4: Risk Agent sets this per-contract via MERGE INTO
        .withColumn("gopacs_congestion_active", F.lit(False))
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
            "delivery_ts", "delivery_date", "hour", "market",
            "price_eur_mwh", "is_negative",
            "renewable_pct", "oversupply_flag",
            "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
            "signal_type", "severity",
            "gopacs_congestion_active",
            "details",
        )
    )


# ── hourly_price_features ─────────────────────────────────────────────────────

@dlt.table(
    name="gold.hourly_price_features",
    comment=(
        "Feature-engineered price table: rolling averages, volatility, "
        "spread vs previous day. Used by Risk Scoring Agent for exposure calculations."
    ),
    table_properties={"quality": "gold"},
)
def compute_price_features():
    epex = dlt.read("silver.epex_dayahead")

    w24  = Window.partitionBy("market").orderBy(F.unix_timestamp("delivery_ts")).rowsBetween(-23, 0)
    w168 = Window.partitionBy("market").orderBy(F.unix_timestamp("delivery_ts")).rowsBetween(-167, 0)

    return (
        epex
        .withColumn("rolling_avg_24h",    F.avg("price_eur_mwh").over(w24))
        .withColumn("rolling_avg_7d",     F.avg("price_eur_mwh").over(w168))
        .withColumn("rolling_stddev_24h", F.stddev("price_eur_mwh").over(w24))
        .withColumn("price_vs_7d_avg",    F.col("price_eur_mwh") - F.col("rolling_avg_7d"))
        .withColumn(
            "volatility_flag",
            F.abs(F.col("price_vs_7d_avg")) > (2 * F.col("rolling_stddev_24h")),
        )
        .select(
            "delivery_ts", "delivery_date", "hour", "market",
            "price_eur_mwh", "is_negative",
            "rolling_avg_24h", "rolling_avg_7d", "rolling_stddev_24h",
            "price_vs_7d_avg", "volatility_flag",
        )
    )


# ── portfolio_exposure_daily ──────────────────────────────────────────────────

@dlt.table(
    name="gold.portfolio_exposure_daily",
    comment=(
        "Daily aggregated negative price exposure per market. "
        "Phase 4 adds per-contract breakdown when contract data joins here."
    ),
    table_properties={"quality": "gold"},
)
def compute_portfolio_exposure():
    signals = dlt.read("gold.market_risk_signals")

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
    name="gold.agent_feedback",
    comment="User feedback on agent responses — thumbs up/down + optional comment",
    table_properties={"quality": "gold"},
)
def init_feedback_table():
    from pyspark.sql.types import StructType, StructField, StringType, BooleanType, TimestampType
    schema = StructType([
        StructField("session_id",        StringType()),
        StructField("query",             StringType()),
        StructField("response_summary",  StringType()),
        StructField("thumbs_up",         BooleanType()),
        StructField("comment",           StringType()),
        StructField("created_at",        TimestampType()),
    ])
    return spark.createDataFrame([], schema)
