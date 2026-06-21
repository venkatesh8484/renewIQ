"""
EPEX SPOT NL Day-Ahead Price Fetcher
-------------------------------------
Source: ENTSO-E Transparency Platform (transparency.entsoe.eu)
        Document type A44 — Day Ahead Prices
        Bidding zone: 10YNL----------L (Netherlands)

Requires: ENTSO_E_API_KEY environment variable
Register free at: https://transparency.entsoe.eu/usrm/user/createPublicUser

Library used: entsoe-py (wraps the ENTSO-E REST API cleanly)
  pip install entsoe-py
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Netherlands bidding zone code on ENTSO-E
NL_AREA_CODE = "10YNL----------L"


class EPEXFetcher:
    """
    Fetches EPEX SPOT NL day-ahead hourly prices from ENTSO-E Transparency.

    Returns a DataFrame with schema:
        delivery_date     DATE
        hour              INT       (0–23)
        delivery_ts       TIMESTAMP (UTC)
        price_eur_mwh     FLOAT
        is_negative       BOOL
        market            STRING    always "NL"
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSO_E_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "ENTSO_E_API_KEY not set — EPEX fetcher will run in mock mode. "
                "Register free at https://transparency.entsoe.eu/usrm/user/createPublicUser"
            )
        self._client = None

    def _get_client(self):
        """Lazy-init the entsoe-py client."""
        if self._client is None:
            try:
                from entsoe import EntsoePandasClient
                self._client = EntsoePandasClient(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "entsoe-py not installed. Run: pip install entsoe-py"
                )
        return self._client

    def fetch_day_ahead(
        self,
        start: date,
        end: Optional[date] = None,
        market: str = "NL",
    ) -> pd.DataFrame:
        """
        Fetch hourly day-ahead prices for a date range.

        Args:
            start: First date to fetch (inclusive)
            end:   Last date to fetch (inclusive). Defaults to start (single day).
            market: Market code — currently only "NL" supported.

        Returns:
            DataFrame with columns: delivery_date, hour, delivery_ts,
                                    price_eur_mwh, is_negative, market
        """
        if end is None:
            end = start

        if not self.api_key:
            logger.info("No API key — returning mock EPEX data")
            return self._mock_data(start, end)

        start_ts = pd.Timestamp(start, tz="Europe/Amsterdam")
        end_ts = pd.Timestamp(end, tz="Europe/Amsterdam") + pd.Timedelta(days=1)

        logger.info(f"Fetching EPEX NL day-ahead prices: {start} → {end}")
        try:
            client = self._get_client()
            series = client.query_day_ahead_prices(
                country_code=NL_AREA_CODE,
                start=start_ts,
                end=end_ts,
            )
        except Exception as e:
            logger.error(f"ENTSO-E API error: {e}. Returning mock data.")
            return self._mock_data(start, end)

        df = series.reset_index()
        df.columns = ["delivery_ts", "price_eur_mwh"]
        df["delivery_ts"] = df["delivery_ts"].dt.tz_convert("UTC")
        df["delivery_date"] = df["delivery_ts"].dt.date
        df["hour"] = df["delivery_ts"].dt.hour
        df["is_negative"] = df["price_eur_mwh"] < 0
        df["market"] = market

        df = df[["delivery_date", "hour", "delivery_ts", "price_eur_mwh", "is_negative", "market"]]
        logger.info(
            f"Fetched {len(df)} price rows. "
            f"Negative hours: {df['is_negative'].sum()}"
        )
        return df

    def detect_negative_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        From a price DataFrame, return only negative price windows with summary stats.

        Returns:
            DataFrame with: window_start, window_end, avg_price, min_price, hours_count
        """
        if df.empty or not df["is_negative"].any():
            return pd.DataFrame(
                columns=["window_start", "window_end", "avg_price", "min_price", "hours_count"]
            )

        neg = df[df["is_negative"]].copy()
        neg = neg.sort_values("delivery_ts")

        # Group consecutive negative hours into windows
        neg["gap"] = neg["delivery_ts"].diff().dt.total_seconds().gt(3700).cumsum()
        windows = (
            neg.groupby("gap")
            .agg(
                window_start=("delivery_ts", "min"),
                window_end=("delivery_ts", "max"),
                avg_price=("price_eur_mwh", "mean"),
                min_price=("price_eur_mwh", "min"),
                hours_count=("price_eur_mwh", "count"),
            )
            .reset_index(drop=True)
        )
        logger.info(f"Found {len(windows)} negative price window(s)")
        return windows

    def _mock_data(self, start: date, end: date) -> pd.DataFrame:
        """
        Generate realistic mock EPEX NL data for local dev / testing.
        Injects negative price hours to exercise the risk scoring logic.
        """
        import numpy as np

        dates = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        rows = []
        rng = np.random.default_rng(seed=42)

        for d in dates:
            for hour in range(24):
                ts = d + pd.Timedelta(hours=hour)
                # Typical NL price profile: low midday (solar), higher morning/evening
                base = 65.0 + 20 * np.sin((hour - 14) * np.pi / 12)
                # Inject negative prices 10:00–15:00 with 30% probability
                if 10 <= hour <= 15 and rng.random() < 0.30:
                    price = rng.uniform(-50, -5)
                else:
                    price = base + rng.normal(0, 8)
                rows.append({
                    "delivery_date": d.date(),
                    "hour": hour,
                    "delivery_ts": ts,
                    "price_eur_mwh": round(float(price), 4),
                    "is_negative": price < 0,
                    "market": "NL",
                })

        logger.info(f"Generated {len(rows)} mock EPEX rows ({start} → {end})")
        return pd.DataFrame(rows)
