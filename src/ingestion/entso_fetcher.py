"""
ENTSO-E Generation Mix Fetcher
--------------------------------
Source: ENTSO-E Transparency Platform
        Document type A75 — Actual Generation Per Production Type
        Country: NL (10YNL----------L)

Used to detect renewable oversupply — a leading indicator for negative prices
and curtailment risk on PPA contracts.

Returns:
    DataFrame with hourly generation by fuel type + derived features:
    - renewable_pct: (wind + solar) / total_load
    - oversupply_flag: renewable_pct > 0.70
"""

import logging
import os
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

NL_AREA_CODE = "10YNL----------L"

# ENTSO-E fuel type codes → human-readable names
FUEL_TYPE_MAP = {
    "B01": "biomass",
    "B09": "geothermal",
    "B10": "hydro_pump_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_reservoir",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B20": "other_renewable",
    "B02": "lignite",
    "B03": "fossil_gas",
    "B04": "fossil_hard_coal",
    "B05": "fossil_oil",
    "B14": "nuclear",
    "B06": "other",
}

RENEWABLE_TYPES = {"solar", "wind_offshore", "wind_onshore", "biomass", "hydro_run_of_river"}


class ENTSOFetcher:
    """
    Fetches actual generation per production type from ENTSO-E Transparency.

    Schema returned:
        delivery_ts       TIMESTAMP (UTC)
        delivery_date     DATE
        hour              INT
        wind_onshore_mw   FLOAT
        wind_offshore_mw  FLOAT
        solar_mw          FLOAT
        total_renewable_mw FLOAT
        total_mw          FLOAT
        renewable_pct     FLOAT      (0.0–1.0)
        oversupply_flag   BOOL       (renewable_pct > 0.70)
        country           STRING
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSO_E_API_KEY", "")
        if not self.api_key:
            logger.warning("ENTSO_E_API_KEY not set — ENTSO-E fetcher will run in mock mode.")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from entsoe import EntsoePandasClient
                self._client = EntsoePandasClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("Run: pip install entsoe-py")
        return self._client

    def fetch_generation(
        self,
        start: date,
        end: Optional[date] = None,
        country: str = "NL",
    ) -> pd.DataFrame:
        """
        Fetch hourly generation mix for a date range.

        Args:
            start:   First date (inclusive)
            end:     Last date (inclusive). Defaults to start.
            country: Country code — currently only "NL" supported.
        """
        if end is None:
            end = start

        if not self.api_key:
            return self._mock_data(start, end, country)

        start_ts = pd.Timestamp(start, tz="Europe/Amsterdam")
        end_ts = pd.Timestamp(end, tz="Europe/Amsterdam") + pd.Timedelta(days=1)

        logger.info(f"Fetching ENTSO-E generation mix NL: {start} → {end}")
        try:
            client = self._get_client()
            raw = client.query_generation(
                country_code=NL_AREA_CODE,
                start=start_ts,
                end=end_ts,
                psr_type=None,  # all types
            )
        except Exception as e:
            logger.error(f"ENTSO-E generation API error: {e}. Returning mock data.")
            return self._mock_data(start, end, country)

        # raw is a DataFrame with MultiIndex columns (psr_type, actual/forecast)
        # flatten to one column per fuel type (actual values only)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs("Actual Aggregated", axis=1, level=1, drop_level=True)

        # Rename columns using fuel type map
        raw.columns = [
            FUEL_TYPE_MAP.get(col, col.lower().replace(" ", "_"))
            for col in raw.columns
        ]
        raw = raw.fillna(0.0)
        raw.index = raw.index.tz_convert("UTC")

        df = raw.reset_index()
        df.columns.values[0] = "delivery_ts"
        df["delivery_date"] = df["delivery_ts"].dt.date
        df["hour"] = df["delivery_ts"].dt.hour
        df["country"] = country

        # Derived features
        wind_cols = [c for c in df.columns if "wind" in c]
        df["wind_onshore_mw"] = df.get("wind_onshore", pd.Series(0, index=df.index))
        df["wind_offshore_mw"] = df.get("wind_offshore", pd.Series(0, index=df.index))
        df["solar_mw"] = df.get("solar", pd.Series(0, index=df.index))
        df["total_renewable_mw"] = df[[c for c in df.columns if c in RENEWABLE_TYPES]].sum(axis=1)
        df["total_mw"] = df[[c for c in df.columns if c in set(FUEL_TYPE_MAP.values())]].sum(axis=1)
        df["renewable_pct"] = (df["total_renewable_mw"] / df["total_mw"].replace(0, float("nan"))).fillna(0.0)
        df["oversupply_flag"] = df["renewable_pct"] > 0.70

        cols = [
            "delivery_ts", "delivery_date", "hour",
            "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
            "total_renewable_mw", "total_mw",
            "renewable_pct", "oversupply_flag", "country",
        ]
        logger.info(
            f"Fetched {len(df)} generation rows. "
            f"Oversupply hours: {df['oversupply_flag'].sum()}"
        )
        return df[cols]

    def _mock_data(self, start: date, end: date, country: str) -> pd.DataFrame:
        """Mock generation data for local dev."""
        import numpy as np

        dates = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        rows = []
        rng = np.random.default_rng(seed=7)

        for d in dates:
            for hour in range(24):
                ts = d + pd.Timedelta(hours=hour)
                # Solar peaks midday, wind is variable
                solar = max(0, 3000 * np.sin((hour - 6) * np.pi / 12) + rng.normal(0, 200))
                wind_on = max(0, 2500 + rng.normal(0, 600))
                wind_off = max(0, 1800 + rng.normal(0, 400))
                total_renewable = solar + wind_on + wind_off
                total = total_renewable + rng.uniform(8000, 14000)
                renewable_pct = total_renewable / total

                rows.append({
                    "delivery_ts": ts,
                    "delivery_date": d.date(),
                    "hour": hour,
                    "wind_onshore_mw": round(float(wind_on), 1),
                    "wind_offshore_mw": round(float(wind_off), 1),
                    "solar_mw": round(float(solar), 1),
                    "total_renewable_mw": round(float(total_renewable), 1),
                    "total_mw": round(float(total), 1),
                    "renewable_pct": round(float(renewable_pct), 4),
                    "oversupply_flag": renewable_pct > 0.70,
                    "country": country,
                })

        return pd.DataFrame(rows)
