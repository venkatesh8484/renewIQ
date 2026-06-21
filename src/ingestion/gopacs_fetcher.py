"""
GOPACS Congestion Fetcher
--------------------------
Source: GOPACS (Grid Operator Platform for Congestion Solutions)
        https://www.gopacs.eu/market-results/

GOPACS is the Dutch grid congestion management platform operated jointly by
TenneT, Liander, Enexis, Stedin, and Westland Infra. It publishes
congestion management orders (redispatch requests) as public market results.

This fetcher scrapes the GOPACS market results page and parses congestion
events relevant to PPA delivery points.

Schema returned:
    event_id        STRING  (UUID generated locally)
    dso_zone        STRING  (e.g. "Liander-Noord", "Enexis-Oost")
    direction       STRING  ("upward" | "downward")
    start_time      TIMESTAMP (UTC)
    end_time        TIMESTAMP (UTC)
    mw_needed       FLOAT
    price_eur_mwh   FLOAT   (winning bid price, if available)
    status          STRING  ("active" | "completed" | "cancelled")
    source_url      STRING
    fetched_at      TIMESTAMP (UTC)
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

GOPACS_BASE_URL = "https://www.gopacs.eu/market-results/"

# Known GOPACS DSO zones with approximate delivery areas
# Used for PPA delivery point → congestion zone matching
DSO_ZONES = [
    "Liander-Noord",
    "Liander-Midden",
    "Liander-Zuid",
    "Enexis-Noord",
    "Enexis-Oost",
    "Enexis-Zuid",
    "Stedin-Rotterdam",
    "Stedin-Utrecht",
    "Stedin-Zeeland",          # Relevant for Zeeland wind PPAs
    "Westland-Infra",
    "TenneT-380kV",
    "TenneT-220kV",
]


class GOPACSFetcher:
    """
    Fetches and parses GOPACS grid congestion events.

    Note: GOPACS does not have a formal REST API. Data is parsed from their
    public market results page. Falls back to mock data when scraping fails
    (e.g. in CI, or when page structure changes).
    """

    def __init__(self, timeout_seconds: int = 15):
        self.timeout = timeout_seconds

    def fetch_announcements(
        self,
        lookback_hours: int = 48,
        dso_zone_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch recent GOPACS congestion announcements.

        Args:
            lookback_hours:    How far back to look for events (default 48h)
            dso_zone_filter:   If set, return only events for this DSO zone

        Returns:
            DataFrame of congestion events
        """
        logger.info(f"Fetching GOPACS announcements (lookback={lookback_hours}h)")

        try:
            df = self._scrape_gopacs(lookback_hours)
        except Exception as e:
            logger.warning(f"GOPACS scrape failed: {e}. Using mock data.")
            df = self._mock_data(lookback_hours)

        if dso_zone_filter:
            df = df[df["dso_zone"].str.contains(dso_zone_filter, case=False, na=False)]
            logger.info(f"Filtered to {len(df)} events in zone '{dso_zone_filter}'")

        return df

    def is_delivery_point_congested(
        self,
        delivery_point: str,
        df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Check if a PPA delivery point is currently in a congested zone.

        Args:
            delivery_point: PPA delivery point description
                            (e.g. "Zeeland", "Midden-Holland", "Flevoland")
            df:             Pre-fetched GOPACS DataFrame (fetches if None)

        Returns:
            dict: {
                "congested": bool,
                "active_events": list[dict],
                "risk_level": "NONE" | "LOW" | "MEDIUM" | "HIGH"
            }
        """
        if df is None:
            df = self.fetch_announcements(lookback_hours=24)

        now = datetime.now(tz=timezone.utc)
        active = df[
            (df["start_time"] <= now) & (df["end_time"] >= now)
        ]

        # Check if any active event's DSO zone matches the delivery point
        relevant = active[
            active["dso_zone"].str.contains(delivery_point, case=False, na=False)
            | active["dso_zone"].str.contains(
                delivery_point.split("-")[0], case=False, na=False
            )
        ]

        if relevant.empty:
            return {"congested": False, "active_events": [], "risk_level": "NONE"}

        max_mw = relevant["mw_needed"].max()
        risk_level = "HIGH" if max_mw > 100 else ("MEDIUM" if max_mw > 50 else "LOW")

        return {
            "congested": True,
            "active_events": relevant.to_dict("records"),
            "risk_level": risk_level,
        }

    def _scrape_gopacs(self, lookback_hours: int) -> pd.DataFrame:
        """
        Attempt to scrape GOPACS market results page.

        GOPACS publishes results as an HTML table at /market-results/
        Structure may change — always falls back to mock on failure.
        """
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Run: pip install httpx beautifulsoup4")

        response = httpx.get(GOPACS_BASE_URL, timeout=self.timeout, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")

        if not tables:
            raise ValueError("No tables found on GOPACS market results page")

        # Parse first results table
        df = pd.read_html(str(tables[0]))[0]
        logger.info(f"Scraped GOPACS table: {len(df)} rows, columns={list(df.columns)}")

        # Normalise column names (GOPACS page structure as of 2025)
        df = self._normalise_gopacs_table(df)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
        return df[df["start_time"] >= cutoff]

    def _normalise_gopacs_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise raw GOPACS HTML table to standard schema."""
        col_map = {
            "Zone": "dso_zone",
            "Direction": "direction",
            "Start": "start_time",
            "End": "end_time",
            "MW": "mw_needed",
            "Price (€/MWh)": "price_eur_mwh",
            "Status": "status",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["start_time", "end_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        for col in ["mw_needed", "price_eur_mwh"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df["source_url"] = GOPACS_BASE_URL
        df["fetched_at"] = datetime.now(tz=timezone.utc)

        return df

    def _mock_data(self, lookback_hours: int) -> pd.DataFrame:
        """
        Realistic mock GOPACS events for local dev / testing.
        Includes a Zeeland congestion event to exercise curtailment risk logic.
        """
        import numpy as np

        now = datetime.now(tz=timezone.utc)
        rng = np.random.default_rng(seed=99)

        events = [
            # Active Zeeland event — affects wind PPA delivery point
            {
                "event_id": str(uuid.uuid4()),
                "dso_zone": "Stedin-Zeeland",
                "direction": "downward",
                "start_time": now - timedelta(hours=2),
                "end_time": now + timedelta(hours=4),
                "mw_needed": float(rng.uniform(80, 150)),
                "price_eur_mwh": float(rng.uniform(10, 40)),
                "status": "active",
                "source_url": GOPACS_BASE_URL,
                "fetched_at": now,
            },
            # Historical completed event
            {
                "event_id": str(uuid.uuid4()),
                "dso_zone": "Enexis-Noord",
                "direction": "upward",
                "start_time": now - timedelta(hours=28),
                "end_time": now - timedelta(hours=24),
                "mw_needed": float(rng.uniform(30, 80)),
                "price_eur_mwh": float(rng.uniform(5, 25)),
                "status": "completed",
                "source_url": GOPACS_BASE_URL,
                "fetched_at": now,
            },
            # Liander event (near-future)
            {
                "event_id": str(uuid.uuid4()),
                "dso_zone": "Liander-Noord",
                "direction": "downward",
                "start_time": now + timedelta(hours=1),
                "end_time": now + timedelta(hours=6),
                "mw_needed": float(rng.uniform(50, 120)),
                "price_eur_mwh": float(rng.uniform(8, 30)),
                "status": "active",
                "source_url": GOPACS_BASE_URL,
                "fetched_at": now,
            },
        ]

        df = pd.DataFrame(events)
        cutoff = now - timedelta(hours=lookback_hours)
        return df[df["start_time"] >= cutoff].reset_index(drop=True)
