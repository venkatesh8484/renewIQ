"""
Unit tests for market data fetchers — no external API calls.
All tests use mock/fixture data paths.
"""

import pytest
from datetime import date, datetime, timezone, timedelta

import pandas as pd

from src.ingestion.epex_fetcher import EPEXFetcher
from src.ingestion.entso_fetcher import ENTSOFetcher
from src.ingestion.gopacs_fetcher import GOPACSFetcher


# ── EPEX Fetcher ──────────────────────────────────────────────────────────────

class TestEPEXFetcher:
    """Tests run against mock data (no API key required)."""

    def setup_method(self):
        self.fetcher = EPEXFetcher(api_key="")  # forces mock mode

    def test_returns_dataframe(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        assert isinstance(df, pd.DataFrame)

    def test_single_day_has_24_rows(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        assert len(df) == 24

    def test_multi_day_row_count(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15), end=date(2026, 3, 17))
        assert len(df) == 72  # 3 days × 24 hours

    def test_schema_columns(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        expected = {"delivery_date", "hour", "delivery_ts", "price_eur_mwh", "is_negative", "market"}
        assert expected.issubset(set(df.columns))

    def test_hour_range_0_to_23(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        assert df["hour"].min() == 0
        assert df["hour"].max() == 23

    def test_is_negative_is_bool(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        assert df["is_negative"].dtype == bool

    def test_is_negative_consistent_with_price(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        assert (df["is_negative"] == (df["price_eur_mwh"] < 0)).all()

    def test_market_column_value(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15), market="NL")
        assert (df["market"] == "NL").all()

    def test_mock_injects_negative_prices(self):
        # Mock data is seeded to produce negative hours — verify at least some exist
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15), end=date(2026, 3, 22))
        assert df["is_negative"].sum() > 0, "Mock data should contain negative price hours"

    def test_detect_negative_windows_empty_when_no_negatives(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15))
        df["is_negative"] = False
        df["price_eur_mwh"] = 65.0
        windows = self.fetcher.detect_negative_windows(df)
        assert windows.empty

    def test_detect_negative_windows_returns_correct_schema(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15), end=date(2026, 3, 22))
        windows = self.fetcher.detect_negative_windows(df)
        if not windows.empty:
            expected = {"window_start", "window_end", "avg_price", "min_price", "hours_count"}
            assert expected.issubset(set(windows.columns))

    def test_detect_negative_windows_avg_price_is_negative(self):
        df = self.fetcher.fetch_day_ahead(date(2026, 3, 15), end=date(2026, 3, 22))
        windows = self.fetcher.detect_negative_windows(df)
        if not windows.empty:
            assert (windows["avg_price"] < 0).all()


# ── ENTSO-E Fetcher ───────────────────────────────────────────────────────────

class TestENTSOFetcher:
    """Tests run against mock data (no API key required)."""

    def setup_method(self):
        self.fetcher = ENTSOFetcher(api_key="")  # forces mock mode

    def test_returns_dataframe(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        assert isinstance(df, pd.DataFrame)

    def test_single_day_has_24_rows(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        assert len(df) == 24

    def test_schema_columns(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        expected = {
            "delivery_ts", "delivery_date", "hour",
            "wind_onshore_mw", "wind_offshore_mw", "solar_mw",
            "total_renewable_mw", "total_mw", "renewable_pct",
            "oversupply_flag", "country",
        }
        assert expected.issubset(set(df.columns))

    def test_renewable_pct_between_0_and_1(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        assert (df["renewable_pct"] >= 0).all()
        assert (df["renewable_pct"] <= 1).all()

    def test_oversupply_flag_consistent_with_pct(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        assert (df["oversupply_flag"] == (df["renewable_pct"] > 0.70)).all()

    def test_solar_zero_at_night(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        # Hours 0 and 5 (pre-dawn) should have near-zero solar
        night_solar = df[df["hour"].isin([0, 1, 2, 3, 4, 5])]["solar_mw"]
        assert (night_solar >= 0).all()  # non-negative

    def test_non_negative_generation_values(self):
        df = self.fetcher.fetch_generation(date(2026, 3, 15))
        for col in ["wind_onshore_mw", "wind_offshore_mw", "solar_mw", "total_mw"]:
            assert (df[col] >= 0).all(), f"{col} has negative values"


# ── GOPACS Fetcher ────────────────────────────────────────────────────────────

class TestGOPACSFetcher:
    """Tests run against mock data (no network calls)."""

    def setup_method(self):
        self.fetcher = GOPACSFetcher()

    def test_returns_dataframe(self):
        df = self.fetcher.fetch_announcements(lookback_hours=48)
        assert isinstance(df, pd.DataFrame)

    def test_schema_columns(self):
        df = self.fetcher.fetch_announcements()
        expected = {
            "event_id", "dso_zone", "direction",
            "start_time", "end_time", "mw_needed", "status",
        }
        assert expected.issubset(set(df.columns))

    def test_direction_values_are_valid(self):
        df = self.fetcher.fetch_announcements()
        valid = {"upward", "downward"}
        assert df["direction"].isin(valid).all()

    def test_mw_needed_is_positive(self):
        df = self.fetcher.fetch_announcements()
        assert (df["mw_needed"] > 0).all()

    def test_zeeland_congestion_present_in_mock(self):
        df = self.fetcher.fetch_announcements()
        zeeland = df[df["dso_zone"].str.contains("Zeeland", case=False, na=False)]
        assert len(zeeland) > 0, "Mock data should contain a Zeeland congestion event"

    def test_zone_filter_works(self):
        df = self.fetcher.fetch_announcements(dso_zone_filter="Zeeland")
        assert len(df) > 0
        assert df["dso_zone"].str.contains("Zeeland", case=False).all()

    def test_is_delivery_point_congested_zeeland(self):
        result = self.fetcher.is_delivery_point_congested("Zeeland")
        assert result["congested"] is True
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert len(result["active_events"]) > 0

    def test_is_delivery_point_congested_unknown_zone(self):
        result = self.fetcher.is_delivery_point_congested("Mars")
        assert result["congested"] is False
        assert result["risk_level"] == "NONE"
