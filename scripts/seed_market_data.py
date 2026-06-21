"""
Seed Market Data Script
------------------------
Pulls 90 days of EPEX NL, ENTSO-E generation, and GOPACS data
and writes them to ADLS Gen2 Bronze layer as JSON/Parquet files
(which Lakeflow Auto Loader then picks up).

Usage:
    python scripts/seed_market_data.py --days 90 --market NL

Environment variables required:
    DATABRICKS_HOST, DATABRICKS_TOKEN   — for ADLS write via dbutils
    ENTSO_E_API_KEY                     — for live data (optional; uses mock if absent)
    ADLS_STORAGE_ACCOUNT                — e.g. "renewiqstorage"

For local dev (no ADLS): use --output-dir to write to a local folder instead.
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.epex_fetcher import EPEXFetcher
from src.ingestion.entso_fetcher import ENTSOFetcher
from src.ingestion.gopacs_fetcher import GOPACSFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Seed market data into RenewIQ Bronze layer")
    parser.add_argument("--days", type=int, default=90, help="Number of days to back-fill")
    parser.add_argument("--market", type=str, default="NL", help="Market code (default: NL)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local output directory (skips ADLS write — for local dev)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["epex", "entso", "gopacs"],
        help="Which sources to seed (default: all)",
    )
    return parser.parse_args()


def seed_epex(start: date, end: date, market: str, output_dir: Path) -> int:
    """Fetch EPEX prices and write to Bronze."""
    logger.info(f"[EPEX] Fetching {(end - start).days + 1} days of day-ahead prices")
    fetcher = EPEXFetcher()
    df = fetcher.fetch_day_ahead(start, end=end, market=market)

    # Write one JSON file per day (matches Auto Loader cloudFiles pattern)
    count = 0
    for day, group in df.groupby("delivery_date"):
        out_path = output_dir / "epex" / f"{day}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_api": "entso-e-transparency",
            "market": market,
            "fetch_date": str(day),
            "ingestion_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "raw_payload": group.drop(columns=["delivery_date"]).to_json(orient="records"),
        }
        out_path.write_text(json.dumps(payload, default=str))
        count += 1

    logger.info(f"[EPEX] Written {count} daily files → {output_dir}/epex/")
    return len(df)


def seed_entso(start: date, end: date, output_dir: Path) -> int:
    """Fetch ENTSO-E generation mix and write to Bronze."""
    logger.info(f"[ENTSO-E] Fetching generation mix")
    fetcher = ENTSOFetcher()
    df = fetcher.fetch_generation(start, end=end)

    count = 0
    for day, group in df.groupby("delivery_date"):
        out_path = output_dir / "entso_generation" / f"{day}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_api": "entso-e-transparency",
            "country": "NL",
            "fetch_date": str(day),
            "ingestion_ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "raw_payload": group.drop(columns=["delivery_date"]).to_json(orient="records", date_format="iso"),
        }
        out_path.write_text(json.dumps(payload, default=str))
        count += 1

    logger.info(f"[ENTSO-E] Written {count} daily files → {output_dir}/entso_generation/")
    return len(df)


def seed_gopacs(output_dir: Path) -> int:
    """Fetch GOPACS congestion events and write to Bronze."""
    logger.info("[GOPACS] Fetching congestion announcements (last 48h)")
    fetcher = GOPACSFetcher()
    df = fetcher.fetch_announcements(lookback_hours=48)

    out_path = output_dir / "gopacs" / f"{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "gopacs.eu",
        "ingestion_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_payload": df.to_json(orient="records", date_format="iso"),
    }
    out_path.write_text(json.dumps(payload, default=str))
    logger.info(f"[GOPACS] Written {len(df)} events → {out_path}")
    return len(df)


def write_to_adls(local_dir: Path, storage_account: str, container: str = "raw"):
    """
    Upload seeded files to ADLS Gen2 Bronze container via Azure CLI.
    Requires: az login (or managed identity in Databricks cluster)
    """
    import subprocess

    adls_url = f"https://{storage_account}.dfs.core.windows.net"
    logger.info(f"Uploading {local_dir} → {adls_url}/{container}/")
    result = subprocess.run(
        [
            "az", "storage", "blob", "upload-batch",
            "--source", str(local_dir),
            "--destination", f"https://{storage_account}.blob.core.windows.net/{container}",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"ADLS upload failed: {result.stderr}")
    else:
        logger.info("ADLS upload complete")


def main():
    args = parse_args()

    end_date = date.today() - timedelta(days=1)   # yesterday
    start_date = end_date - timedelta(days=args.days - 1)

    logger.info(f"Seeding market data: {start_date} → {end_date} | Market: {args.market}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "bronze_seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = {}

    if "epex" in args.sources:
        totals["epex_rows"] = seed_epex(start_date, end_date, args.market, output_dir)

    if "entso" in args.sources:
        totals["entso_rows"] = seed_entso(start_date, end_date, output_dir)

    if "gopacs" in args.sources:
        totals["gopacs_events"] = seed_gopacs(output_dir)

    # Optionally upload to ADLS
    storage_account = os.getenv("ADLS_STORAGE_ACCOUNT")
    if storage_account and not args.output_dir:
        write_to_adls(output_dir, storage_account)
    else:
        logger.info(f"Local seed complete. Files written to: {output_dir}")
        logger.info("To upload to ADLS: set ADLS_STORAGE_ACCOUNT env var and re-run without --output-dir")

    logger.info(f"Seed complete: {totals}")


if __name__ == "__main__":
    main()
