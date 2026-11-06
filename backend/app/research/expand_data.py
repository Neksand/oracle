#!/usr/bin/env python3
"""
Download 2+ years of Binance 5m data and compute Realized Variance.

Uses the production binance_ingest.py module to download real market data,
then computes RV estimators and saves everything for downstream ML training.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.research.binance_ingest import download_symbol, create_session
from app.research.rv_estimators import compute_all_estimators, aggregate_to_daily_rv

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
YEARS = 2.0


def download_and_compute():
    """Download 2+ years of BTC/ETH/SOL 5m data from Binance and compute RV."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(YEARS * 365.25))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"Downloading {YEARS} years of 5m data from {start_dt.date()} to {end_dt.date()}")

    session = create_session()
    all_data = {}

    for asset in ASSETS:
        print(f"\nDownloading {asset}...")
        stats = download_symbol(session, asset, start_ms, end_ms, force_refresh=True)

        if stats.status != "ok":
            print(f"  WARNING: {asset} download status={stats.status}")
            continue

        # Load cached bars
        from app.research.binance_ingest import CACHE_DIR, klines_to_array
        npz_path = CACHE_DIR / f"BINANCE_{asset}.npz"
        bars = np.load(npz_path, allow_pickle=True)["bars"]
        closes = bars["close"]

        # Compute daily RV
        dates, daily_rv = aggregate_to_daily_rv(bars, "rv")

        # Compute log returns
        log_rets = np.diff(np.log(closes))

        all_data[asset] = {
            "prices": closes,
            "timestamps": bars["timestamp"],
            "log_returns": log_rets,
            "daily_rv": daily_rv,
            "daily_dates": dates,
            "n_bars": len(bars),
            "n_daily_rv": len(daily_rv),
            "source": "binance_api",
            "rv_mean": float(daily_rv.mean()),
            "rv_std": float(daily_rv.std()),
        }

        print(f"  {stats.received_bars} bars, {len(daily_rv)} daily RV obs")
        print(f"  RV mean={daily_rv.mean():.8f}, std={daily_rv.std():.8f}")

    session.close()

    # Save combined dataset
    npz_path = DATA_DIR / "rv_data.npz"
    save_dict = {}
    for asset in ASSETS:
        if asset in all_data:
            save_dict[asset] = all_data[asset]

    np.savez_compressed(npz_path, **save_dict)

    # Save metadata
    meta = {
        "assets": [a for a in ASSETS if a in all_data],
        "timeframe": "5m",
        "years": YEARS,
        "n_bars_per_asset": {a: all_data[a]["n_bars"] for a in all_data},
        "n_daily_rv_per_asset": {a: all_data[a]["n_daily_rv"] for a in all_data},
        "total_bars": sum(all_data[a]["n_bars"] for a in all_data),
        "total_daily_rv": sum(all_data[a]["n_daily_rv"] for a in all_data),
        "rv_means": {a: all_data[a]["rv_mean"] for a in all_data},
        "rv_stds": {a: all_data[a]["rv_std"] for a in all_data},
        "date_generated": datetime.now(timezone.utc).isoformat(),
        "source": "binance_api",
        "description": f"{YEARS}+ years of REAL Binance 5m realized variance data for ORACLE project",
    }
    with open(DATA_DIR / "rv_data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATA SAVED: {npz_path}")
    print(f"Assets: {meta['assets']}")
    print(f"Total bars: {meta['total_bars']}")
    print(f"Total daily RV obs: {meta['total_daily_rv']}")
    print(f"{'='*60}")

    return all_data


if __name__ == "__main__":
    data = download_and_compute()