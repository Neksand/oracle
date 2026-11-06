#!/usr/bin/env python3
"""
Binance 5-minute klines bulk downloader for ORACLE volatility research.

Downloads spot market data (BTCUSDT, ETHUSDT, SOLUSDT) for 2 years,
computes realized variance estimators, and caches as structured numpy arrays.

Rate limits: 1200 requests/minute (IP), 50 requests/second (UID)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "5m"
LIMIT_PER_REQUEST = 1000  # Binance max
RATE_LIMIT_PER_MIN = 1100  # leave headroom
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Structured array dtype for 5m bars
BAR_DTYPE = np.dtype([
    ("timestamp", "i8"),      # UTC milliseconds (Binance open time)
    ("open", "f8"),
    ("high", "f8"),
    ("low", "f8"),
    ("close", "f8"),
    ("volume", "f8"),
    ("close_time", "i8"),
    ("quote_asset_volume", "f8"),
    ("num_trades", "i4"),
    ("taker_buy_base_volume", "f8"),
    ("taker_buy_quote_volume", "f8"),
])

META_DTYPE = np.dtype([
    ("symbol", "U16"),
    ("interval", "U8"),
    ("start_ms", "i8"),
    ("end_ms", "i8"),
    ("num_bars", "i4"),
    ("gaps", "i4"),
    ("checksum", "U64"),
    ("downloaded_at", "U32"),
])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("binance_ingest")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DownloadStats:
    symbol: str
    requested_bars: int
    received_bars: int
    gaps_found: int
    duplicates_removed: int
    duration_sec: float
    checksum: str
    status: str  # "ok" | "partial" | "failed"


# ---------------------------------------------------------------------------
# HTTP session with retry/backoff
# ---------------------------------------------------------------------------


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------


def fetch_klines(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch klines for a time range, handling pagination."""
    all_klines = []
    current_start = start_ms
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT_PER_REQUEST,
    }

    while current_start < end_ms:
        params["startTime"] = current_start
        params["endTime"] = end_ms

        try:
            resp = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log.warning("Request failed for %s @ %d: %s", symbol, current_start, e)
            break

        if resp.status_code == 429:
            # Rate limited - wait and retry
            retry_after = int(resp.headers.get("Retry-After", "60"))
            log.warning("Rate limited, sleeping %ds", retry_after)
            time.sleep(retry_after)
            continue

        if resp.status_code != 200:
            log.error("HTTP %d for %s: %s", resp.status_code, symbol, resp.text[:200])
            break

        data = resp.json()
        if not data:
            break

        all_klines.extend(data)

        # Next page starts after the last returned bar's close time
        last_bar = data[-1]
        current_start = last_bar[6] + 1  # close_time + 1ms

        # Respect rate limit
        time.sleep(60.0 / RATE_LIMIT_PER_MIN)

        if len(data) < LIMIT_PER_REQUEST:
            break

    return all_klines


def klines_to_array(klines: list[list]) -> np.ndarray:
    """Convert Binance kline JSON to structured numpy array."""
    if not klines:
        return np.array([], dtype=BAR_DTYPE)

    arr = np.empty(len(klines), dtype=BAR_DTYPE)
    for i, k in enumerate(klines):
        arr[i] = (
            k[0],   # open time (ms)
            float(k[1]),  # open
            float(k[2]),  # high
            float(k[3]),  # low
            float(k[4]),  # close
            float(k[5]),  # volume
            k[6],   # close time (ms)
            float(k[7]),  # quote asset volume
            int(k[8]),    # number of trades
            float(k[9]),  # taker buy base volume
            float(k[10]), # taker buy quote volume
        )
    return arr


def deduplicate_and_sort(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove duplicate timestamps (keep first), sort by timestamp."""
    if len(arr) == 0:
        return arr, 0

    # Sort by timestamp
    arr = np.sort(arr, order="timestamp")

    # Find unique timestamps
    _, unique_idx = np.unique(arr["timestamp"], return_index=True)
    duplicates = len(arr) - len(unique_idx)

    if duplicates > 0:
        arr = arr[unique_idx]

    return arr, duplicates


def detect_gaps(arr: np.ndarray, expected_interval_ms: int = 5 * 60 * 1000) -> int:
    """Count gaps larger than 1.5x expected interval."""
    if len(arr) < 2:
        return 0

    diffs = np.diff(arr["timestamp"])
    gap_threshold = int(expected_interval_ms * 1.5)
    gaps = int(np.sum(diffs > gap_threshold))
    return gaps


def compute_checksum(arr: np.ndarray) -> str:
    """SHA256 of the raw bytes for integrity verification."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:64]


def save_cache(symbol: str, bars: np.ndarray, meta: dict) -> tuple[Path, Path]:
    """Save bars and metadata to cache directory."""
    npz_path = CACHE_DIR / f"BINANCE_{symbol}.npz"
    meta_path = CACHE_DIR / f"BINANCE_{symbol}.meta.json"

    # Save structured array
    np.savez_compressed(npz_path, bars=bars)

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return npz_path, meta_path


def load_existing_cache(symbol: str) -> tuple[np.ndarray | None, dict | None]:
    """Load existing cache if present and valid."""
    npz_path = CACHE_DIR / f"BINANCE_{symbol}.npz"
    meta_path = CACHE_DIR / f"BINANCE_{symbol}.meta.json"

    if not npz_path.exists() or not meta_path.exists():
        return None, None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        data = np.load(npz_path)
        bars = data["bars"]
        # Verify checksum
        if compute_checksum(bars) != meta.get("checksum"):
            log.warning("Checksum mismatch for %s, will re-download", symbol)
            return None, None
        return bars, meta
    except Exception as e:
        log.warning("Failed to load cache for %s: %s", symbol, e)
        return None, None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def download_symbol(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    force_refresh: bool = False,
) -> DownloadStats:
    """Download and cache data for a single symbol."""
    start_time = time.time()

    # Check existing cache
    if not force_refresh:
        existing_bars, existing_meta = load_existing_cache(symbol)
        if existing_bars is not None:
            # Determine what's missing
            cached_end = existing_meta.get("end_ms", 0)
            if cached_end >= end_ms:
                log.info("%s: cache complete (up to %s)", symbol,
                         datetime.fromtimestamp(cached_end / 1000, tz=timezone.utc))
                return DownloadStats(
                    symbol=symbol,
                    requested_bars=0,
                    received_bars=len(existing_bars),
                    gaps_found=existing_meta.get("gaps", 0),
                    duplicates_removed=0,
                    duration_sec=time.time() - start_time,
                    checksum=existing_meta["checksum"],
                    status="ok",
                )
            # Need incremental update
            start_ms = cached_end + 1
            log.info("%s: incremental update from %s", symbol,
                     datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc))

    # Fetch new data
    log.info("%s: downloading %s to %s", symbol,
             datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
             datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc))

    klines = fetch_klines(session, symbol, start_ms, end_ms)
    if not klines:
        return DownloadStats(
            symbol=symbol, requested_bars=0, received_bars=0,
            gaps_found=0, duplicates_removed=0,
            duration_sec=time.time() - start_time,
            checksum="", status="failed",
        )

    new_bars = klines_to_array(klines)
    new_bars, dup = deduplicate_and_sort(new_bars)

    # Merge with existing
    if not force_refresh:
        existing_bars, _ = load_existing_cache(symbol)
        if existing_bars is not None:
            # Combine and deduplicate again
            combined = np.concatenate([existing_bars, new_bars])
            combined, dup = deduplicate_and_sort(combined)
            new_bars = combined

    # Final validation
    gaps = detect_gaps(new_bars)
    checksum = compute_checksum(new_bars)

    # Metadata
    meta = {
        "symbol": symbol,
        "interval": INTERVAL,
        "start_ms": int(new_bars["timestamp"].min()) if len(new_bars) > 0 else start_ms,
        "end_ms": int(new_bars["timestamp"].max()) if len(new_bars) > 0 else end_ms,
        "num_bars": len(new_bars),
        "gaps": gaps,
        "checksum": checksum,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save
    save_cache(symbol, new_bars, meta)

    duration = time.time() - start_time
    log.info("%s: done — %d bars, %d gaps, %d dup removed, %.1fs",
             symbol, len(new_bars), gaps, dup, duration)

    return DownloadStats(
        symbol=symbol,
        requested_bars=(end_ms - start_ms) // (5 * 60 * 1000),
        received_bars=len(new_bars),
        gaps_found=gaps,
        duplicates_removed=dup,
        duration_sec=duration,
        checksum=checksum,
        status="ok" if gaps == 0 else "partial",
    )


def main():
    parser = argparse.ArgumentParser(description="Binance 5m klines bulk downloader")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS,
                        help="Symbols to download (default: BTCUSDT ETHUSDT SOLUSDT)")
    parser.add_argument("--years", type=float, default=2.0,
                        help="Years of history to download (default: 2.0)")
    parser.add_argument("--force", action="store_true",
                        help="Force full re-download, ignore cache")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD), default: now")
    args = parser.parse_args()

    # Time range
    end_dt = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc) if args.end_date \
        else datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int((end_dt - timedelta(days=args.years * 365.25)).timestamp() * 1000)

    log.info("=" * 60)
    log.info("Binance Ingest: %s | %s to %s",
             ", ".join(args.symbols),
             datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date(),
             datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date())
    log.info("=" * 60)

    session = create_session()
    stats_list = []

    for symbol in args.symbols:
        try:
            stats = download_symbol(session, symbol, start_ms, end_ms, args.force)
            stats_list.append(stats)
        except Exception as e:
            log.exception("Failed to download %s: %s", symbol, e)
            stats_list.append(DownloadStats(
                symbol=symbol, requested_bars=0, received_bars=0,
                gaps_found=0, duplicates_removed=0,
                duration_sec=0, checksum="", status="failed",
            ))

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    for s in stats_list:
        log.info("  %-10s | %6d bars | %2d gaps | %s | %.1fs",
                 s.symbol, s.received_bars, s.gaps_found, s.status, s.duration_sec)

    # Save aggregate metadata
    aggregate_meta = {
        "symbols": args.symbols,
        "interval": INTERVAL,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "years": args.years,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "stats": [asdict(s) for s in stats_list],
    }
    meta_path = CACHE_DIR / "BINANCE_aggregate.meta.json"
    with open(meta_path, "w") as f:
        json.dump(aggregate_meta, f, indent=2)
    log.info("Aggregate metadata saved to %s", meta_path)

    # Exit code
    failed = sum(1 for s in stats_list if s.status == "failed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()