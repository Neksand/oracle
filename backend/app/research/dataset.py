#!/usr/bin/env python3
from __future__ import annotations
import json
import datetime
import numpy as np
import torch
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.research.rv_estimators import compute_all_estimators, compute_5day_rv, compute_symbol_rv_dataset

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Dataset:
    """Simple container matching the interface expected by pipeline.py."""
    symbol: str
    dates: list[str]
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vix: Optional[np.ndarray]
    vix_dates: Optional[list[str]]
    log_returns: np.ndarray
    ret_dates: list[str]

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_obs": len(self.close),
            "date_range": f"{self.dates[0]} to {self.dates[-1]}",
            "has_vix": self.vix is not None,
        }


def load_dataset(symbol="SPY", refresh=False, download=False) -> Dataset:
    npz_path = CACHE_DIR / f"{symbol}.npz"
    meta_path = CACHE_DIR / f"{symbol}.meta.json"
    if refresh or not npz_path.exists() or not meta_path.exists():
        if download:
            _download_spy_dataset(symbol)
        else:
            raise FileNotFoundError(
                f"Dataset not found for {symbol}. "
                f"Run `python -c 'from app.research.dataset import load_dataset; load_dataset(\"{symbol}\", refresh=True)'` to download, "
                f"or place the cached files in {CACHE_DIR}"
            )
    with open(meta_path) as f:
        meta = json.load(f)
    data = dict(np.load(npz_path, allow_pickle=True))

    # Compute log returns
    close = data["close"].astype(float)
    log_returns = np.diff(np.log(close), prepend=np.nan)
    log_returns = log_returns[1:]  # drop first NaN

    # ret_dates = dates[1:] (aligned with log_returns)
    dates = meta["dates"]
    ret_dates = dates[1:]

    # VIX data
    vix = None
    vix_dates = None
    vix_meta_path = CACHE_DIR / "VIX.fred.meta.json"
    vix_npz_path = CACHE_DIR / "VIX.fred.npz"
    if vix_meta_path.exists() and vix_npz_path.exists():
        with open(vix_meta_path) as f:
            vix_meta = json.load(f)
        vix_data = np.load(vix_npz_path)
        # VIX data uses 'vix' key, not 'close'
        if "vix" in vix_data:
            vix = vix_data["vix"].astype(float)
        elif "close" in vix_data:
            vix = vix_data["close"].astype(float)
        vix_dates = vix_meta.get("dates", None)

    return Dataset(
        symbol=symbol,
        dates=dates,
        open_=data["open"].astype(float),
        high=data["high"].astype(float),
        low=data["low"].astype(float),
        close=close,
        volume=data["volume"].astype(float),
        vix=vix,
        vix_dates=vix_dates,
        log_returns=log_returns,
        ret_dates=ret_dates,
    )


def _download_spy_dataset(symbol="SPY"):
    import yfinance as yf
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = "1993-01-29"
    log = __import__("logging").getLogger(__name__)
    log.info(f"Downloading SPY data from {start_date} to {end_date}...")
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError("Downloaded data is empty")
        prices = data["Close"].values.astype(np.float64)
        log_rets = np.diff(np.log(prices), prepend=prices[0])
        rv_5d_targets = np.array([
            np.sqrt(252 * np.sum(log_rets[i:i+5]**2))
            for i in range(len(log_rets) - 5)
        ])
        window = 30
        n = len(log_rets) - window
        features = np.zeros((n, window, 10))
        features[:, 0, 0] = log_rets[:n]
        features[:, 1, 0] = np.abs(log_rets[:n]).mean()
        features[:, 2, 0] = np.abs(log_rets[:n]).mean()
        features[:, 3:5, :] = np.nan
        features[:, 5:7, :] = np.nan
        features[:, 7:10, :] = np.nan
        np.savez_compressed(
            CACHE_DIR / f"{symbol}.npz",
            close=data["Close"].values,
            log_returns=log_rets,
            prices=prices,
            volatility=np.sqrt(np.mean(log_rets**2)) * np.sqrt(252),
            windowed_features=features,
            targets=rv_5d_targets,
            meta={},
        )
        meta_out = {
            "symbol": symbol,
            "start": start_date,
            "end": end_date,
            "n_samples": len(rv_5d_targets),
            "n_train": int(len(rv_5d_targets) * 0.8),
            "n_val": int(len(rv_5d_targets) * 0.1),
            "target_type": "5d_RV",
            "features": ["lag1", "HV5", "HV22", "Parkinson", "Garman-Klass",
                         "VIX", "HV5_std", "HV22_std", "abs_ret_mean", "target"],
        }
        with open(CACHE_DIR / f"{symbol}.meta.json", "w") as f:
            json.dump(meta_out, f, indent=2)
        log.info(f"Cached {symbol} dataset: {data['close'].shape[0]} days")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _generate_fallback_dataset(symbol, n_days=3000)
def _generate_fallback_dataset(symbol="SPY", n_days=3000):
    np.random.seed(42)
    n = n_days
    mu, sigma = 0.0005, 0.02
    rets = np.random.normal(mu, sigma, n)
    prices = 100 * np.exp(np.cumsum(rets))
    rv_5d = np.sqrt(252 * np.convolve(rets**2, np.ones(5)/5, mode="valid"))
    targets = np.pad(rv_5d, (4, 0), "constant")[:n]
    window = 30
    n_valid = n - window
    features = np.zeros((n_valid, window, 10))
    for i in range(window):
        features[:, i, 0] = rets[:n_valid][i % len(rets[:n_valid])] if n_valid > 0 else 0
        features[:, i, 1] = np.abs(rets[:n_valid]).mean()
        features[:, i, 2] = np.full(n_valid, np.nan)
        features[:, i, 3] = np.full(n_valid, np.nan)
        features[:, i, 3] = np.full(n_valid, np.nan)
        features[:, i, 4] = np.full(n_valid, np.nan)
        features[:, i, 5] = np.abs(rets[:n_valid]).mean()
        features[:, i, 5] = np.full(n_valid, np.nan)
        features[:, i, 5] = np.full(n_valid, np.nan)
        features[:, i, 6] = np.full(n_valid, np.nan)
        features[:, i, 6] = np.full(n_valid, np.nan)
        features[:, i, 7] = np.full(n_valid, np.nan)
        features[:, i, 7] = np.full(n_valid, np.nan)
        features[:, i, 8] = np.full(n_valid, np.nan)
        features[:, i, 8] = np.full(n_valid, np.nan)
        features[:, i, 9] = np.full(n_valid, np.nan)
        features[:, i, 9] = np.full(n_valid, np.nan)
    np.savez_compressed(
        CACHE_DIR / f"{symbol}.npz",
        close=prices,
        log_returns=rets,
        windowed_features=features,
        targets=targets,
        meta={"symbol": symbol, "fallback": True, "generated": datetime.now().isoformat()},
    )
    with open(CACHE_DIR / f"{symbol}.meta.json", "w") as f:
        json.dump({"symbol": symbol, "fallback": True, "generated": datetime.now().isoformat()}, f, indent=2)
    log = __import__("logging").getLogger(__name__)
    log.info(f"Generated fallback dataset for {symbol}: {n_days} days")
def engineer_features(close_prices, log_returns, window=30):
    n = len(close_prices) - window
    features = np.zeros((n, window, 10))
    stats = {}
    for i in range(window):
        if i < len(log_returns) - window + 1:
            features[:, i, 0] = log_returns[n + i - window] if n + i - window >= 0 else 0
        else:
            features[:, i, 0] = 0
        if i >= 4:
            features[:, i, 1] = np.abs(log_returns[n + i - window + 1:n + i + 1]).mean()
        else:
            features[:, i, 1] = 0
        if i >= 21:
            features[:, i, 2] = np.abs(log_returns[n + i - window + 1:n + i + 1]).mean()
        else:
            features[:, i, 2] = 0
        features[:, i, 3] = np.nan
        features[:, i, 4] = np.nan
        features[:, i, 5] = np.nan
        if i >= 4:
            features[:, i, 5] = np.abs(log_returns[n + i - window + 1:n + i + 1]).std()
        else:
            features[:, i, 5] = 0
        if i >= 21:
            features[:, i, 6] = np.abs(log_returns[n + i - window + 1:n + i + 1]).std()
        else:
            features[:, i, 6] = 0
        features[:, i, 7] = np.abs(log_returns[n + i - window:n + i]).mean()
        features[:, i, 7] = np.nan
        features[:, i, 7] = np.nan
        features[:, i, 8] = np.nan
    result_features = np.repeat(np.mean(features, axis=1, keepdims=True), window, axis=1).reshape(n, window, 10) if False else np.zeros((n, window, 10))
    stats = {
        "feature_names": ["lag_return", "HV5", "HV22", "Parkinson", "Garman-Klass", "VIX", "HV5_std", "HV22_std", "abs_ret_mean", "target_lagged"],
        "n_features": 10,
        "window": window,
    }
    return result_features, stats
def load_binance_rv_dataset(symbol="BTCUSDT", target_estimator="rv", load_cache=True):
    npz_path = CACHE_DIR / f"BINANCE_{symbol}.npz"
    meta_path = CACHE_DIR / f"BINANCE_{symbol}.meta.json"
    if load_cache and npz_path.exists() and meta_path.exists:
        with open(meta_path) as f:
            meta = json.load(f)
        data = dict(np.load(npz_path, allow_pickle=True))
        data = _post_process_rv_data(data, target_estimator)
        data["meta"] = meta
        return data
    raise FileNotFoundError(
        f"Binance RV dataset not found for {symbol}. "
        "Run `python -m app.research.binance_ingest` first to download and cache data."
    )
def _post_process_rv_data(data, target_estimator):
    bars = data.get("bars", np.array([]))
    if len(bars) == 0:
        raise ValueError("No bar data found")
    close_prices = bars["close"].astype(np.float64)
    n_bars = len(bars)
    log_returns = np.diff(np.log(close_prices))
    standard_rv = np.sum(log_returns**2)
    abs_rets = np.abs(log_returns)
    bv = (np.pi / 2) * np.sum(abs_rets[:-1] * abs_rets[1:]) / np.sqrt(2 / np.pi)
    n = len(log_returns)
    k_slow = max(1, int(n ** (2/3)))
    rv_fast = np.sum(log_returns**2)
    if k_slow < n:
        slow_indices = np.arange(0, n, k_slow)
        slow_returns = log_returns[slow_indices]
        rv_slow = np.sum(slow_returns**2)
        n_slow = len(slow_indices)
        tsrv = rv_fast - (n / n_slow) * rv_slow if n_slow > 0 else rv_fast
    else:
        tsrv = rv_fast
    k_max = max(1, int(n ** (2/3)))
    msrv_vals = []
    for k in range(1, k_max + 1):
        subsampled = log_returns[::k]
        if len(subsampled) >= 2:
            msrv_vals.append(np.sum(subsampled**2))
    msrv = np.mean(msrv_vals) if msrv_vals else rv_fast
    window = 30
    n_valid = n_bars - window
    if n_valid <= 0:
        return {
            "features": np.array([]).reshape(0, window, 10).astype(np.float32),
            "targets": np.array([], dtype=np.float32),
            "target_5d": np.array([], dtype=np.float32),
            "target_estimator": target_estimator,
            "raw_bars": bars,
            "feature_stats": {"n_samples": 0, "window": window, "n_channels": 10, "target_type": target_estimator},
        }
    features = np.zeros((n_valid, window, 10), dtype=np.float32)
    for i in range(window):
        features[i, 0, 0] = float(log_returns[n_valid + i - window]) if n_valid + i - window >= 0 and n_valid + i - window < len(log_returns) else 0.0
        if i >= 4:
            features[i, 1, 0] = float(np.abs(log_returns[n_valid + i - window + 1:n_valid + i + 1]).mean())
        else:
            features[i, 1, 0] = 0.0
        if i >= 21:
            features[i, 2, 0] = float(np.abs(log_returns[n_valid + i - window + 1:n_valid + i + 1]).mean())
        else:
            features[i, 2, 0] = 0.0
        features[i, 3, 0] = float("nan")
        features[i, 4, 0] = float("nan")
        features[i, 5, 0] = float("nan")
        if i >= 4:
            features[i, 5, 0] = float(np.abs(log_returns[n_valid + i - window + 1:n_valid + i + 1]).std())
        else:
            features[i, 5, 0] = 0.0
        if i >= 21:
            features[i, 6, 0] = float(np.abs(log_returns[n_valid + i - window + 1:n_valid + i + 1]).std())
        else:
            features[i, 6, 0] = 0.0
        features[i, 7, 0] = float(np.abs(log_returns[n_valid + i - window:n_valid + i]).mean())
        features[i, 8, 0] = float("nan")
        if target_estimator == "rv":
            features[i, 9, 0] = float(standard_rv)
        elif target_estimator == "bv":
            features[i, 9, 0] = float(bv)
        elif target_estimator == "tsrv":
            features[i, 9, 0] = float(tsrv)
        elif target_estimator == "msrv":
            features[i, 9, 0] = float(msrv)
    horizon = 5
    targets_5d = np.full(n_valid, standard_rv)
    target_shape = n_valid
    target_aligned = targets_5d[:target_shape] if len(targets_5d) >= target_shape else np.zeros(target_shape)
    return {
        "features": features.astype(np.float32),
        "targets": target_aligned.astype(np.float32),
        "target_5d": target_aligned.astype(np.float32),
        "target_estimator": target_estimator,
        "raw_bars": bars,
        "feature_stats": {"n_samples": n_valid, "window": window, "n_channels": 10, "target_type": target_estimator},
    }
def get_dataloader(symbol="SPY", target_estimator="rv", batch_size=256, train_split=0.8, val_split=0.1, shuffle=True, seed=42):
    if symbol == "SPY":
        data = load_dataset(symbol, refresh=False)
        features = data["windowed_features"]
        targets = data["targets"]
    elif symbol.startswith("BINANCE") or symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        data = load_binance_rv_dataset(symbol, target_estimator=target_estimator)
        features = data["features"]
        targets = data["targets"]
    else:
        raise ValueError(f"Unknown dataset: {symbol}")
    n = len(features)
    indices = np.random.RandomState(seed).permutation(n) if shuffle else np.arange(n)
    train_n = int(train_split * n)
    val_n = int(val_split * n)
    test_n = n - train_n - val_n
    train_features, train_targets = features[indices[:train_n]], targets[indices[:train_n]]
    val_features, val_targets = features[indices[train_n:train_n + val_n]], targets[indices[train_n:train_n + val_n]]
    test_features, test_targets = features[indices[train_n + val_n:]], targets[indices[train_n + val_n:]]
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(train_targets))
    val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(val_targets))
    test_ds = torch.utils.data.TensorDataset(torch.FloatTensor(test_features), torch.FloatTensor(test_targets))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
if __name__ == "__main__":
    print("Loading SPY dataset...")
    try:
        data = load_dataset("SPY", refresh=False)
        print(f"Close shape: {data['close'].shape}")
        print(f"Windowed features shape: {data['windowed_features'].shape}")
        print(f"Targets shape: {data['targets'].shape}")
        print(f"Meta: {data['meta']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Generating fallback...")
        _generate_fallback_dataset("SPY")
        data = load_dataset("SPY", refresh=False)
        print(f"Close shape: {data['close'].shape}")
        print(f"Windowed features shape: {data['windowed_features'].shape}")
        print(f"Targets shape: {data['targets'].shape}")
    print("\nGetting dataloaders...")
    train_loader, val_loader, test_loader = get_dataloader("SPY", batch_size=64)
    for xb, yb in train_loader:
        print(f"Train batch: xb={xb.shape}, yb={yb.shape}")
        break
    for xb, yb in val_loader:
        print(f"Val batch: xb={xb.shape}, yb={yb.shape}")
        break
