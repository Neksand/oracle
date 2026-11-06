#!/usr/bin/env python3
"""
Realized Variance Estimators for 5-minute intraday data.

Implements:
- RV (Realized Variance): sum of squared 5-min returns
- BV (Bipower Variation): robust to jumps
- TSRV (Two-Scale Realized Variance): Zhang et al. 2005 - noise robust
- MSRV (Multi-Scale Realized Variance): Zhang 2006 - optimal noise reduction

References:
- Barndorff-Nielsen & Shephard (2004) - BV
- Zhang, Mykland, Aït-Sahalia (2005) - TSRV
- Zhang (2006) - MSRV
- Patton (2011) - Volatility forecast comparison with imperfect proxies
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RVResult:
    """Container for realized variance estimates."""
    rv: float           # Standard realized variance
    rv_annualized: float
    bv: float           # Bipower variation
    bv_annualized: float
    tsrv: float         # Two-scale RV
    tsrv_annualized: float
    msrv: float         # Multi-scale RV
    msrv_annualized: float
    n_bars: int
    noise_variance: float  # Estimated microstructure noise variance


def log_returns(close_prices: np.ndarray) -> np.ndarray:
    """Compute log returns from close prices."""
    return np.diff(np.log(close_prices))


def realized_variance(returns: np.ndarray) -> float:
    """Standard realized variance: sum of squared returns."""
    return float(np.sum(returns ** 2))


def bipower_variation(returns: np.ndarray) -> float:
    """
    Bipower variation (Barndorff-Nielsen & Shephard 2004).
    Robust to jumps: uses product of adjacent absolute returns.
    """
    if len(returns) < 2:
        return 0.0
    mu1 = np.sqrt(2 / np.pi)  # E[|Z|] for Z ~ N(0,1)
    abs_ret = np.abs(returns)
    bv = (np.pi / 2) * np.sum(abs_ret[:-1] * abs_ret[1:]) / (mu1 ** 2)
    return float(bv)


def two_scale_rv(returns: np.ndarray, k_slow: Optional[int] = None) -> tuple[float, float]:
    """
    Two-Scale Realized Variance (Zhang et al. 2005).
    Combines fast (noisy) and slow (less noisy) sampling.
    
    RV_tsrv = RV_fast - (n_fast / n_slow) * RV_slow
    
    Returns (tsrv, noise_variance_estimate)
    """
    n = len(returns)
    if n < 10:
        return 0.0, 0.0

    # Fast scale: all returns
    rv_fast = realized_variance(returns)

    # Slow scale: optimal k ~ n^(2/3) for noise robust estimation
    if k_slow is None:
        k_slow = max(1, int(n ** (2/3)))
    
    # Subsample every k_slow
    slow_returns = returns[::k_slow]
    if len(slow_returns) < 2:
        return rv_fast, 0.0
    
    rv_slow = realized_variance(slow_returns)
    
    # TSRV combination
    n_fast = n
    n_slow = len(slow_returns)
    tsrv = rv_fast - (n_fast / n_slow) * rv_slow
    
    # Noise variance estimate
    noise_var = rv_fast / (2 * n_fast)
    
    return float(max(tsrv, 0.0)), float(noise_var)


def multi_scale_rv(returns: np.ndarray, k_max: Optional[int] = None) -> tuple[float, float]:
    """
    Multi-Scale Realized Variance (Zhang 2006).
    Averages RV over multiple subsampling scales for optimal noise reduction.
    
    MSRV = (1/K) * sum_{k=1}^K RV^{(k)}  where RV^{(k)} uses every k-th return
    
    Returns (msrv, noise_variance_estimate)
    """
    n = len(returns)
    if n < 10:
        return 0.0, 0.0

    if k_max is None:
        # Optimal k_max ~ n^(2/3) (Bandorff-Nielsen et al.)
        k_max = max(1, int(n ** (2/3)))
    
    rv_scales = []
    for k in range(1, k_max + 1):
        subsampled = returns[::k]
        if len(subsampled) >= 2:
            rv_scales.append(realized_variance(subsampled))
    
    if not rv_scales:
        return 0.0, 0.0
    
    msrv = float(np.mean(rv_scales))
    
    # Noise variance from finest scale
    noise_var = rv_scales[0] / (2 * n) if rv_scales else 0.0
    
    return msrv, float(noise_var)


def compute_all_estimators(
    close_prices: np.ndarray,
    annualization_factor: float = 252 * 288  # 5-min bars per year
) -> RVResult:
    """
    Compute all RV estimators from 5-minute close prices.
    
    annualization_factor: bars per year (252 trading days * 288 5-min bars/day = 72576)
    """
    if len(close_prices) < 2:
        return RVResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    returns = log_returns(close_prices)
    n = len(returns)

    # Standard RV
    rv = realized_variance(returns)
    
    # Bipower Variation
    bv = bipower_variation(returns)
    
    # Two-Scale RV
    tsrv, noise_var_tsrv = two_scale_rv(returns)
    
    # Multi-Scale RV
    msrv, noise_var_msrv = multi_scale_rv(returns)
    
    # Use average noise estimate
    noise_var = (noise_var_tsrv + noise_var_msrv) / 2

    return RVResult(
        rv=rv,
        rv_annualized=rv * annualization_factor,
        bv=bv,
        bv_annualized=bv * annualization_factor,
        tsrv=tsrv,
        tsrv_annualized=tsrv * annualization_factor,
        msrv=msrv,
        msrv_annualized=msrv * annualization_factor,
        n_bars=n,
        noise_variance=noise_var,
    )


def aggregate_to_daily_rv(
    bars_5m: np.ndarray,
    target: str = "rv"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate 5-minute RV estimates to daily level.
    
    Returns (dates, daily_rv) where dates are UTC midnight timestamps (ms).
    """
    if len(bars_5m) == 0:
        return np.array([], dtype="i8"), np.array([], dtype="f8")

    timestamps = bars_5m["timestamp"]
    closes = bars_5m["close"]

    # Convert ms to datetime for grouping
    dates = (timestamps // 86400000) * 86400000  # UTC midnight ms
    unique_dates = np.unique(dates)

    daily_rv = []
    daily_dates = []

    for d in unique_dates:
        mask = dates == d
        day_closes = closes[mask]
        if len(day_closes) < 10:  # Need minimum bars for reliable RV
            continue
        
        result = compute_all_estimators(day_closes)
        daily_dates.append(d)
        daily_rv.append(getattr(result, target))

    return np.array(daily_dates, dtype="i8"), np.array(daily_rv, dtype="f8")


def compute_5day_rv(
    daily_rv: np.ndarray,
    horizon_days: int = 5
) -> np.ndarray:
    """
    Compute forward 5-day realized variance (target for forecasting).
    
    y_t = mean of RV_{t+1} ... RV_{t+5}
    """
    if len(daily_rv) <= horizon_days:
        return np.array([])

    # Rolling mean of next horizon_days
    kernel = np.ones(horizon_days) / horizon_days
    # Shift by 1 to make it forward-looking (target at t uses t+1...t+5)
    forward_rv = np.convolve(daily_rv, kernel, mode="valid")[horizon_days-1:]
    return forward_rv


# ---------------------------------------------------------------------------
# Convenience: compute RV for a full symbol dataset
# ---------------------------------------------------------------------------


def compute_symbol_rv_dataset(bars_5m: np.ndarray) -> dict:
    """
    Full pipeline: 5m bars -> daily RV estimators -> 5-day forward targets.
    
    Returns dict with all estimators at daily frequency and 5-day targets.
    """
    if len(bars_5m) == 0:
        return {}

    # Daily RV for each estimator
    dates, rv_daily = aggregate_to_daily_rv(bars_5m, "rv")
    _, bv_daily = aggregate_to_daily_rv(bars_5m, "bv")
    _, tsrv_daily = aggregate_to_daily_rv(bars_5m, "tsrv")
    _, msrv_daily = aggregate_to_daily_rv(bars_5m, "msrv")

    # 5-day forward targets
    rv_5d = compute_5day_rv(rv_daily)
    bv_5d = compute_5day_rv(bv_daily)
    tsrv_5d = compute_5day_rv(tsrv_daily)
    msrv_5d = compute_5day_rv(msrv_daily)

    # Align dates (target at t corresponds to RV over t+1...t+5)
    target_dates = dates[horizon_days:len(dates)-horizon_days+1] if len(dates) > 2*horizon_days else dates[:-5]

    return {
        "dates": dates.tolist(),
        "rv_daily": rv_daily.tolist(),
        "bv_daily": bv_daily.tolist(),
        "tsrv_daily": tsrv_daily.tolist(),
        "msrv_daily": msrv_daily.tolist(),
        "target_dates": target_dates.tolist() if len(target_dates) > 0 else [],
        "rv_5day_target": rv_5d.tolist(),
        "bv_5day_target": bv_5d.tolist(),
        "tsrv_5day_target": tsrv_5d.tolist(),
        "msrv_5day_target": msrv_5d.tolist(),
        "n_5m_bars": len(bars_5m),
        "n_daily_obs": len(rv_daily),
    }


# Default 5-day horizon
horizon_days = 5


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 288 * 10  # 10 days of 5m bars
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
    
    result = compute_all_estimators(prices)
    print("Test on synthetic 10-day data:")
    print(f"  RV:     {result.rv_annualized:.6f} (annualized)")
    print(f"  BV:     {result.bv_annualized:.6f}")
    print(f"  TSRV:   {result.tsrv_annualized:.6f}")
    print(f"  MSRV:   {result.msrv_annualized:.6f}")
    print(f"  Noise:  {result.noise_variance:.8f}")
    print(f"  Bars:   {result.n_bars}")