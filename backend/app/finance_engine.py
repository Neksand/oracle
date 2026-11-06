"""ORACLE - Financial GBM engine.

Geometric Brownian Motion in price space (not abstract XYZ drift):
    S_T = S0 * exp((mu - 0.5 sigma^2) T + sigma sqrt(T) Z)

Multi-asset via Cholesky on correlation matrix. 3D economic cloud
encoding: X = idiosyncratic shock, Y = log-return level, Z = residual.
Colors: base 60% cyan / stress 25% purple / crisis 15% red.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _year_fraction(t_days: int, trading_days: int) -> float:
    # Calendar-day convention scaled by trading calendar density.
    # t_days=252 with trading_days=252 -> T=1.0y. Crypto 365/365 -> 1.0y.
    return float(t_days) / 252.0 * (252.0 / float(trading_days)) * (float(trading_days) / 252.0) \
        if False else float(t_days) / 252.0 if trading_days == 252 else float(t_days) / 365.0


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(s0: float, k: float, r: float, sigma: float, t: float, kind: str) -> float:
    """Closed-form Black-Scholes for cross-check of Monte-Carlo option price."""
    import math
    if t <= 0 or sigma <= 0 or s0 <= 0 or k <= 0:
        intr = max(s0 - k, 0.0) if kind == "call" else max(k - s0, 0.0)
        return float(intr)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if kind == "call":
        return float(s0 * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2))
    return float(k * math.exp(-r * t) * _norm_cdf(-d2) - s0 * _norm_cdf(-d1))


def _colorize(distances: np.ndarray) -> Tuple[np.ndarray, float, float, float, float, int]:
    p60 = float(np.percentile(distances, 60))
    p85 = float(np.percentile(distances, 85))
    var95 = float(np.percentile(distances, 95))
    var99 = float(np.percentile(distances, 99))
    colors = np.zeros((distances.shape[0], 3), dtype=np.float32)
    m_cyan = distances <= p60
    m_purple = (distances > p60) & (distances <= p85)
    colors[m_cyan] = [0.0, 0.95, 1.0]
    colors[m_purple] = [0.65, 0.12, 0.95]
    colors[~(m_cyan | m_purple)] = [1.0, 0.05, 0.12]
    tail = int(np.sum(distances > p85))
    return colors, p60, p85, var95, var99, tail


def _money_stats(
    st: np.ndarray, s0: float, mu_ann: float, sigma_ann: float,
    t_years: float, risk_free: float, strike: Optional[float],
) -> Dict[str, Any]:
    logret = np.log(st / s0)
    losses = s0 - st  # positive = capital shortfall
    var95 = float(np.percentile(losses, 95))
    cvar95 = float(np.mean(losses[losses >= var95])) if np.any(losses >= var95) else var95
    var99 = float(np.percentile(losses, 99))
    mean_lr, std_lr = float(np.mean(logret)), float(np.std(logret))
    # skew / kurtosis of terminal prices
    m = float(np.mean(st)); sd = float(np.std(st)) or 1e-9
    z = (st - m) / sd
    skew = float(np.mean(z ** 3)); kurt = float(np.mean(z ** 4) - 3.0)
    excess = mu_ann - risk_free
    sharpe = float(excess / sigma_ann) if sigma_ann > 0 else 0.0
    disc = float(np.exp(-risk_free * t_years))
    call_mc = put_mc = call_bs = put_bs = None
    if strike and strike > 0 and t_years > 0 and sigma_ann > 0:
        # Risk-neutral MC via measure change: reuse implied Z from real paths
        denom = sigma_ann * np.sqrt(t_years)
        z_impl = (np.log(np.maximum(st, 1e-9) / s0) - (mu_ann - 0.5 * sigma_ann ** 2) * t_years) / denom
        st_rn = s0 * np.exp((risk_free - 0.5 * sigma_ann ** 2) * t_years + denom * z_impl)
        call_mc = float(np.mean(np.maximum(st_rn - strike, 0.0)) * disc)
        put_mc = float(np.mean(np.maximum(strike - st_rn, 0.0)) * disc)
        call_bs = black_scholes_price(s0, strike, risk_free, sigma_ann, t_years, "call")
        put_bs = black_scholes_price(s0, strike, risk_free, sigma_ann, t_years, "put")
    return {
        "expected_ST": round(float(np.mean(st)), 4),
        "median_ST": round(float(np.median(st)), 4),
        "p5_ST": round(float(np.percentile(st, 5)), 4),
        "p25_ST": round(float(np.percentile(st, 25)), 4),
        "p75_ST": round(float(np.percentile(st, 75)), 4),
        "p95_ST": round(float(np.percentile(st, 95)), 4),
        "mean_logret": round(mean_lr, 5),
        "std_logret": round(std_lr, 5),
        "var95_loss": round(var95, 4),
        "cvar95_loss": round(cvar95, 4),
        "var99_loss": round(var99, 4),
        "prob_decline": round(float(np.mean(st < s0)), 4),
        "prob_loss_gt10": round(float(np.mean(st < 0.9 * s0)), 4),
        "prob_gain_gt10": round(float(np.mean(st > 1.1 * s0)), 4),
        "sharpe_like": round(sharpe, 4),
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "call_price_mc": None if call_mc is None else round(call_mc, 4),
        "put_price_mc": None if put_mc is None else round(put_mc, 4),
        "call_price_bs": None if call_bs is None else round(call_bs, 4),
        "put_price_bs": None if put_bs is None else round(put_bs, 4),
    }


def simulate_single_asset(
    s0: float, mu_ann: float, sigma_ann: float, t_days: int,
    trading_days: int = 252, num_paths: int = 8000,
    num_sample_paths: int = 160, bins: int = 60,
    risk_free: float = 0.04, strike: Optional[float] = None,
    option_type: str = "call", seed: Optional[int] = None,
    symbol: str = "CUSTOM",
) -> Tuple[List[float], List[float], List[List[float]], List[float], Dict, Dict, Dict]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    t_years = _year_fraction(t_days, trading_days)
    dt = t_years / max(t_days, 1)

    # Full path simulation for 2D fan (downsampled time to <=100 cols)
    max_cols = 100
    stride = max(1, t_days // max_cols)
    n_cols = t_days // stride
    if n_cols < 10:
        stride = 1
        n_cols = t_days
    shocks = rng.normal(size=(num_paths, t_days)).astype(np.float64)
    drift_step = (mu_ann - 0.5 * sigma_ann ** 2) * dt
    diff_step = sigma_ann * np.sqrt(dt)
    log_steps = drift_step + diff_step * shocks
    log_cum = np.cumsum(log_steps, axis=1)
    st_all = s0 * np.exp(log_cum[:, -1])
    # sample trajectories
    idx = rng.choice(num_paths, size=min(num_sample_paths, num_paths), replace=False)
    sample_idx = np.arange(0, t_days, stride)[:n_cols]
    sample_log = log_cum[np.ix_(idx, sample_idx)]
    sample_paths = (s0 * np.exp(sample_log)).tolist()
    time_axis = [round(float((i * stride + 1) / t_days * t_days), 1) for i in range(len(sample_idx))]

    # 3D economic cloud: X=shock component, Y=log-return (price level), Z=residual
    logret_t = log_cum[:, -1]
    z1 = shocks[:, -1] if t_days > 0 else rng.normal(size=num_paths)
    z2 = rng.normal(size=num_paths)
    scale_y = 9.0
    y = (logret_t * scale_y).astype(np.float32)
    x = (z1 * (0.6 + sigma_ann) * 3.0).astype(np.float32)
    z = (z2 * (0.6 + sigma_ann) * 3.0).astype(np.float32)
    endpoints = np.stack([x, y, z], axis=1)
    distances = np.linalg.norm(endpoints, axis=1).astype(np.float32)
    colors, p60, p85, var95_r, var99_r, tail = _colorize(distances)

    stats = _money_stats(st_all, s0, mu_ann, sigma_ann, t_years, risk_free, strike)
    counts, edges = np.histogram(st_all, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    # legacy mirror (keeps old StatisticsPanel working if needed)
    legacy = {
        "mean": [0.0, round(float(np.mean(logret_t)), 4), 0.0],
        "std": [round(float(np.std(z1)), 4), round(float(np.std(logret_t)), 4), round(float(np.std(z2)), 4)],
        "percentile_5": [0.0, round(float(np.percentile(logret_t, 5)), 4), 0.0],
        "percentile_95": [0.0, round(float(np.percentile(logret_t, 95)), 4), 0.0],
        "var_95": round(float(var95_r) / 9.0, 4),
        "var_99": round(float(var99_r) / 9.0, 4),
    }
    elapsed = (time.perf_counter() - start) * 1000.0
    metadata = {
        "execution_time": round(elapsed, 2),
        "total_points": int(num_paths),
        "paths": int(num_paths),
        "symbol": symbol, "s0": s0, "mu_ann": mu_ann, "sigma_ann": sigma_ann,
        "t_days": t_days, "t_years": round(t_years, 4), "trading_days": trading_days,
        "tail_particles_count": tail,
        "tail_percentage": round(tail / num_paths * 100, 1),
        "strike": strike, "risk_free": risk_free,
    }
    hist = {"centers": [round(float(c), 4) for c in centers], "counts": [int(c) for c in counts]}
    return endpoints.ravel().tolist(), colors.ravel().tolist(), sample_paths, time_axis, hist, stats, legacy, metadata


def simulate_portfolio(
    s0_list: List[float], mu_list: List[float], sig_list: List[float],
    weights: List[float], corr: List[List[float]],
    t_days: int, trading_days: int = 252, num_paths: int = 8000,
    num_sample_paths: int = 160, bins: int = 60,
    risk_free: float = 0.04, seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
    """Correlated GBM portfolio. Returns (portfolio_ST, per_asset_expected, extra)."""
    rng = np.random.default_rng(seed)
    n = len(s0_list)
    t_years = _year_fraction(t_days, trading_days)
    dt = t_years / max(t_days, 1)
    corr_m = np.array(corr, dtype=np.float64)
    # sanitize correlation matrix
    corr_m = (corr_m + corr_m.T) / 2.0
    np.fill_diagonal(corr_m, 1.0)
    corr_m = np.clip(corr_m, -0.95, 0.95)
    np.fill_diagonal(corr_m, 1.0)
    try:
        chol = np.linalg.cholesky(corr_m)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(corr_m + np.eye(n) * 1e-6)
    w = np.array(weights, dtype=np.float64)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    s0_arr = np.array(s0_list, dtype=np.float64)
    port_st = np.zeros(num_paths)
    per_asset_exp = []
    port_paths_accum = np.zeros((min(num_sample_paths, num_paths), t_days))
    for step_block in range(1):  # single vectorized pass over time
        z_uncorr = rng.normal(size=(num_paths, t_days, n))
        z_corr = z_uncorr @ chol.T
        asset_st = np.zeros((n, num_paths))
        for i in range(n):
            drift = (mu_list[i] - 0.5 * sig_list[i] ** 2) * dt
            diff = sig_list[i] * np.sqrt(dt)
            log_cum = np.cumsum(drift + diff * z_corr[:, :, i], axis=1)
            asset_st[i] = s0_arr[i] * np.exp(log_cum[:, -1])
            per_asset_exp.append(float(np.mean(asset_st[i])))
        port_st = asset_st.T @ w
    # sample portfolio paths for fan chart
    z_uncorr = rng.normal(size=(min(num_sample_paths, num_paths), t_days, n))
    z_corr = z_uncorr @ chol.T
    paths = np.zeros((z_corr.shape[0], t_days))
    for i in range(n):
        drift = (mu_list[i] - 0.5 * sig_list[i] ** 2) * dt
        diff = sig_list[i] * np.sqrt(dt)
        log_cum = np.cumsum(drift + diff * z_corr[:, :, i], axis=1)
        paths += (w[i] * s0_arr[i] * np.exp(log_cum))
    # diversification benefit: weighted VaR vs portfolio VaR
    s0_port = float(s0_arr @ w)
    losses_port = s0_port - port_st
    var_port = float(np.percentile(losses_port, 95))
    indiv_var = 0.0
    for i in range(n):
        li = s0_arr[i] - (s0_arr[i] * np.exp((mu_list[i] - 0.5 * sig_list[i] ** 2) * t_years +
              sig_list[i] * np.sqrt(t_years) * rng.normal(size=num_paths)))
        indiv_var += w[i] * float(np.percentile(li, 95))
    benefit = round(indiv_var - var_port, 4)
    return port_st, per_asset_exp[:n], {"sample_port_paths": paths.tolist(), "benefit": benefit,
                                        "s0_port": s0_port, "t_years": t_years}
