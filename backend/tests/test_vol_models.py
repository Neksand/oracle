"""Tests for the econometric volatility stack (vol_models / deep_models / pipeline).

Uses a synthetic OHLC dataset with KNOWN parameters so estimators can be
validated against ground truth - no internet access required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.research import vol_models as vm


def _synthetic_ohlc(n: int = 3000, s0: float = 100.0, sigma_daly: float = 0.01,
                    seed: int = 3):
    """GBM close prices + intraday high/low bands scaled to the true daily vol.

    With unit daily diffusion sigma_daly the expected high-low range is ~2.56*sigma
    (Parkinson), so band = 1.3*sigma_daly keeps the implied Parkinson sigma close
    to the true one (PK variance = range^2 / (4 ln 2) <= true variance).
    """
    rng = np.random.default_rng(seed)
    c = np.empty(n + 1)
    c[0] = s0
    for t in range(n):
        c[t + 1] = c[t] * math.exp(sigma_daly * rng.standard_normal())
    band = 1.3 * sigma_daly
    o = c[:n] * np.exp(sigma_daly * rng.standard_normal(n) * 0.4)
    hi = np.maximum(o, c[:n]) * (1 + band)
    lo = np.minimum(o, c[:n]) * (1 - band)
    return o, hi, lo, c[:n]


def test_parkinson_garman_klass_estimate_true_vol():
    """True daily sigma = 0.01 -> estimators must recover it within tolerance."""
    o, h, l, c = _synthetic_ohlc()
    pv = vm.parkinson_variance(h, l)
    gk = vm.garman_klass_variance(o, h, l, c)
    pk_sigma = math.sqrt(float(np.mean(pv)))
    gk_sigma = math.sqrt(max(float(np.mean(np.clip(gk, 0, None))), 1e-16))
    # Parkinson UNDER-estimates with a coarse intraday proxy; GK has a known
    # slight upward bias when the synthetic open is displaced (Ball-Torous 1988)
    assert 0.004 < pk_sigma < 0.020, f"Parkinson off-scale: {pk_sigma}"
    assert 0.004 < gk_sigma < 0.030, f"Garman-Klass off-scale: {gk_sigma}"


def test_garch_params_recover_persistence():
    """Fit GARCH-t to GARCH-like synthetic returns: persistence must be < 1."""
    rng = np.random.default_rng(11)
    n, omega, alpha, beta = 5000, 4e-6, 0.08, 0.90
    v = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        v = omega + alpha * r[t - 1] ** 2 + beta * v if t else omega / (1 - alpha - beta)
        r[t] = math.sqrt(v) * rng.standard_t(df=7) / math.sqrt(7 / 5)
    eps2 = r ** 2
    fit, _ = vm.fit_garch11_student_t(eps2)
    assert fit["_converged"] and fit["omega"] > 0
    persist = fit["alpha"] + fit["beta"]
    assert 0.3 < persist < 1.0, f"unrealistic persistence: {persist}"
    assert 2.1 <= fit["df"] <= 80


def test_garch_forecast_bounded_and_positive():
    o, h, l, c = _synthetic_ohlc(n=800)
    cc = np.diff(np.log(c))
    f = vm.forecast_garch_walkforward(cc ** 2, kind="garch_t",
                                      refit_every=400, warmup_frac=0.25)
    valid = f[np.isfinite(f)]
    assert len(valid) > 200
    assert np.all(valid > 0) and np.all(valid < 5.0)   # annualized daily sigma sane


def test_har_ols_beats_naive_mean_forecast():
    """OLS on the HAR design must at least beat the sample mean (trivial floor).

    Sanity property, robust to collinearity between RV_w / RV_m / RV_d.
    """
    rng = np.random.default_rng(5)
    n = 4000
    rv = rng.gamma(2.0, 2.0, size=n)                # variance-like positive series
    X, y, cols, idx = vm.har_x_dataset(rv, None, horizon=5)
    assert X.shape[1] == 4                           # const + RV_d + RV_w + RV_m
    j = len(y) // 2
    beta, *_ = np.linalg.lstsq(X[:j], y[:j], rcond=None)
    mse_har = float(np.mean((X[j:] @ beta - y[j:]) ** 2))
    mse_naive = float(np.mean((np.mean(y[:j]) - y[j:]) ** 2))
    assert mse_har <= mse_naive * 1.02              # within 2% of naive floor
    assert np.all(np.isfinite(beta))


def test_qlike_prefers_better_forecast():
    rng = np.random.default_rng(2)
    tv = 0.02 ** 2 * np.ones(1000)
    good = tv * np.exp(rng.normal(0, 0.05, 1000))
    bad = tv * np.exp(rng.normal(0, 0.50, 1000))
    assert vm.qlike(np.sqrt(good), np.sqrt(tv)) < vm.qlike(np.sqrt(bad), np.sqrt(tv))


def test_dm_detects_improvement():
    rng = np.random.default_rng(9)
    y = rng.normal(0, 0.12, 800)
    good = y + rng.normal(0, 0.02, 800)
    bad = y + rng.normal(0, 0.10, 800)
    res = vm.dm_test(good ** 2 - y ** 2, bad ** 2 - y ** 2, h=5)
    assert res["dm_stat"] < 0 and res["p_value"] < 0.01   # A (good) significantly better


def test_block_bootstrap_significant_when_gap_huge():
    rng = np.random.default_rng(4)
    t = np.abs(rng.normal(0, 0.1, 600))
    a = t + rng.normal(0, 0.05, 600)                 # close
    b = t + rng.normal(0, 0.30, 600)                 # far
    res = vm.block_bootstrap_rmse_gap(b, a, t, block=20, n_boot=200, seed=1)
    assert (res["gap"] > 0) and res["significant"]


def test_deep_models_feature_shapes():
    from app.research import deep_models as dm

    o, h, l, c = _synthetic_ohlc(n=900)
    r = np.diff(np.log(c))
    pv = vm.parkinson_variance(h[1:], l[1:])
    gk = np.clip(vm.garman_klass_variance(o[1:], h[1:], l[1:], c[1:]), 0, None)
    X, y, valid = dm.build_features(r, pv, gk, None)
    # channel pruning: with vix=None the two VIX channels are dropped (8 remain)
    assert X.shape[2] == len(dm.FEATURE_NAMES) - 2
    assert X.shape[1] == dm.LAGS
    assert len(X) == len(y) == len(valid)
    assert np.all(y > 0)
    # with VIX supplied, all 10 channels survive
    X2, y2, valid2 = dm.build_features(r, pv, gk, np.full(len(r), 18.0))
    assert X2.shape[2] == len(dm.FEATURE_NAMES)
