"""Additional scientific tests for the doctoral paper empirical claims.

These tests lock in properties used in the Word paper: QLIKE ranking,
Diebold-Mariano sign, conformal coverage under exchangeability vs serial
dependence, Kupiec LR, and Black-Scholes/VaR engine identities.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.finance_engine import black_scholes_price, simulate_single_asset
from app.research import benchmark as bm
from app.research import vol_models as vm


def _garch_returns(n=2500, omega=1e-6, alpha=0.08, beta=0.90, df=7.0, seed=21):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    v = omega / (1.0 - alpha - beta)
    scale = math.sqrt(df / (df - 2.0))
    for t in range(n):
        z = rng.standard_t(df) / scale
        r[t] = math.sqrt(v) * z
        v = omega + alpha * r[t] ** 2 + beta * v
    return r


def test_qlike_ranks_perfect_above_noisy():
    """Patton QLIKE (log pv + tv/pv - 1) is smaller for a perfect forecast than a noisy one."""
    y = np.linspace(0.08, 0.35, 400)
    noisy = y * 1.35
    assert vm.qlike(y, y) < vm.qlike(noisy, y)


def test_put_call_parity_black_scholes():
    s0, k, r, sig, t = 100.0, 100.0, 0.03, 0.2, 1.0
    call = black_scholes_price(s0, k, r, sig, t, "call")
    put = black_scholes_price(s0, k, r, sig, t, "put")
    parity = call - put - (s0 - k * math.exp(-r * t))
    assert abs(parity) < 1e-8


def test_kupiec_accepts_correct_quantile():
    rng = np.random.default_rng(8)
    x = rng.normal(size=8000)
    # Theoretical N(0,1) 5% quantile, not the in-sample quantile.
    var = stats.norm.ppf(0.05)
    failures = int(np.sum(x < var))
    n = len(x)
    p_exp = 0.05
    phat = failures / n
    lr = -2.0 * (
        (n - failures) * math.log(1 - p_exp) + failures * math.log(p_exp)
        - (n - failures) * math.log(1 - phat) - failures * math.log(phat)
    )
    crit = stats.chi2.ppf(0.95, 1)
    assert lr <= crit
    assert abs(failures - p_exp * n) < 120


def test_ks_rejects_fat_tailed_as_normal():
    r = _garch_returns()
    ks = bm.ks_test(r)
    assert ks["normal"]["p_value"] < 0.05
    assert ks["student_t"]["p_value"] >= ks["normal"]["p_value"]


def test_split_conformal_coverage_iid():
    """Exchangeable residuals: split conformal must be near the nominal level."""
    rng = np.random.default_rng(13)
    n = 4000
    y = rng.normal(size=n)
    f = np.zeros(n)
    scores = np.abs(y - f)
    n_cal = 1000
    cal, test = scores[:n_cal], scores[n_cal:]
    alpha = 0.10
    q = float(np.quantile(cal, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher"))
    cover = float(np.mean(test <= q))
    assert 0.86 <= cover <= 0.94, cover


def test_naive_split_conformal_can_miscover_under_garch():
    """Dependent GARCH residuals break exchangeability; coverage may drift.

    The test only asserts that the procedure still *runs* and that the
    interval is finite. The paper reports the measured coverage rather than
    claiming a guarantee.
    """
    r = _garch_returns(n=3000, seed=4)
    # one-step EWMA as a misspecified mean forecast of |r|
    lam = 0.94
    f = np.empty_like(r)
    v = float(np.var(r[:50]))
    for i, x in enumerate(r):
        f[i] = math.sqrt(v)
        v = lam * v + (1 - lam) * x ** 2
    y = np.abs(r)
    scores = np.abs(y - f)
    n_cal = 800
    cal, test = scores[200:200 + n_cal], scores[200 + n_cal:]
    q = float(np.quantile(cal, 0.90))
    cover = float(np.mean(test <= q))
    assert 0.5 <= cover <= 1.0
    assert np.isfinite(q) and q > 0


def test_engine_var_cvar_finite_and_ordered():
    _, _, _, _, _, stats, _, _ = simulate_single_asset(
        100.0, 0.08, 0.22, 252, num_paths=8000, seed=11)
    assert stats["cvar95_loss"] >= stats["var95_loss"]
    assert stats["var99_loss"] >= stats["var95_loss"]
    assert np.isfinite(stats["p5_ST"])
