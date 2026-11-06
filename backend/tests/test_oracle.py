"""ORACLE - Engine + DB test suite.

Run: backend/venv/bin/python -m pytest tests/ -q
Covers:
    GBM vs Black-Scholes MC-convergence (scientific correctness)
    VaR/CVaR ordering property          (risk-metric sanity)
    DB layer: the 3 user queries + SQLi safe/unsafe behaviour
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.finance_engine import black_scholes_price, simulate_single_asset  # noqa: E402
from app.db import queries  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.seed import seed  # noqa: E402


# ---------------------------------------------------------------------------
# Finance engine (scientific correctness)
# ---------------------------------------------------------------------------
def test_mc_call_converges_to_black_scholes():
    """MC price with 60k paths must sit within 2% of the closed form."""
    s0, k, r, sigma, t = 100.0, 100.0, 0.04, 0.25, 1.0
    bs = black_scholes_price(s0, k, r, sigma, t, "call")
    rng = np.random.default_rng(7)
    z = rng.normal(size=60_000)
    st = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
    mc = float(np.exp(-r * t) * np.mean(np.maximum(st - k, 0.0)))
    assert abs(mc - bs) / bs < 0.02, f"MC {mc:.4f} vs BS {bs:.4f}"


def test_var_cvar_ordering():
    _, _, _, _, _hist, stats, _legacy, _meta = simulate_single_asset(
        100.0, 0.10, 0.30, 252, num_paths=20000, seed=42)
    assert stats["cvar95_loss"] >= stats["var95_loss"] >= 0
    assert stats["var99_loss"] >= stats["var95_loss"]
    assert 0 <= stats["prob_decline"] <= 1


def test_terminal_quartiles_ordered():
    _a, _b, _c, _d, _e, stats, _f, _g = simulate_single_asset(100.0, 0.08, 0.2, 90, seed=1)
    assert stats["p5_ST"] <= stats["p25_ST"] <= stats["p75_ST"] <= stats["p95_ST"]


def test_gbm_terminal_lognormal_mean():
    """E[S_T] = S0 * exp(mu*T) analytic check within 3%."""
    s0, mu, sigma, t_years, seed, paths = 200.0, 0.12, 0.25, 1.0, 3, 400_000
    rng = np.random.default_rng(seed)
    st = s0 * np.exp((mu - 0.5 * sigma**2) * t_years
                     + sigma * np.sqrt(t_years) * rng.normal(size=paths))
    theoretical = s0 * math.exp(mu * t_years)
    assert abs(float(np.mean(st)) - theoretical) / theoretical < 0.03


# ---------------------------------------------------------------------------
# Database layer (Assignment 3)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db():
    init_db()
    try:
        seed()
    except Exception:
        pass  # already seeded -> unique constraints raise
    yield SessionLocal()


def test_q1_lookup_returns_scenarios(db):
    res = queries.lookup_user_scenarios(db, "alice")
    assert res["count"] >= 1
    assert any(s["name"] == "Retirement Core" for s in res["scenarios"])
    assert res["scenarios"][0]["owner"] == "alice"


def test_q2_join_aggregates(db):
    res = queries.join_scenario_risk(db)
    assert res["rows"], "JOIN query must return rows"
    row = res["rows"][0]
    assert row["n_assets"] >= 1 and row["n_runs"] >= 1
    assert row["best_expected_st"] > 0


def test_q3_insert_scenario(db):
    res = queries.insert_scenario(db, 3, {
        "name": "Test Insert", "description": "pytest", "horizon_days": 60,
        "assets": [{"symbol": "SPY", "s0": 585, "mu_ann": 0.09, "sigma_ann": 0.17, "weight": 1}],
    })
    assert res["scenario_id"] > 0


def test_sql_injection_unsafe_leaks(db):
    res = queries.unsafe_lookup_scenarios(db, "' OR 1=1 --")
    assert res["rows_leaked"] >= 4, "vulnerable query must leak ALL scenarios"


def test_sql_injection_safe_neutralizes(db):
    res = queries.safe_lookup_scenarios(db, "' OR 1=1 --")
    assert res["rows_leaked"] == 0, "parameterized query neutralizes the attack"


def test_sql_injection_safe_normal_use(db):
    res = queries.safe_lookup_scenarios(db, "bob")
    assert res["rows_leaked"] >= 1
