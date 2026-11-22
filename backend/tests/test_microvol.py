"""MicroVol fallback contract: a torch-less run must not claim the 100-param model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.research import microvol as mv
from app.research.data import PriceSeries


def _series(symbol: str, n: int = 400) -> PriceSeries:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    dates = [d.date().isoformat() for d in pd.bdate_range("2022-01-03", periods=n)]
    return PriceSeries(symbol, closes, "synthetic (test)", dates)


@pytest.mark.parametrize(
    "symbol,annualization",
    [("BTC-USD", 365), ("SPY", 252)],
)
def test_fallback_reports_honest_model(monkeypatch, symbol, annualization):
    monkeypatch.setattr(mv, "_TORCH_OK", False)
    monkeypatch.setattr(mv, "load_history", lambda sym, days: _series(sym))

    out = mv.run_microvol(symbol, 400)

    assert out["fallback"] is True
    assert out["torch_available"] is False
    assert out["params"] == mv.M3_PARAMS != 100
    assert out["params_breakdown"]["sequence"] == 0
    assert "fallback" in out["model"]
    assert out["annualization"] == annualization
    assert out["forecast"]["vol_annualized"] == pytest.approx(
        out["forecast"]["vol_daily"] * np.sqrt(annualization), rel=1e-9
    )
    assert out["qlike_gap"] == pytest.approx(out["qlike_model"] - out["qlike_har"], abs=2e-6)
