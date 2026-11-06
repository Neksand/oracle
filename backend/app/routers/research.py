"""ORACLE - Research router: model benchmarking against internet data."""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import datetime as dt
import json
import pathlib

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.research.data import load_history
from app.research import benchmark as bm

router = APIRouter(prefix="/research", tags=["research"])

BENCH_CACHE = pathlib.Path(__file__).resolve().parents[2] / "data" / "cache" / "vol_bench.json"


class BenchmarkRequest(BaseModel):
    symbol: str = Field(default="SPY", max_length=24)
    lookback_days: int = Field(default=750, ge=100, le=1250)
    num_paths: int = Field(default=20000, ge=1000, le=100000)
    horizon_days: int = Field(default=252, ge=30, le=504)
    use_lstm: bool = Field(default=True)


@router.post("/benchmark", summary="Full scientific benchmark: GBM vs Student-t vs Merton vs LSTM")
async def research_benchmark(req: BenchmarkRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        series = load_history(req.symbol, req.lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Data loading failure: {exc}") from exc
    r = series.log_returns
    if len(r) < 80:
        raise HTTPException(status_code=422, detail="Not enough observations after cleaning")

    fits = bm.mle_gbm(r)
    diag = bm.ks_test(r)
    qq = bm.qq_points(r)

    s0 = float(series.closes[-1])
    t_years = req.horizon_days / 252.0

    # Model 1: GBM terminal (closed form sampling)
    g = np.random.default_rng(0)
    st_gbm = s0 * np.exp((fits["mu_ann"] - 0.5 * fits["sigma_ann"] ** 2) * t_years
                         + fits["sigma_ann"] * math.sqrt(t_years) * g.normal(size=req.num_paths))

    # Model 2: Student-t bootstrap (fat tails)
    tp = bm.fit_student_t(r)
    st_t = bm.sample_student_t_terminal(tp, s0, req.horizon_days, req.num_paths, seed=1)

    # Model 3: Merton jump diffusion
    jp = bm.fit_merton_jd(r)
    st_jd = bm.sample_merton_terminal(jp, s0, req.horizon_days, req.num_paths, seed=2)

    def dist_stats(st: np.ndarray, name: str) -> dict[str, Any]:
        lret = np.log(st / s0)
        var95 = float(np.quantile(lret, 0.05))
        return {
            "model": name,
            "expected_ST": round(float(np.mean(st)), 2),
            "median_ST": round(float(np.median(st)), 2),
            "p5_ST": round(float(np.quantile(st, 0.05)), 2),
            "p95_ST": round(float(np.quantile(st, 0.95)), 2),
            "VaR95_return": round(var95, 4),
            "CVaR95_return": round(float(np.mean(lret[lret <= var95])), 4),
            "skew": round(float(np.mean(((lret - np.mean(lret)) / np.std(lret)) ** 3)), 4),
            "excess_kurt": round(float(np.mean(((lret - np.mean(lret)) / np.std(lret)) ** 4) - 3), 4),
            "prob_decline": round(float(np.mean(st < s0)), 4),
        }

    vol_rmse: Dict[str, Any]
    lstm_available = False
    if req.use_lstm:
        try:
            vol_rmse = bm.forecast_rmse_table(r)
            lstm_available = any(row["model"].startswith("LSTM") for row in vol_rmse["rows"])
            vol_rmse["lstm_available"] = lstm_available
            if not lstm_available:
                vol_rmse["note"] = "PyTorch not installed for this Python version; LSTM excluded"
        except Exception as exc:  # torch failure must never kill the endpoint
            vol_rmse = {"rows": [], "winner": None, "error": str(exc)[:200]}
    else:
        vol_rmse = {"rows": [], "winner": None}

    return {
        "data": {
            "symbol": series.symbol,
            "source": series.source,
            "n_obs": int(len(r)),
            "first_date": (series.dates[0] if series.dates else None),
            "last_close": round(s0, 2),
        },
        "calibration": fits,
        "diagnostics": {
            "ks_test": diag,
            "qq_points": qq,
            "kupiec_var95": bm.kupiec_pof(r),
        },
        "terminal_distribution_comparison": [
            dist_stats(st_gbm, "GBM (baseline)"),
            dist_stats(st_t, "Student-t bootstrap"),
            dist_stats(st_jd, "Merton jump diffusion"),
        ],
        "vol_forecast_rmse": vol_rmse,
        "params": {
            "student_t": {"df": round(tp["df"], 2), "drift_ann": round(tp["drift_ann"], 4)},
            "merton_jump": {"lambda": round(jp["lambda"], 4), "jump_size": jp["jump_size"]},
        },
        "execution_time_ms": round((time.perf_counter() - start) * 1000, 1),
        "methodology": [
            "MLE calibration of GBM on daily log-returns",
            "Kolmogorov-Smirnov goodness-of-fit: Normal vs fitted Student-t",
            "QQ-quantile comparison exposes tail underestimation",
            "Kupiec 1995 unconditional-coverage backtest of VaR breaches",
            "Terminal-distribution stress comparison priced at 20k MC paths",
        ],
    }


@router.get("/health", summary="Research subsystem status")
async def research_health():
    torch_ok = False
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception:
        pass
    return {"research": "online", "torch": torch_ok,
            "note": None if torch_ok else "torch unavailable -> LSTM excluded from benchmarks"}


class VolBenchRequest(BaseModel):
    symbol: str = Field(default="SPY", max_length=24)
    refresh: bool = Field(default=False,
                          description="recompute the full walk-forward benchmark (ignores cache)")


@router.post("/vol_benchmark", summary="Deep econometric vol benchmark (HAR/GARCH/GJR/LSTM/VolFormer, walk-forward)")
async def vol_benchmark(req: VolBenchRequest) -> dict:
    """Full-sample model horse race with sample-size sensitivity.

    Cached on disk (backend/data/cache/vol_bench.json) because the full
    run includes ~30 GARCH re-fits and takes ~2 minutes. Pass refresh=true
    to recompute (e.g. after a dataset update).
    """
    from app.research import pipeline as pl

    started = dt.datetime.now(dt.UTC).isoformat()
    if req.refresh or not BENCH_CACHE.exists():
        try:
            heavy = pl.run_full_benchmark(req.symbol, use_transformer=True, recent_days=None)
            trimmed = pl.run_full_benchmark(req.symbol, use_transformer=False, recent_days=2500)
            short = pl.run_full_benchmark(req.symbol, use_transformer=False, recent_days=750)
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail=f"vol benchmark failed: {exc}") from exc
        payload = {
            "started": started,
            "full": heavy,
            "recent_2500": trimmed,
            "recent_750": short,
            "methodology": [
                "Target identical for all models: annualized 5-day close-to-close RV",
                "Strict information sets (bar t known at close of day t; no leakage)",
                "GARCH/GJR-GARCH: student-t MLE, rolling refits on last 2500 obs",
                "HAR-X: expanding OLS on RV_d/RV_w/RV_m + GK/PV/VIX extras",
                "LSTM-X: 30-lag window + HV/PK/GK/VIX features, early stopping",
                "Diebold-Mariano (HAC) tests + circular block bootstrap CI",
            ],
        }
        BENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BENCH_CACHE.write_text(json.dumps(payload, default=str))
    else:
        payload = json.loads(BENCH_CACHE.read_text())
    slim = {k: payload.get(k) for k in ("full", "recent_2500", "recent_750", "started")}
    slim["methodology"] = payload.get("methodology", [])
    slim["schema_note"] = ("blocks: full=8491-day sample, recent_2500=last 10y, "
                           "recent_750=last 3y; each block has rows/dm_pairs/bootstrap/info")
    for block in ("full", "recent_2500", "recent_750"):
        slim[block].pop("y", None)
        slim[block].pop("idx_dates", None)
        slim[block].pop("series", None)      # large arrays live on disk only
    return slim


@router.get("/vol_benchmark/dataset", summary="Dataset provenance for the vol benchmark")
async def vol_benchmark_dataset():
    from app.research.dataset import load_dataset

    ds = load_dataset("SPY")
    return ds.summary()
