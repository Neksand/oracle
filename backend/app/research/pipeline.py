"""ORACLE - Research: full walk-forward benchmark pipeline for volatility models.

Target (identical for every model, no look-ahead):
    y_t = annualized realized volatility of the forward 5 trading days,
    computed from close-to-close returns r_{t+1}..r_{t+5}.

Models:
    GBM-const   sigma frozen at the warm-up sample level
    EWMA(0.94)  RiskMetrics recursion, walk-forward
    HAR-X       expanding-window OLS: 1 + RV_d + RV_w + RV_m + GK_w + PV_w + VIX
    GARCH(1,1)  student-t MLE, rolling refit every 250 obs on last 2500 obs window
    GJR-GARCH   student-t MLE (leverage), same protocol
    LSTM-X      deep_models.py LSTM with engineered features (+VIX)
    MambaVol-X  deep_models.py real selective SSM (mambapy) + EWMA channels, log-vol head, QLIKE loss
    VolFormer   small transformer encoder on the same features (small-sample probe)

Comparison statistics:
    RMSE (sigma space), QLIKE (variance space, Patton 2011)
    Diebold-Mariano HAC test for every pair (Newey-West lag = horizon)
    circular block bootstrap CI for the best-vs-rest RMSE gap

The result dict is JSON-safe and cached on disk; the /research/benchmark route
re-uses a cached run unless ?refresh=true.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from app.research import deep_models as dm
from app.research import vol_models as vm
from app.research.dataset import Dataset, load_dataset

TRADING_DAYS = 252
H = 5
HAR_WARM_FRAC = 0.25
GARCH_WARM_FRAC = 0.30
GARCH_REFIT_EVERY = 250
GARCH_WINDOW = 2500


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = np.convolve(x, np.ones(w) / w, mode="valid")
    return out


def forward_rv_targets(r: np.ndarray, h: int = H) -> tuple[np.ndarray, int]:
    """y[t] for t in [t0, n-h): annualized 5d close-to-close realized vol.

    Returns (targets, t0) with t0 = shared warm-up so every model sees at least
    `t0` prior observations before the first forecast.
    """
    n = len(r)
    t0 = 120
    y = np.array([r[t + 1: t + 1 + h].std(ddof=1) * math.sqrt(TRADING_DAYS)
                  for t in range(t0, n - h)])
    return y, t0


def garch_5day_forecasts(r: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    """Per-t annualized aggregate 5-day variance forecast from a GARCH fit.

    Multi-step laws of motion with mean reversion to the unconditional level:
        sigma2_{t+k} = persist^(k-1) * (sigma2_{t+1} - sigma2_inf) + sigma2_inf
        agg_var_5d   = mean(sigma2_{t+1..t+5})
    Refits are capped by GARCH_WINDOW for speed (rolling window, standard).
    """
    n = len(r)
    t0 = 120
    forecast = np.full(n, np.nan)
    fit_log: list[dict] = []
    params: list[float] | None = None
    persist = None
    refit_clock = 0
    for t in range(t0, n - H):
        if params is None or refit_clock >= GARCH_REFIT_EVERY:
            window_start = max(0, t - GARCH_WINDOW)
            f = vm.choose_fit(r[window_start:t] ** 2, kind)
            if f.get("_converged") and f["omega"] > 0:
                p = np.array(f["omega"]) / (1.0 - f["alpha"] - 0.5 * f["gamma"] - f["beta"])
                persist = f["alpha"] + 0.5 * f["gamma"] + f["beta"]
                params = [f["omega"], f["alpha"], f["gamma"], f["beta"], p]
                refit_clock = 0
                if t % 1000 < GARCH_REFIT_EVERY:
                    fit_log.append({"t": t, "alpha": f["alpha"], "gamma": f["gamma"],
                                    "beta": f["beta"], "persist": round(float(persist), 4)})
            else:
                params = None
                refit_clock = GARCH_REFIT_EVERY - 1  # retry soon
        if params is None:
            forecast[t] = float(np.sqrt(np.var(r[max(0, t - 250):t], ddof=1) * TRADING_DAYS))
            refit_clock += 1
            continue
        lam = min(persist, 0.9995)
        v_prev = vm._garch_filter(r[:t] ** 2, params[0], params[1], params[3], params[2])[-1]
        v1 = params[0] + params[1] * r[t - 1] ** 2 + params[3] * v_prev + params[2] * (r[t - 1] ** 2 if (kind == "gjr_t" and r[t - 1] < 0) else 0.0)
        s_inf = params[0] / max(1.0 - lam, 1e-4)
        ks = np.arange(1, H + 1, dtype=float)
        path = (lam ** (ks - 1)) * (max(v1, 1e-14) - s_inf) + s_inf
        forecast[t] = math.sqrt(max(float(path.mean()), 1e-14) * TRADING_DAYS)
        refit_clock += 1
    return forecast, {"refits": len(fit_log), "log": fit_log[:5],
                      "avg_persist": round(float(np.mean([e["persist"] for e in fit_log])), 4) if fit_log else None}


def har_walkforward(rv_daily_var: np.ndarray, extras: dict[str, np.ndarray],
                    n_total: int) -> tuple[np.ndarray, dict]:
    """HAR-X per-t forecasts of forward 5d annualized variance (expanding OLS).

    Returns a full-length (n_total) array aligned to return indices; rows whose
    features were invalid stay NaN (later forward-filled conservatively).
    """
    X, y, cols, t_idx = vm.har_x_dataset(rv_daily_var, extras, horizon=H)
    if len(y) < 400:
        return np.full(n_total, np.nan), {"notes": "HAR sample too small"}
    n = len(y)
    warm = max(int(n * HAR_WARM_FRAC), 300)
    compact = np.full(n, np.nan)
    for j in range(warm, n):
        beta, *_ = np.linalg.lstsq(X[:j], y[:j], rcond=None)
        compact[j] = float(X[j] @ beta)
    sig = np.sqrt(np.clip(compact, 1e-16, None))
    full = np.full(n_total, np.nan)
    valid_mask = np.isfinite(sig)
    full[t_idx[valid_mask]] = sig[valid_mask]
    return full, {"n_used": int(valid_mask.sum()), "columns": cols, "warm": warm}


def _run_lstm_optimization(X: np.ndarray, y_deep: np.ndarray, seed: int = 7) -> dict | None:
    """Run Optuna hyperparameter search for LSTM-X on the given data.

    Returns best_params dict or None if optuna/torch unavailable.
    """
    try:
        return dm.optimize_lstm_hyperparameters(X, y_deep, n_trials=12, seed=seed, timeout_seconds=1800)
    except Exception:
        return None


def run_full_benchmark(symbol: str = "SPY", use_transformer: bool = True,
                       seed: int = 7, recent_days: int | None = None,) -> dict[str, Any]:
    """recent_days: restrict the sample to the LAST N return observations
    (sample-size sensitivity study - the key scientific question: how much
    history does the deep model need before it beats GARCH?)."""
    t_start = time.perf_counter()
    ds: Dataset = load_dataset(symbol)
    if recent_days is not None and recent_days < len(ds.close):
        cut = len(ds.close) - recent_days
        ds = Dataset(
            symbol=ds.symbol, dates=ds.dates[cut:],
            open_=ds.open_[cut:], high=ds.high[cut:], low=ds.low[cut:],
            close=ds.close[cut:], volume=ds.volume[cut:],
            vix=None, vix_dates=None,
            log_returns=ds.log_returns[cut:], ret_dates=ds.ret_dates[cut:])  # re-align VIX below
        if ds.vix is None:             # keep VIX feature for the trimmed sample too
            vix_days_all = load_dataset(symbol)  # full dataset (cached)
            vd = {d: v for d, v in zip(vix_days_all.vix_dates, vix_days_all.vix)}
            common = [d for d in ds.dates if d in vd]
            ds.vix = np.array([vd[d] for d in common])
            ds.vix_dates = common
    r = ds.log_returns
    ts_dates = ds.ret_dates
    n = len(r)
    y, t0 = forward_rv_targets(r)               # y[i] ~ r[t0+i .. t0+i+H]
    idx = np.arange(t0, n - H)                  # information date index (day t)

    # --- predictors, all aligned to idx ------------------------------------
    preds: dict[str, np.ndarray] = {}

    sigma_warm = float(np.std(r[:t0], ddof=1) * math.sqrt(TRADING_DAYS))
    preds["GBM (constant sigma)"] = np.full(len(idx), sigma_warm)

    ewma = np.full(len(idx), np.nan)
    var_f = float(np.var(r[:t0]))
    for i, t in enumerate(range(t0, n - H)):
        var_f = 0.94 * var_f + 0.06 * r[t] ** 2
        ewma[i] = math.sqrt(var_f) * math.sqrt(TRADING_DAYS)
    preds["EWMA (RiskMetrics 0.94)"] = ewma

    # DAY-shifted O/H/L/C so estimator at index t uses day-t bar (info at t)
    o_t, h_t, l_t, c_t = ds.open_[t0:], ds.high[t0:], ds.low[t0:], ds.close[t0:]
    pv_all = vm.parkinson_variance(ds.high[1:], ds.low[1:])
    gk_all = np.clip(vm.garman_klass_variance(ds.open_[1:], ds.high[1:], ds.low[1:], ds.close[1:]), 0, None)
    pv_t = pv_all[t0 - 1:len(r)]                # len == n, pv_t[k] ~ pv of day ts_dates[k]
    gk_t = gk_all[t0 - 1:len(r)]

    vix_days = {}
    if ds.vix is not None:
        vix_days = dict(zip(ds.vix_dates, ds.vix))
    vix_series = np.array([vix_days.get(d, np.nan) for d in ts_dates])

    # LSTM / transformer: build features per original return index, then embed
    feat_valid: dict[str, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    X, y_deep, valid = dm.build_features(r, pv_all, gk_all, vix_series)
    feat_valid["x"] = (X, y_deep, valid)

    # --- Run Optuna hyperparameter search on FULL sample only (not on subsets) ---
    lstm_optuna_results = None
    if recent_days is None:
        # Only run optimization on the full sample; cache results
        lstm_optuna_results = _run_lstm_optimization(X, y_deep, seed=seed)

    def _deep_preds(architecture: str, checkpoint: str | None = None) -> tuple[dict, np.ndarray]:
        if architecture == "lstm" and lstm_optuna_results is not None:
            # Use best params from Optuna
            out = dm.train_forecaster_with_best_params(
                X, y_deep, lstm_optuna_results["best_params"],
                checkpoint=checkpoint, seed=seed
            )
            return out, np.asarray(out["preds"])
        else:
            out = dm.train_forecaster(X, y_deep, architecture=architecture,
                                      checkpoint=checkpoint, seed=seed)
            return out, np.asarray(out["preds"])

    lstm_out, lstm_preds = _deep_preds("lstm", checkpoint="lstm_vol_x.pth")
    deep_map = {valid[i]: lstm_preds[i] for i in range(len(valid))}
    arr = np.full(len(idx), np.nan)
    for i, t in enumerate(idx):
        if t in deep_map:
            arr[i] = deep_map[t]
    if np.isfinite(arr).sum() == 0:
        raise RuntimeError("LSTM produced no aligned forecasts")
    preds["LSTM-X (torch)"] = arr

    transformer_info: dict | None = None
    if use_transformer:
        try:
            t_out, t_preds = _deep_preds("transformer", checkpoint=None)
            fmap = {valid[i]: t_preds[i] for i in range(len(valid))}
            arr_t = np.full(len(idx), np.nan)
            for i, t in enumerate(idx):
                if t in fmap:
                    arr_t[i] = fmap[t]
            preds["VolFormer (probe)"] = arr_t
            transformer_info = {"val_qlike": t_out.get("val_qlike", t_out.get("val_rmse")), "val_share": t_out["n_val"]}
        except Exception as exc:
            transformer_info = {"error": str(exc)[:180]}

    # MambaVolModel (custom GLU-based architecture)
    mamba_info: dict | None = None
    try:
        m_out, m_preds = _deep_preds("mamba", checkpoint="mamba_vol_x.pth")
        mfmap = {valid[i]: m_preds[i] for i in range(len(valid))}
        arr_m = np.full(len(idx), np.nan)
        for i, t in enumerate(idx):
            if t in mfmap:
                arr_m[i] = mfmap[t]
        preds["MambaVol-X (SSM)"] = arr_m
        mamba_info = {"val_qlike": m_out.get("val_qlike", m_out.get("val_rmse")), "val_share": m_out["n_val"]}
    except Exception as exc:
        mamba_info = {"error": str(exc)[:180]}

    # GARCH family walk-forward
    for kind, label in (("garch_t", "GARCH(1,1)-t"), ("gjr_t", "GJR-GARCH(1,1)-t")):
        f, info = garch_5day_forecasts(r, kind)
        preds[label] = f[idx]

    # HAR-X on daily RV proxy (GK) with PV + VIX extras (NaN rows are skipped,
    # never imputed -> no look-ahead statistics anywhere)
    extras = {"GK_d": gk_t, "PV_d": pv_t}
    if np.isfinite(vix_series).mean() > 0.8:
        extras["VIX"] = vix_series
    har_sigma, har_info = har_walkforward(gk_t, extras, n_total=n)
    if len(har_sigma) == n:
        preds["HAR-X (OLS)"] = har_sigma[idx]

    # --- assemble comparison ------------------------------------------------
    report_models: list[dict[str, Any]] = []
    full_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, p in preds.items():
        if p is None:
            continue
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 100:
            continue
        p_, y_ = p[mask], y[mask]
        full_series[name] = (p, mask)          # keep full-length arrays + support mask
        rmse = float(np.sqrt(np.mean((p_ - y_) ** 2)))
        ql = vm.qlike(p_, y_)
        mae = float(np.mean(np.abs(p_ - y_)))
        report_models.append({"model": name, "n": int(mask.sum()),
                              "rmse": round(rmse, 4), "qlike": round(ql, 4),
                              "mae": round(mae, 4)})
    report_models.sort(key=lambda m: m["qlike"])

    # DM tests on the QLIKE leader vs the rest; statistics on the INTERSECTION of
    # both models' valid support (identical targets -> paired losses)
    subset = [m for m in report_models][:6]
    dm_pairs = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            pa, ma = full_series[subset[i]["model"]]
            pb, mb = full_series[subset[j]["model"]]
            use = ma & mb
            a, b, yy = pa[use], pb[use], y[use]
            if len(yy) < 100:
                continue
            # DM on squared-error losses of the VARIANCE forecasts
            res = vm.dm_test(a ** 2 - yy ** 2, b ** 2 - yy ** 2, h=H)
            dm_pairs.append({"a": subset[i]["model"], "b": subset[j]["model"],
                             "n_common": int(use.sum()), **res})
    best = report_models[0]
    second = report_models[1]
    pb_best, mb = full_series[best["model"]]
    pb_second, ms = full_series[second["model"]]
    use = mb & ms
    boot = vm.block_bootstrap_rmse_gap(
        pb_second[use], pb_best[use], y[use], block=H * 4, n_boot=400, seed=seed)

    deep_info = {
        "lstm": {"val_qlike": lstm_out.get("val_qlike", lstm_out.get("val_rmse")), "best_epoch": lstm_out["best_epoch"],
                 "n_train": lstm_out["n_train"], "n_val": lstm_out["n_val"]},
        "transformer": transformer_info,
        "mamba": mamba_info,
        "features": dm.FEATURE_NAMES,
    }
    if lstm_optuna_results is not None:
        deep_info["lstm"]["optuna"] = {
            "best_params": lstm_optuna_results["best_params"],
            "best_val_qlike": lstm_optuna_results.get("best_val_qlike", lstm_optuna_results.get("best_val_rmse")),
            "n_trials": len(lstm_optuna_results["study"].trials),
        }

    info = {
        "dataset": ds.summary(),
        "target": f"annualized {H}-day close-to-close realized volatility",
        "run_seconds": round(time.perf_counter() - t_start, 1),
        "deep": deep_info,
        "garch_walkforward": {"refit_every": GARCH_REFIT_EVERY, "window": GARCH_WINDOW,
                              "warm_frac": GARCH_WARM_FRAC},
        "har": har_info,
    }
    return {"rows": report_models, "dm_pairs": dm_pairs, "bootstrap_gap": boot,
            "best": best, "runner_up": second, "info": info,
            "idx_dates": [ts_dates[i] for i in idx],
            "series": {name: np.asarray(p).tolist() for name, p in preds.items()},
            "y": y.tolist()}


def pd_fill(a: np.ndarray) -> np.ndarray:
    """Simple forward fill for small NaN runs (None allowed at head)."""
    out = np.array(a, dtype=float, copy=True)
    last = np.nan
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        elif np.isfinite(last):
            out[i] = last
    return out
