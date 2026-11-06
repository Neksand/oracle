"""Shared data prep + eval for Mamba volatility experiments (SPY).

Verified data prep per task spec. train/OOS split: first 80% / last 20%.
"""
import sys, math, json, time
import numpy as np

sys.path.insert(0, '.')

from app.research.dataset import load_dataset
from app.research import deep_models as dm, vol_models as vm


def load_data():
    ds = load_dataset('SPY')
    pv_all = vm.parkinson_variance(ds.high[1:], ds.low[1:])
    gk_all = np.clip(vm.garman_klass_variance(ds.open_[1:], ds.high[1:], ds.low[1:], ds.close[1:]), 0, None)
    vix_days = dict(zip(ds.vix_dates, ds.vix))
    vix_series = np.array([vix_days.get(d, np.nan) for d in ds.ret_dates])
    X, y, valid = dm.build_features(ds.log_returns, pv_all, gk_all, vix_series)
    n = len(y)
    split = int(n * 0.8)
    return X, y, valid, split


def eval_metrics(pred_vol, y_true):
    q = vm.qlike(np.asarray(pred_vol, float), np.asarray(y_true, float))
    r = float(np.sqrt(np.mean((np.asarray(pred_vol, float) - np.asarray(y_true, float)) ** 2)))
    return q, r


def add_ewma_channels(X, ds_log_returns, valid_idx):
    """Append 2 extra channels: causal EWMA(0.94) annualized daily vol + its 5-day avg.

    X is (N, LAGS, F) for valid rows of the raw feature builder; ds_log_returns
    is the raw returns series (len n_raw). EWMA at day t uses data up to day t.
    Channel value at position (row, lag, :) is the same *time-of-row* signal so
    all lags carry it (mimics GARCH-X conditioning); to keep sequence shape we
    put the current EWMA vol into every timestep (scalar channel broadcast).
    """
    r = np.asarray(ds_log_returns, float)
    n = len(r)
    sig2 = np.zeros(n)
    s2 = r[0] ** 2 if np.isfinite(r[0]) else 0.0
    for t in range(n):
        s2 = 0.94 * s2 + 0.06 * (r[t] ** 2 if np.isfinite(r[t]) else 0.0)
        sig2[t] = s2
    ew_vol = np.sqrt(np.maximum(sig2, 0.0) * 252.0)  # annualized daily EWMA vol
    ew5 = np.array([ew_vol[max(0, i - 4): i + 1].mean() for i in range(n)])
    # map valid feature-row t-index -> raw day index (lag 0 row of sample t is day t)
    t_idx = np.asarray(valid_idx, int)
    c1 = ew_vol[t_idx][:, None, None] * np.ones((1, X.shape[1], 1), dtype=np.float32)
    c2 = ew5[t_idx][:, None, None] * np.ones((1, X.shape[1], 1), dtype=np.float32)
    return np.concatenate([X, c1, c2], axis=2)


def log_result(tag, config, oos_q, oos_r, full_q, full_r, epochs, extra=None):
    rec = {"tag": tag, "config": config, "oos20_qlike": oos_q, "oos20_rmse": oos_r,
           "full_qlike": full_q, "full_rmse": full_r, "epochs": epochs}
    if extra:
        rec.update(extra)
    with open("experiments/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("LOGGED:", json.dumps(rec))
