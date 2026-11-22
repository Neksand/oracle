"""MicroVol-100: 100-parameter CPU volatility model (close-only demo variant).

Architecture (adapted from analysis/v34_micro/tsr.py + ladder.py):
    m3 backbone : 19-parameter calendar-HAR OLS
                  [1, lrv, lrv5, lrv22, lvov, ldown5, ret5]
                  + 6 weekday dummies + 6 weekday x lrv interactions
    residual    : 2-unit GRU over the last 30 days x 10 features,
                  zero-initialised linear head (81 parameters by the TSR
                  accounting 3*(d*h + h^2 + h) + h + 1, 100 total).

Prediction target is next-day close-to-close variance r_{t+1}^2.  QLIKE is
computed in variance space.  Split is strictly chronological 70/10/20 and the
feature scaler is fit on the train slice only.

All inputs are built from daily close-to-close log returns, so this is a
deliberately reduced demo: no OHLC range, Parkinson, Garman-Klass or VIX
inputs.  Absolute QLIKE is therefore not comparable to the full HAR-X
benchmark in app/research/pipeline.py.

If torch is missing (or training raises) the run degrades to the 19-parameter
m3 calendar-HAR backbone and reports fallback=True, params=19 and a fallback
model name -- it must never claim the 100-parameter GRU model.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.research.data import load_history

SEQ_LEN = 30
HIDDEN = 2
SEEDS = (0, 1, 2)
N_FEATURES = 10
M3_PARAMS = 19
VAR_FLOOR = 1e-12
VOV_FLOOR = 1e-10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10

try:  # torch is optional; the endpoint must still work without it
    import torch
    from torch import nn

    _TORCH_OK = True
except Exception:  # pragma: no cover - exercised only on torch-less installs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_OK = False


def _qlike(y: np.ndarray, p: np.ndarray) -> float:
    """Patton QLIKE in variance space (lower is better)."""
    e = np.clip(y, 1e-12, None) / np.clip(p, 1e-12, None)
    return float(np.mean(e - np.log(e) - 1.0))


def _roll(x: np.ndarray, w: int, how: str, min_periods: int = 1) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    r = getattr(s.rolling(w, min_periods=min_periods), how)()
    return r.to_numpy(dtype=float)


def _weekdays(symbol: str, n: int, dates: list[str] | None) -> np.ndarray:
    """Weekday index per close, using real dates when available.

    load_history() returns no dates for the yfinance path, so fall back to a
    business-day calendar for equities and a calendar-day one for crypto.
    """
    if dates and len(dates) == n:
        return np.asarray(pd.to_datetime(dates).day_of_week)
    sym = symbol.upper()
    crypto = any(tag in sym for tag in ("BTC", "ETH", "SOL", "USDT", "-USD"))
    end = pd.Timestamp.today().normalize()
    rng = pd.date_range(end=end, periods=n, freq="D" if crypto else "B")
    return np.asarray(rng.day_of_week)


def _is_crypto(symbol: str) -> bool:
    sym = symbol.upper()
    return any(tag in sym for tag in ("BTC", "ETH", "SOL", "USDT", "-USD"))


def _next_weekday(d: int, crypto: bool) -> int:
    if crypto:
        return int((d + 1) % 7)
    return 0 if d >= 4 else int(d + 1)  # Fri/Sat/Sun -> Monday


def _dummies(dow: np.ndarray) -> np.ndarray:
    return np.column_stack([(np.asarray(dow) == k).astype(float) for k in range(1, 7)])


def _design(r: np.ndarray, dow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features (10) and m3 base columns (7) from daily log returns."""
    var = r * r
    lrv = np.log(np.clip(var, VAR_FLOOR, None))
    rv5 = _roll(var, 5, "mean")
    rv22 = _roll(var, 22, "mean")
    vov = np.nan_to_num(_roll(lrv, 22, "std", min_periods=2), nan=0.0)
    down5 = _roll(np.minimum(r, 0.0) ** 2, 5, "mean")
    ret5 = _roll(r, 5, "sum")
    ret22 = _roll(r, 22, "sum")

    lrv5 = np.log(np.clip(rv5, VAR_FLOOR, None))
    lrv22 = np.log(np.clip(rv22, VAR_FLOOR, None))
    lvov = np.log(np.clip(vov, VOV_FLOOR, None))
    ldown = np.log(np.clip(down5, VAR_FLOOR, None))

    ang = 2.0 * math.pi * dow / 7.0
    x = np.column_stack(
        [r, lrv, lrv5, lrv22, lvov, ldown, ret5, ret22, np.sin(ang), np.cos(ang)]
    )
    base = np.column_stack([np.ones(len(r)), lrv, lrv5, lrv22, lvov, ldown, ret5])
    return x, base


if _TORCH_OK:

    class _TSR(nn.Module):
        """2-unit GRU with a zero-initialised residual head."""

        def __init__(self, d: int, h: int):
            super().__init__()
            self.gru = nn.GRU(d, h, batch_first=True)
            self.head = nn.Linear(h, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, x):
            return self.head(self.gru(x)[0][:, -1]).squeeze(-1)


def _train_gru(
    x_tr: np.ndarray,
    b_tr: np.ndarray,
    x_va: np.ndarray,
    b_va: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    epochs: int = 300,
    patience: int = 10,
):
    """Train one GRU residual, early-stopped on validation QLIKE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    model = _TSR(x_tr.shape[-1], HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    xt = torch.as_tensor(x_tr, dtype=torch.float32)
    bt = torch.as_tensor(b_tr, dtype=torch.float32)
    xv = torch.as_tensor(x_va, dtype=torch.float32)
    bv = torch.as_tensor(b_va, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    yv = torch.as_tensor(y_va, dtype=torch.float32)

    best = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pv = torch.exp(torch.clamp(bt + model(xt), -20.0, 5.0))
        tv = torch.exp(torch.clamp(yt, -20.0, 5.0))
        loss = (torch.log(pv) + tv / pv - 1.0).mean()
        loss.backward()
        opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                pv = torch.exp(torch.clamp(bv + model(xv), -20.0, 5.0))
                tv = torch.exp(torch.clamp(yv, -20.0, 5.0))
                v = (torch.log(pv) + tv / pv - 1.0).mean().item()
            if v < best - 1e-6:
                best = v
                best_state = {k: val.clone() for k, val in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _gru_residual(model, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.as_tensor(x, dtype=torch.float32))
        return (torch.as_tensor(b, dtype=torch.float32) + out).numpy().astype(float)


def run_microvol(symbol: str, lookback_days: int = 750) -> Dict[str, Any]:
    """Fit m3 + GRU residual and return test QLIKE plus a next-day forecast."""
    started = time.perf_counter()
    series = load_history(symbol, lookback_days)
    closes = np.asarray(series.closes, dtype=float)
    if len(closes) < 120:
        raise ValueError("Not enough closes for MicroVol (need >= 120)")
    r = np.diff(np.log(closes))
    m = len(r)
    start = SEQ_LEN - 1
    if m <= start + 30:
        raise ValueError("Not enough returns for a 30-day sequence split")

    dow = _weekdays(series.symbol, len(closes), series.dates)
    dow_r = dow[:-1]  # information day (close i)
    crypto = _is_crypto(series.symbol)

    x_raw, base = _design(r, dow_r)
    if not np.isfinite(x_raw[start:]).all():
        raise ValueError("Non-finite features after warm-up")

    # Target weekday: actual next close for training rows, next session for
    # the live forecast row.
    dow_next = np.empty(m)
    dow_next[:-1] = dow[1:-1]
    dow_next[-1] = _next_weekday(int(dow[-1]), crypto)
    dum = _dummies(dow_next)
    xpair = np.column_stack([base, dum, dum * base[:, 1:2]])  # 19 columns
    xhar = np.column_stack([base[:, :4], dum])  # 10 columns

    var = r * r
    z = np.full(m, np.nan)  # log variance of the next day
    z[:-1] = np.log(np.clip(var[1:], VAR_FLOOR, None))

    idx = np.arange(start, m - 1)  # rows with an observed next-day target
    n = len(idx)
    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC * n)
    tr, va, te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]
    if len(tr) < 40 or len(va) < 5 or len(te) < 10:
        raise ValueError("Not enough rows after the chronological split")

    # Train-only scaler.
    mu = x_raw[tr].mean(axis=0)
    sd = x_raw[tr].std(axis=0) + 1e-8
    zfeat = (x_raw - mu) / sd

    def seq(rows: np.ndarray) -> np.ndarray:
        return np.stack([zfeat[i - SEQ_LEN + 1 : i + 1] for i in rows]).astype(
            np.float32
        )

    x_tr, x_va, x_te = seq(tr), seq(va), seq(te)
    x_next = seq(np.array([m - 1]))  # forecast row: through the last close

    # --- m3 calendar-HAR backbone (OLS on train only) ---------------------
    beta_m3, *_ = np.linalg.lstsq(xpair[tr], z[tr], rcond=None)
    logvar_m3 = np.clip(xpair @ beta_m3, -20.0, 5.0)

    # --- HAR-X baseline (HAR + weekday dummies, OLS on train only) --------
    beta_har, *_ = np.linalg.lstsq(xhar[tr], z[tr], rcond=None)
    logvar_har = np.clip(xhar @ beta_har, -20.0, 5.0)

    truth_te = var[te + 1]
    qlike_har = _qlike(truth_te, np.exp(logvar_har[te]))
    qlike_m3 = _qlike(truth_te, np.exp(logvar_m3[te]))

    # --- GRU residual, 3 seeds, averaged in variance space -----------------
    train_seconds = 0.0
    torch_error = None
    var_preds = []
    next_preds = []
    if _TORCH_OK:
        try:
            for seed in SEEDS:
                t0 = time.perf_counter()
                model = _train_gru(x_tr, logvar_m3[tr], x_va, logvar_m3[va], z[tr], z[va], seed)
                train_seconds += time.perf_counter() - t0
                te_pred = np.exp(np.clip(_gru_residual(model, x_te, logvar_m3[te]), -20.0, 5.0))
                fwd_pred = np.exp(
                    np.clip(_gru_residual(model, x_next, logvar_m3[[m - 1]]), -20.0, 5.0)
                )
                var_preds.append(te_pred)
                next_preds.append(float(fwd_pred[0]))
        except Exception as exc:  # fall back to the m3 backbone
            torch_error = f"{type(exc).__name__}: {exc}"[:200]
            var_preds, next_preds = [], []
    else:
        torch_error = "torch unavailable -> zero (flat) residual fallback"

    if var_preds:
        var_te = np.mean(var_preds, axis=0)
        next_var = float(np.mean(next_preds))
    else:
        # Zero-initialised tanh residual degenerates to a flat zero.
        var_te = np.exp(logvar_m3[te])
        next_var = float(np.exp(logvar_m3[m - 1]))

    qlike_model = _qlike(truth_te, var_te)
    next_var = float(np.clip(next_var, 1e-12, 1e6))
    next_vol = math.sqrt(next_var)

    # 24/7 crypto trades every calendar day; equities only weekdays. The paper
    # annualises crypto with 365, so expose the exact factor used.
    annualization = 365 if crypto else 252
    fallback = not var_preds

    if fallback:
        # Only the linear m3 calendar-HAR coefficients exist; no sequence
        # parameters were trained, so never advertise the 100-param model card.
        model_name = "m3 calendar-HAR (fallback)"
        params: Any = M3_PARAMS
        params_breakdown = {"m3_backbone": M3_PARAMS, "sequence": 0}
        note = (
            "Fallback run: the sequence residual was not trained "
            f"({torch_error}). Metrics are the 19-parameter m3 calendar-HAR "
            "backbone only, on a close-only demo target."
        )
    else:
        model_name = "MicroVol-100 (m3 calendar-HAR + 2-unit GRU residual)"
        params = 100
        params_breakdown = {"m3_backbone": M3_PARAMS, "gru_recurrent": 78, "residual_head": 3}
        note = (
            "Close-only demo variant: features are daily close-to-close returns "
            "only (no OHLC range, Parkinson, Garman-Klass or VIX), so absolute "
            "QLIKE is not comparable to the full HAR-X benchmark."
        )

    return {
        "symbol": series.symbol,
        "source": series.source,
        "n_obs": int(len(r)),
        "lookback_days": int(lookback_days),
        "model": model_name,
        "params": params,
        "params_breakdown": params_breakdown,
        "fallback": fallback,
        "hidden_units": HIDDEN,
        "sequence_length": SEQ_LEN,
        "seeds": list(SEEDS),
        "split": {
            "scheme": "70/10/20 chronological (demo split != paper split)",
            "train": int(len(tr)),
            "validation": int(len(va)),
            "test": int(len(te)),
        },
        "qlike_model": round(qlike_model, 6),
        "qlike_har": round(qlike_har, 6),
        "qlike_m3": round(qlike_m3, 6),
        # Paper convention: gap = QLIKE(model) - QLIKE(HAR); negative = better.
        "qlike_gap": round(qlike_model - qlike_har, 6),
        "train_seconds": round(train_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 2),
        "torch_available": bool(_TORCH_OK and not torch_error),
        "torch_error": torch_error,
        "annualization": annualization,
        "forecast": {
            "horizon_days": 1,
            "variance_daily": next_var,
            "vol_daily": next_vol,
            "variance_annualized": next_var * annualization,
            "vol_annualized": next_vol * math.sqrt(annualization),
            "last_close": float(closes[-1]),
        },
        "note": note,
    }


if __name__ == "__main__":
    out = run_microvol("SPY", 750)
    assert math.isfinite(out["qlike_model"]), out["qlike_model"]
    assert math.isfinite(out["qlike_har"]), out["qlike_har"]
    assert math.isfinite(out["train_seconds"])
    assert math.isfinite(out["forecast"]["variance_daily"])
    assert out["forecast"]["variance_daily"] > 0
    assert math.isfinite(out["forecast"]["vol_annualized"])
    assert out["annualization"] == 252
    assert out["qlike_gap"] == round(out["qlike_model"] - out["qlike_har"], 6)
    if out["fallback"]:
        assert out["params"] == M3_PARAMS
        assert "fallback" in out["model"]
    else:
        assert out["params"] == 100
    print(
        f"SPY qlike_model={out['qlike_model']:.6f} qlike_har={out['qlike_har']:.6f} "
        f"gap={out['qlike_gap']:+.6f} params={out['params']} fallback={out['fallback']} "
        f"train={out['train_seconds']:.3f}s "
        f"vol_ann={out['forecast']['vol_annualized']:.4f} torch={out['torch_available']}"
    )
