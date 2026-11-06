"""ORACLE - Research: econometric volatility models + scientific model comparison.

Everything here is dependency-light (numpy/scipy only) so the whole research
stack runs locally inside the backend with no additional services.

Contents
--------
Range-based realized-volatility estimators (input: daily OHLC):
    parkinson(1980), garman_klass(1980), rogers_satchell(1991) -> daily sigma
    weekly / monthly averages -> HAR-RV regressors (Corsi 2009)

GARCH family via constrained MLE (scipy.optimize.minimize, L-BFGS-B):
    garch11_normal, garch11_t (Bollerslev 1986 with Student-t innovations),
    gjr_garch (Glosten-Jagannathan-Runkle 1993 leverage asymmetry)
    -- fitted once per window, filtered forward (rolling refits every 60d)

HAR-X: OLS regression of the forward 5-day realized variance on
    [1, RV_d, RV_w, RV_m, PV_gk_5, VIX] -- refit each step (OLS is closed form)

Model comparison (Hansen & Lunde 2005 spirit):
    qlike(pred_var, target_var)  - robust loss for variance forecasts
    diebold_mariano(pred_a, pred_b, target) - HAC (Newey-West) corrected test
    paired_errors_ci             - circular block bootstrap of RMSE gap
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

TRADING_DAYS = 252
LOG_2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# Range-based variance estimators (annualization factor applied outside)
# ---------------------------------------------------------------------------
def parkinson_variance(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) high-low range variance, daily, in squared-return units."""
    hl = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
    return hl ** 2 / (4.0 * math.log(2.0))


def garman_klass_variance(open_: np.ndarray, high: np.ndarray,
                          low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Garman-Klass (1980) OHLC estimator, daily, squared-return units."""
    o, h, l, c = (np.log(np.asarray(x, dtype=float)) for x in (open_, high, low, close))
    half = 0.5 * (h - l) ** 2
    corr = (2.0 * math.log(2.0) - 1.0) * (c - o) ** 2
    return half - corr


def rogers_satchell_variance(open_: np.ndarray, high: np.ndarray,
                             low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Rogers-Satchell (1991) drift-independent OHLC estimator."""
    o, h, l, c = (np.log(np.asarray(x, dtype=float)) for x in (open_, high, low, close))
    return (h - o) * (h - c) + (l - o) * (l - c)


def window_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = np.convolve(x, np.ones(w) / w, mode="valid")
    return out


# ---------------------------------------------------------------------------
# GARCH(1,1) & GJR-GARCH by constrained MLE
# ---------------------------------------------------------------------------
def _garch_filter(eps2: np.ndarray, omega: float, alpha: float, beta: float,
                  gamma: float = 0.0) -> np.ndarray:
    n = len(eps2)
    v = np.empty(n)
    v[0] = omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-6)
    for t in range(1, n):
        v[t] = omega + (alpha + gamma * float(eps2[t - 1] > 0.0)) * eps2[t - 1] + beta * v[t - 1]
    return v


class GarchResult(dict):
    @property
    def ok(self) -> bool:
        return bool(self.get("_converged"))


def _fit_garch_core(eps2: np.ndarray, student_t: bool, asym: bool, starts: list[list[float]]):
    """Scaled multi-start Nelder-Mead MLE of the GARCH likelihood.

    Scaling eps2 by its mean puts all params O(1) (conditioning); df is bounded
    inside the likelihood (rejection) because Nelder-Mead ignores box bounds.
    """
    n = len(eps2)
    scale = float(np.mean(eps2)) or 1e-8
    u = eps2 / scale

    def unpack(p):
        if student_t:
            omega_u, alpha, gamma, beta, log_df = list(p)
        else:
            omega_u, alpha, gamma, beta = list(p)
            log_df = 0.0
        gamma = float(gamma) if asym else 0.0
        return float(omega_u), float(alpha), float(gamma), float(beta), float(np.exp(float(log_df)))

    def nll(p):
        omega_u, alpha, gamma, beta, df = unpack(p)
        if omega_u <= 1e-9 or alpha < 0 or beta < 0 or gamma < 0 or \
           alpha + 0.5 * gamma + beta >= 0.999:
            return 1e9
        if student_t and not (2.1 <= df <= 80.0):
            return 1e9
        v = _garch_filter(u, omega_u, alpha, beta, gamma)
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            return 1e9
        z2 = u / v
        if not student_t:
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(v) + z2)
        else:
            c = math.lgamma(df / 2.0 + 0.5) - math.lgamma(df / 2.0) - 0.5 * math.log(math.pi * (df - 2.0))
            ll = n * c - 0.5 * np.sum((df + 1.0) * np.log1p(z2 / (df - 2.0)) + np.log(v))
        return float(-ll) if np.isfinite(ll) else 1e9

    best = None
    for s in starts:
        try:
            res = optimize.minimize(nll, np.array(s, dtype=float), method="Nelder-Mead",
                                    options={"maxiter": 6000, "maxfev": 8000,
                                             "xatol": 1e-7, "fatol": 1e-7})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return GarchResult({"_converged": False, "loglik": float("-inf")}), np.full(n, np.nan)

    omega_u, alpha, gamma, beta, df = unpack(best.x)
    v = _garch_filter(u, omega_u, alpha, beta, gamma)
    out = {"omega": float(omega_u * scale), "alpha": round(alpha, 4), "gamma": round(gamma, 4),
           "beta": round(beta, 4), "df": round(df, 2) if student_t else None,
           "loglik_ups": float(-best.fun), "_converged": bool(best.success)}
    return GarchResult(out), v * scale


_STARTS_BASE = [
    [0.5, 0.05, 0.0, 0.75],   # (omega_u, alpha, gamma, beta)
    [0.2, 0.10, 0.0, 0.85],
    [0.5, 0.15, 0.0, 0.80],
    [0.2, 0.30, 0.0, 0.65],
]
_STARTS_JR = [
    [0.5, 0.04, 0.06, 0.75],
    [0.2, 0.08, 0.12, 0.80],
    [0.3, 0.10, 0.05, 0.85],
]
_STARTS_T = [
    [0.5, 0.05, 0.0, 0.75, math.log(7.0)],
    [0.2, 0.10, 0.0, 0.85, math.log(8.0)],
    [0.5, 0.15, 0.0, 0.80, math.log(6.0)],
]
_STARTS_JR_T = [
    [0.5, 0.04, 0.06, 0.75, math.log(7.0)],
    [0.2, 0.08, 0.12, 0.80, math.log(6.0)],
    [0.3, 0.10, 0.05, 0.85, math.log(8.0)],
]


def fit_garch11_normal(eps2: np.ndarray) -> tuple[GarchResult, np.ndarray]:
    return _fit_garch_core(eps2, False, False, _STARTS_BASE)


def fit_garch11_student_t(eps2: np.ndarray) -> tuple[GarchResult, np.ndarray]:
    return _fit_garch_core(eps2, True, False, _STARTS_T)


def fit_gjr_garch_student_t(eps2: np.ndarray) -> tuple[GarchResult, np.ndarray]:
    return _fit_garch_core(eps2, True, True, _STARTS_JR_T)


def choose_fit(eps2: np.ndarray, kind: str):
    """Public fit dispatcher used by the walk-forward loop."""
    if kind == "garch_t":
        r, _ = fit_garch11_student_t(eps2)
    elif kind == "gjr_t":
        r, _ = fit_gjr_garch_student_t(eps2)
    else:
        r, _ = fit_garch11_normal(eps2)
    return r


def forecast_garch_walkforward(eps2: np.ndarray, kind: str = "gjr_t",
                               refit_every: int = 60,
                               warmup_frac: float = 0.25) -> np.ndarray:
    """Public API: sigma_{t,t+1} forecasts (daily, close-to-close)."""
    n = len(eps2)
    warm = max(int(n * warmup_frac), 500)
    out = np.full(n, np.nan)
    params: list[float] | None = None
    for t in range(warm, n):
        if params is None or (t - warm) % refit_every == 0:
            f = choose_fit(eps2[:t], kind)
            cand = [f["omega"], f["alpha"], f["gamma"], f["beta"]]
            if not np.isfinite(cand[0]) or cand[0] <= 0:
                out[t] = math.sqrt(float(eps2[:t].mean()))
                continue
            params = cand
        v_prev = _garch_filter(eps2[:t], params[0], params[1], params[3], params[2])
        out[t] = math.sqrt(max(v_prev[-1], 1e-12)) * math.sqrt(TRADING_DAYS)
    return out


# ---------------------------------------------------------------------------
# HAR-X: forward h-day RV regressed on RV_d / RV_w / RV_m + PV & VIX extras
# ---------------------------------------------------------------------------
def har_x_dataset(rv_daily: np.ndarray, extra: dict[str, np.ndarray] | None,
                  horizon: int = 5) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Columns: [1, RV_d, RV_w(5), RV_m(22), (extra columns...)].

    ``rv_daily``: daily realized-variance PROXY (squared or GK units); row t uses
    RV of day t inclusive (backward window - same info set as GARCH/EWMA).
    Target: annualized realized variance of days t+1..t+h (aggregate RV).
    """
    rv_daily = np.asarray(rv_daily, dtype=float)
    n = len(rv_daily)
    rw = window_mean(rv_daily, 5)
    rm = window_mean(rv_daily, 22)
    rows_X, rows_y, idx = [], [], []
    cols = ["const", "RV_d", "RV_w", "RV_m"]
    extras: list[tuple[str, np.ndarray | None]] = list((extra or {}).items())
    cols += [k for k, arr in extras if arr is not None]
    for t in range(25, n - horizon):
        y = np.nanmean(rv_daily[t + 1: t + 1 + horizon])
        if not np.isfinite(y):
            continue
        xs = [1.0, rv_daily[t], rw[t], rm[t]]
        if any(not np.isfinite(v) for v in xs):
            continue
        uses_extra_vals: list[float] = []
        skip = False
        for name, arr in extras:
            if arr is None:
                continue
            v = float(arr[t]) if t < len(arr) else np.nan
            if not np.isfinite(v):
                skip = True
                break
            uses_extra_vals.append(v)
        if skip:
            continue
        xs += uses_extra_vals
        rows_X.append(xs)
        rows_y.append(y)
        idx.append(t)
    return (np.array(rows_X, dtype=float),
            np.array(rows_y, dtype=float) * TRADING_DAYS,   # annualized variance target
            cols, np.array(idx, dtype=int))


def har_x_walkforward(rv_daily: np.ndarray, extra: dict[str, np.ndarray] | None,
                      horizon: int = 5, warmup_frac: float = 0.25) -> np.ndarray:
    """Expanding-window OLS refit each step, predict at t for t+1..t+h."""
    X, y, cols, t_idx = har_x_dataset(rv_daily, extra, horizon)
    n = len(y)
    warm = max(int(n * warmup_frac), 200)
    out = np.full(n, np.nan)
    for j in range(warm, n):
        beta, *_ = np.linalg.lstsq(X[:j], y[:j], rcond=None)
        out[j] = float(X[j] @ beta)
    return out, y, cols


# ---------------------------------------------------------------------------
# Robust loss & significance testing for forecast comparison
# ---------------------------------------------------------------------------
def qlike(pred_sigma_ann: np.ndarray, target_sigma_ann: np.ndarray) -> float:
    """QLIKE on variances (Patton 2011): mean of  log(pv) + tv/pv - 1 ... down-scaled."""
    pv = (np.asarray(pred_sigma_ann, dtype=float)) ** 2
    tv = (np.asarray(target_sigma_ann, dtype=float)) ** 2
    ok = np.isfinite(pv) & np.isfinite(tv) & (pv > 1e-12) & (tv > 0)
    if not ok.any():
        return float("inf")
    return float(np.mean(np.log(pv[ok]) + tv[ok] / pv[ok] - 1.0))


def dm_test(err_a: np.ndarray, err_b: np.ndarray, h: int = 5) -> dict[str, float]:
    """Diebold-Mariano (1995) with Newey-West HAC variance of the loss diff.

    Loss = squared error of the *variance* forecast (QLIKE-style squared loss
    on variance keeps unit consistency with the RMSE table).
    H0: equal accuracy. Negative stat = model A better.
    """
    a = np.asarray(err_a, float) ** 2
    b = np.asarray(err_b, float) ** 2
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = float(d.mean())
    # Newey-West with Bartlett kernel, lag = h
    L = max(1, min(h, n // 2))
    g0 = float(np.dot(d - dbar, d - dbar)) / n
    gw = g0
    for l in range(1, L + 1):
        cov_l = float(np.dot((d - dbar)[: n - l], (d - dbar)[l:])) / n
        gw += 2.0 * (1.0 - l / (L + 1.0)) * cov_l
    var = max(gw / n, 1e-18)
    stat = dbar / math.sqrt(var)
    p = float(2.0 * stats.norm.sf(abs(stat)))
    return {"dm_stat": round(stat, 4), "p_value": round(p, 4),
            "n": n, "lag": L,
            "better": "A" if stat < 0 else "B"}


def block_bootstrap_rmse_gap(pred_a: np.ndarray, pred_b: np.ndarray,
                             target: np.ndarray, block: int = 20,
                             n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Circular block bootstrap CI for RMSE(A) - RMSE(B)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(pred_a, float)
    b = np.asarray(pred_b, float)
    t = np.asarray(target, float)
    n = min(len(a), len(b), len(t))
    a, b, t = a[:n], b[:n], t[:n]
    def rmse(p):
        return float(np.sqrt(np.mean((p - t) ** 2)))
    gap = rmse(a) - rmse(b)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([
            (j + np.arange(block)) % n
            for j in rng.integers(0, n, size=max(1, n // block))
        ])[:n]
        ea = float(np.sqrt(np.mean((a[idx] - t[idx]) ** 2)))
        eb = float(np.sqrt(np.mean((b[idx] - t[idx]) ** 2)))
        boots.append(ea - eb)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"gap": round(gap, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0)}
