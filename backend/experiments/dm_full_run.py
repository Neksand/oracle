"""DM tests + block bootstrap for the NEW full-sample benchmark (2026-10-24).

Reproduces the winning deep forecasts (LSTM-X best-params, MambaVol-X SSM winner
recipe) and the GARCH(1,1)-t walk-forward reference on the identical alignment,
computes HAC Diebold-Mariano tests on the QLIKE loss differential and circular
block-bootstrap CIs of the QLIKE gap. Writes results to
experiments/dm_full_results.json and refreshes data/cache/vol_bench.json
('full' rows) so paper/build_paper.py reads updated honest numbers.
"""
import sys, json, math
import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from app.research.dataset import load_dataset
from app.research import deep_models as dm, vol_models as vm
from app.research.pipeline import garch_5day_forecasts
from experiments.common import load_data, log_result

OUT = "experiments/dm_full_results.json"


def qlike_losses(pred_vol, y_true):
    pv = np.asarray(pred_vol, float) ** 2
    tv = np.asarray(y_true, float) ** 2
    ok = np.isfinite(pv) & np.isfinite(tv) & (pv > 1e-12) & (tv > 0)
    out = np.full(len(pv), np.nan)
    out[ok] = np.log(pv[ok]) + tv[ok] / pv[ok] - 1.0
    return out


def block_bootstrap_gap(loss_a, loss_b, n_boot=2000, block=20, seed=7):
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
        idx = idx[idx < n]
        means[b] = d[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi), float(means.mean())


def dm_qlike_loss(loss_a, loss_b, h=5):
    """Diebold-Mariano test on QLIKE loss differentials (A - B), HAC lag h.

    Negative stat => model A has lower QLIKE loss (A better).
    """
    a = np.asarray(loss_a, float)
    b = np.asarray(loss_b, float)
    d = a - b
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": int(n),
                "lag": 0, "better": "?"}
    dbar = float(d.mean())
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
            "n": int(n), "lag": int(L), "better": "A" if stat < 0 else "B"}


def main():
    X, y, valid, split_dummy = load_data()
    n = len(y)

    # Deep models (identical configs to the new full benchmark, seed 42)
    lstm_out = dm.train_forecaster_with_best_params(
        X, y, {"hidden_size": 64, "num_layers": 1, "lr": 0.0016575242940205508,
               "dropout": 0.2, "head_hidden": 24}, seed=42)
    lstm_preds = np.asarray(lstm_out["preds"])
    mamba_out = dm.train_forecaster(X, y, architecture="mamba", seed=42)
    mamba_preds = np.asarray(mamba_out["preds"])

    # GARCH(1,1)-t walk-forward on the same return series and day alignment
    ds = load_dataset('SPY')
    r = ds.log_returns
    # valid[i] is a raw-day index; need forecasts for every raw day, then map
    f, info = garch_5day_forecasts(r, "garch_t")
    garch_pred_vol = np.array([f[t] for t in valid], dtype=float)
    tgt = np.array([np.asarray(r[t + 1: t + 1 + 5]).std(ddof=1) * math.sqrt(252) for t in valid])

    results = {"n_rows": int(n)}
    for name, pv_pred in (("garch_t", garch_pred_vol), ("lstm_x", lstm_preds), ("mamba_ssm", mamba_preds)):
        ql = vm.qlike(pv_pred, tgt)
        rm = float(np.sqrt(np.mean((np.asarray(pv_pred, float) - tgt) ** 2)))
        results[name] = {"qlike": round(ql, 4), "rmse": round(rm, 4)}

    loss_g = qlike_losses(garch_pred_vol, tgt)
    loss_m = qlike_losses(mamba_preds, tgt)
    loss_l = qlike_losses(lstm_preds, tgt)

    for a_name, pa, pb_name, pb in (("mamba_ssm", mamba_preds, "garch_t", garch_pred_vol),
                                    ("lstm_x", lstm_preds, "garch_t", garch_pred_vol),
                                    ("mamba_ssm", mamba_preds, "lstm_x", lstm_preds)):
        ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(tgt)
        la, lb = qlike_losses(pa, tgt), qlike_losses(pb, tgt)
        okq = ok & np.isfinite(la) & np.isfinite(lb)
        # QLIKE-loss Diebold-Mariano (the metric used for the ranking claim)
        dres = dm_qlike_loss(la[okq], lb[okq], h=5)
        # Block bootstrap of the QLIKE gap: d = loss_b - loss_a (gap > 0 => a better)
        lo, hi, mean_gap = block_bootstrap_gap(lb[okq], la[okq])
        results[f"dm_{a_name}_vs_{pb_name}"] = {
            "dm_stat_qlike": dres["dm_stat"],
            "p_value_qlike": dres["p_value"],
            "qlike_gap_mean_b_minus_a": round(mean_gap, 5),
            "boot_ci_low_b_minus_a": round(lo, 5), "boot_ci_high_b_minus_a": round(hi, 5),
            "n_used": int(dres["n"]),
        }

    # Persist aligned forecasts so the DM/bootstrap can be reproduced without retraining.
    np.savez(OUT.replace(".json", "_preds.npz"),
             garch_t=garch_pred_vol, lstm_x=lstm_preds, mamba_ssm=mamba_preds,
             target=tgt, valid=np.asarray(valid, int))

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1)
    log_result("dm_full_run", {"note": "loss-based DM on identical alignment"},
               results["mamba_ssm"]["qlike"], results["mamba_ssm"]["rmse"],
               results["garch_t"]["qlike"], results["garch_t"]["rmse"], 0)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
