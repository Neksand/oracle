"""Exp4: seed ensemble + honest calibration of the winning exp2 config.

3 seeds, d_model=32, n_layers=2, EWMA channels, QLIKE loss, inner-val best-state.
Ensemble prediction = sqrt(mean_i(pred_i^2)) (variance pooling).
Calibration factor cal = sqrt(mean(tv/pv^2)) uses inner val only (train tail).
"""
import sys, time, json
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from experiments.common import eval_metrics, log_result
import exp1_mamba
from exp1_mamba import run as run_mamba


def main():
    seeds = [7, 13, 21]
    preds_o, preds_cal, preds_all = [], [], []
    y = None
    t0 = time.time()
    for s in seeds:
        oq, orr, po, pc, pa = run_mamba(
            f"exp4_seed{s}", d_model=32, n_layers=2, lr=1e-3, epochs_max=100,
            stride=3, batch=1024, seed=s, use_ewma=True, patience=18,
            calibrate=True)
        preds_o.append(po); preds_cal.append(pc); preds_all.append(pa)
    y = None
    # recompute y via common (cheap: ~1 s)
    from experiments.common import load_data
    X, y, valid, split = load_data()
    pv_ens = np.mean([p ** 2 for p in preds_o], axis=0)
    pred_ens = np.sqrt(pv_ens)
    oq, orr = eval_metrics(pred_ens, y[split:])
    pv_all = np.mean([p ** 2 for p in preds_all], axis=0)
    pred_all = np.sqrt(pv_all)
    fq, fr = eval_metrics(pred_all, y)
    cfg = {"type": "var-ensemble", "seeds": seeds, "d_model": 32, "n_layers": 2,
           "lr": 1e-3, "epochs_max": 100, "stride": 3, "use_ewma": True,
           "patience": 18, "batch": 1024}
    log_result("exp4_ensemble3", cfg, oq, orr, fq, fr, epochs=0,
               extra={"wall_min": round((time.time() - t0) / 60, 1)})
    # calibrated ensemble
    pv_cal = np.mean([p ** 2 for p in preds_cal if p is not None], axis=0)
    pred_c = np.sqrt(pv_cal)
    oq2, orr2 = eval_metrics(pred_c, y[split:])
    pv_all_cal = np.mean([p ** 2 for p in preds_all], axis=0)
    fq2, fr2 = eval_metrics(pred_all, y)
    log_result("exp4_ensemble3_cal", {**cfg, "cal": "inner-val seed-mean"},
               oq2, orr2, fq2, fr2, epochs=0)
    np.savez("experiments/exp4_preds.npz",
             **{f"seed{s}": p for s, p in zip(seeds, preds_o)},
             **{f"seed{s}_cal": p for s, p in zip(seeds, preds_cal)},
             ensemble=pred_ens, ensemble_full=pred_all, y=y, split=split)


if __name__ == "__main__":
    main()
