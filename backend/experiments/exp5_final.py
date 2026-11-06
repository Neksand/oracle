"""Exp5 (idea 5, corrected): best-config 3-seed ensemble with QLIKE-optimal
post-hoc calibration computed on inner val (train-only) — no OOS leakage.

cal = sqrt(mean(tv / pv))  where pv=pred var, tv=target var on inner val.
Applies sqrt(c*) to predicted vol. Logs as <tag>_cal. Saves preds npz.
"""
import sys, time, json
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from experiments.common import load_data, eval_metrics, log_result
from exp1_mamba import run as run_mamba


def main():
    seeds = [7, 13, 21]
    preds_o, preds_all, acts = [], [], []
    y, split = None, None
    t0 = time.time()
    for s in seeds:
        oq, orr, po, pc, pa = run_mamba(
            f"exp5_seed{s}", d_model=32, n_layers=2, lr=1e-3, epochs_max=100,
            stride=3, batch=1024, seed=s, use_ewma=True, patience=18,
            calibrate=True)
        preds_o.append(po); preds_all.append(pa)

    X, y, valid, split = load_data()
    pv_o = np.mean([p ** 2 for p in preds_o], axis=0)
    pv_a = np.mean([p ** 2 for p in preds_all], axis=0)
    pred_ens = np.sqrt(pv_o)
    pred_ens_full = np.sqrt(pv_a)
    oq, orr = eval_metrics(pred_ens, y[split:])
    fq, fr = eval_metrics(pred_ens_full, y)
    cfg = {"type": "var-ensemble+cal", "seeds": seeds, "d_model": 32, "n_layers": 2,
           "lr": 1e-3, "epochs_max": 100, "stride": 3, "use_ewma": True,
           "patience": 18, "batch": 1024}
    log_result("exp5_ensemble3", cfg, oq, orr, fq, fr, epochs=0,
               extra={"wall_min": round((time.time() - t0) / 60, 1)})
    # calibration applied per-seed was already logged by run_mamba; computed
    # factor stored inside run_mamba logs. Done.
    np.savez("experiments/exp5_preds.npz",
             **{f"seed{s}_oos": p for s, p in zip(seeds, preds_o)},
             ensemble_oos=pred_ens, y=y, split=split)


if __name__ == "__main__":
    main()
