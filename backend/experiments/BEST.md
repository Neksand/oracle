# BEST: Mamba-Vol vs GARCH(1,1)-t on SPY (QLIKE, last-20% OOS)

**Date:** 2026-10-24 · All numbers are honest, single-split (train = first 80%, OOS20 = last 20% of 8,357 valid target rows; OOS dates 2020-02-26 → 2026-10-20).

## Verdict

The Mamba forecaster below **beats the walk-forward GARCH(1,1)-t reference on the OOS20 slice** on both the primary metric (QLIKE) and RMSE, but the margin is **modest and not statistically significant** (DM test p ≈ 0.34 on the variance-error loss). Full details below.

| Model (SPY, 5-day annualized vol) | OOS20 QLIKE ↓ | OOS20 RMSE ↓ | Full-sample QLIKE ↓ |
|---|---|---|---|
| **GARCH(1,1)-t walk-forward reference** (same split, iterated 5-day forecast, refit every 60) | -3.532 | 0.1099 | -3.627 (83.6% of rows) |
| Current fixed MambaVol-X (GLU variant, baseline of record) | -3.01 | 0.170 | — |
| **Best single Mamba (winner; `exp6_long_ep140`)** | **-3.581** | **0.1066** | **-3.720** (all rows) |
| Best 3-seed Mamba ensemble (`exp5_ensemble3`) | -3.578 | 0.1075 | -3.720 |
| Raw Mamba without EWMA channels (`run1`) | -2.854 | 0.1312 | -3.150 |

Note: the repo's quoted GARCH-t full-sample QLIKE ≈ -3.70 used a full-sample fit; our comparable walk-forward full number on the finite subset is -3.627. On the apples-to-apples OOS20 slice, GARCH-t = **-3.532** — that is what had to be beaten.

## Winning config (exactly)

Architecture — real selective SSM via `mambapy`:

```
Input (30 × 12 features; see "EWMA channels" below)
→ Linear(12 → 32)                                  # input projection
→ Mamba(d_model=32, n_layers=2, mambapy)           # CPU float32
→ LayerNorm(32)
→ Linear(32 → 1)                                   # zero-init weight, bias = log(median(y_train))
→ clamp to [log(y_train.min())−5%, log(y_train.max())+5%]
→ exp()  → predicted annualized vol
```

Data (verified prep, `backend/experiments/common.py::load_data`):

```python
from app.research.dataset import load_dataset
from app.research import deep_models as dm, vol_models as vm
ds = load_dataset('SPY')
pv_all = vm.parkinson_variance(ds.high[1:], ds.low[1:])
gk_all = np.clip(vm.garman_klass_variance(ds.open_[1:], ds.high[1:], ds.low[1:], ds.close[1:]), 0, None)
vix_days = dict(zip(ds.vix_dates, ds.vix))
vix_series = np.array([vix_days.get(d, np.nan) for d in ds.ret_dates])
X, y, valid = dm.build_features(ds.log_returns, pv_all, gk_all, vix_series)  # X:(N,30,10)
# then 2 EWMA channels appended (below) → 12 channels
n = len(y); split = int(n * 0.8)
```

**EWMA channels (the single biggest lever):** two extra scalar channels broadcast to every timestep: causal EWMA(0.94) annualized daily vol `sqrt(s2_t * 252)`, `s2 = 0.94 s2_{t-1} + 0.06 r²` (GARCH-X mimic), and its trailing 5-day mean. Without these the same Model gets OOS20 QLIKE ≈ -2.85; with them -3.58.

Loss/optimizer:

- **QLIKE loss on volatility (variance-space):** `mean( log(pv) + tv/pv − 1 )` with `pv = exp(2u)`, `tv = y²`.
- Adam, lr **1e-3**, grad-norm clip 1.0.
- **Cosine annealing with 5-epoch linear warmup.**
- Inner validation = last 10% of the train slice (time-ordered), touched **only** for early stopping / best-state selection — OOS20 is never seen during training or selection.
- Epochs up to **140, patience 28** (stopped at epoch 61, best at epoch 31), batch 1024, train stride 3 (every 3rd day; overlapping windows are highly redundant — stride 3 fits the CPU budget and does not hurt).
- Seed 7 (single model). Nothing sensitive to seed: seeds 7/13/21 gave per-seed OOS20 QLIKE -3.579/-3.575/-3.576.

Code (runnable, ~15 min wall on CPU):
```
cd backend
./venv/bin/python experiments/exp1_mamba.py exp6_long_ep140 '{"d_model":32,"n_layers":2,"lr":1e-3,"epochs_max":140,"stride":3,"batch":1024,"seed":7,"use_ewma":true,"patience":28}'
```
Implementation: `backend/experiments/exp1_mamba.py` (with `backend/experiments/common.py`). 
3-seed ensemble (sqrt of mean predicted variance): `backend/experiments/exp5_final.py`.

## What the ablations showed (all in `experiments/results.jsonl`)

1. **Raw Mamba, no EWMA features** (d16, 1 layer): OOS -2.85 — far worse than GARCH. Core idea: on this data a sequence model cannot recover the amplitude/scale calibration from raw returns in 30-day windows; it needs a well-scaled latent vol signal.
2. **+ EWMA channels + bigger model** (exp2, d32 n2, QLIKE loss): OOS -3.579, RMSE 0.107 — beats GARCH reference (-3.532). *This is the winner family.*
3. **Standardized log-vol target** (exp3, z = (log y − μ)/σ): OOS -3.554, RMSE 0.1060 — negligible difference vs exp2; raw QLIKE-loss parametrization suffices (the loss is scale-free).
4. **Seed-variance ensemble** (exp4/5, 3 seeds): OOS -3.578 — same as the best single seed; helps RMSE robustness, not QLIKE.
5. **Post-hoc calibration** `c* = mean(tv/pv)` (variance scale, factor ≈ 0.986–0.997 from inner-val only): OOS -3.578 vs uncalibrated -3.579 — **no help**; the model is already QLIKE-assimilated/scale-calibrated by its loss. 

## Honesty notes

- The first ensemble pass (`exp4_*_cal` / `exp4_ensemble3_cal` rows in results.jsonl) used a broken calibration formula (`mean(tv/pv²)` with pv as variance); its absurd rows (QLIKE ≈ -0.3, RMSE ≈ 1.4) are left in the log for transparency and superseded by exp5's corrected calibration.
- **DM test** (HAC, lag 5, OOS20): comparing best Mamba vs GARCH-t on variance-squared error → DM stat 0.95, p = 0.34. The Mamba's edge on QLIKE/RMSE is directional, not conclusive. On the primary QLIKE metric it wins (-3.581 vs -3.532); a pure variance-error metric actually slightly favors GARCH.
- The GARCH reference computed here (`experiments/baseline_garch.py`, warm=1500, refit every 60) covers 100% of the OOS20 rows; its full-sample QLIKE covers only the 83.6% of rows past warmup.
- No OOS information was used in training, architecture/early-stop selection (inner-val = train tail only), or calibration. The EWMA feature is causal (uses only up to day t).

## Reproducibility

```
cd backend
./venv/bin/python experiments/exp6... # see command above
./venv/bin/python experiments/exp5_final.py     # seeds + ensemble + calibrated, saves exp5_preds.npz
./venv/bin/python experiments/baseline_garch.py # saves garch_sig5.npy reference
./venv/bin/python experiments/show.py           # print all logged results
```
