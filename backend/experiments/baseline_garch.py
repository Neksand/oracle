"""Reference GARCH(1,1)-t 5-day-ahead forecast on the same split (honest baseline).

sigma5[t] = sqrt(mean(v_{t+1..t+5}) * 252), recursive iterated GARCH(1,1)-t
forecast (E[eps2 subsequent days] = filtered variance), expanding refit every 60.
Aligned to y[t] = stdev(ret_{t+1..t+5}) * sqrt(252) (the dataset target).
"""
import sys, math, time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from experiments.common import load_data, eval_metrics, log_result
from app.research import vol_models as vm
from app.research.dataset import load_dataset


def main():
    ds = load_dataset('SPY')
    r = np.asarray(ds.log_returns, float)
    r = np.clip(r, -0.2, 0.2)
    X, y, valid, split = load_data()
    n = len(r)
    t0 = time.time()
    # expanding refit every 60, one-step daily GARCH-t forecasts, then iterate
    warm = 1500
    refit_every = 60
    params = None
    sig5 = np.full(n, np.nan)
    for t in range(warm, n):
        if params is None or (t - warm) % refit_every == 0:
            f = vm.choose_fit(r[:t] ** 2, "garch_t")
            p = [f["omega"], f["alpha"], f["gamma"], f["beta"]]
            if not (np.isfinite(p[0]) and p[0] > 0):
                params = None
            else:
                params = p
        hist = r[:t] ** 2
        if params is not None:
            v = vm._garch_filter(hist, params[0], params[1], params[3], params[2])
            v_next = max(v[-1], 1e-12)
            o, a, g, b = params
            acc = v_next
            vars_ = [v_next]
            v_cur = v_next
            for h in range(4):
                v_cur = o + (a + g / 2.0) * v_cur + b * v_cur
                vars_.append(v_cur)
            sig5[t] = math.sqrt(np.mean(vars_) * 252.0)
        else:
            sig5[t] = math.sqrt(hist[-min(22, t):].mean() * 252.0)
        if t % 500 == 0:
            print(t, round(time.time() - t0, 1), "s", flush=True)
    # align to y rows (valid row t)
    pred = sig5[np.asarray(valid, int)]
    ok = np.isfinite(pred)
    print("finite frac:", ok.mean())
    oq, orr = eval_metrics(pred[split:], y[split:])
    fq, fr = eval_metrics(pred, y)
    log_result("garch_t_5day_ref", {"refit_every": refit_every, "kind": "garch_t"},
               oq, orr, fq, fr, epochs=0)
    np.save("experiments/garch_sig5.npy", sig5)


if __name__ == "__main__":
    main()
