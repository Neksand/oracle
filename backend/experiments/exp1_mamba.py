"""Exp1: real Mamba (mambapy) encoder -> LayerNorm -> linear log-vol head -> exp with train clamps.

QLIKE loss on vol^2. Adam, cosine + 5-ep warmup, early stop on inner val (train tail).
Train uses stride sampling to fit CPU budget (overlapping windows are redundant).
"""
import sys, time, math, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from experiments.common import load_data, eval_metrics, log_result
from mambapy.mamba import Mamba, MambaConfig


def run(tag, d_model=16, n_layers=1, lr=5e-4, epochs_max=60, stride=2, batch=1024,
        seed=7, warmup=5, patience=12, use_ewma=False, clamp_margin=0.05,
        calibrate=False, ensembler=None):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, valid, split = load_data()
    if use_ewma:
        from app.research.dataset import load_dataset
        ds = load_dataset('SPY')
        X = add_ewma_local(X, valid, ds.log_returns)
    Xte = torch.from_numpy(X).float()
    yte = torch.from_numpy(y).float()
    n = len(y)
    iv = int(split * 0.9)  # inner val = last 10% of train (time-ordered)

    tr_idx = np.arange(0, iv, stride)
    va_idx = np.arange(iv, split)
    lo = math.log(max(y[:split].min(), 1e-4)); hi = math.log(y[:split].max())
    m = (hi - lo) * clamp_margin
    clamp_lo, clamp_hi = lo - m, hi + m
    median_log = float(np.log(np.median(y[:iv])))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(X.shape[2], d_model)
            self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.fill_(median_log)

        def forward(self, xb):
            z = self.mamba(self.proj(xb))
            z = self.ln(z)
            u = self.head(z[:, -1, :]).squeeze(-1)
            u = torch.clamp(u, clamp_lo, clamp_hi)
            return u

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def qlike_var_loss(u, yt):
        pv = torch.exp(2.0 * u)
        tv = yt ** 2
        return (torch.log(pv) + tv / pv - 1.0).mean()

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1.0, (e + 1) / warmup) if e < warmup else
        0.5 * (1 + math.cos(math.pi * (e - warmup) / max(1, epochs_max - warmup))))

    # initial step with lr=0 for scheduler smoothness
    full_epoch = 0
    best_q, best_state, best_ep = float('inf'), None, -1
    t0 = time.time()
    epochs_run = 0
    hist = []
    for ep in range(epochs_max):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xte[idx], yte[idx]
            opt.zero_grad()
            u = model(xb)
            loss = qlike_var_loss(u, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        epochs_run = ep + 1
        # inner validation (QLIKE metrics on vol)
        model.eval()
        with torch.no_grad():
            vb = va_idx
            u_v = model(Xte[vb])
            pred_v = torch.exp(u_v).detach().numpy()
        q, r = eval_metrics(pred_v, y[vb])
        hist.append(q)
        if q < best_q:
            best_q, best_ep = q, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep - best_ep > patience:
            break
        if time.time() - t0 > 20 * 60:
            print("TIME BUDGET HIT"); break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        uo = model(Xte[split:]).detach().numpy()
        uall = model(Xte).detach().numpy()
    pred_o = np.exp(uo); pred_all = np.exp(uall)
    oq, orr = eval_metrics(pred_o, y[split:])
    fq, fr = eval_metrics(pred_all, y)
    cfg = dict(d_model=d_model, n_layers=n_layers, lr=lr, epochs_max=epochs_max,
               stride=stride, batch=batch, seed=seed, warmup=warmup, patience=patience,
               use_ewma=use_ewma, clamp_margin=clamp_margin, calibrate=calibrate)
    extras = {"inner_val_qlike": best_q, "best_ep": best_ep,
              "wall_min": round((time.time() - t0) / 60, 1), "hist_hd": hist[:5]}
    calibrated = None
    if calibrate:
        # honest scale calibration factor from inner val only (train tail)
        # QLIKE-optimal: c* = mean(tv/pv) on variance scale; apply sqrt(c*) to vol
        with torch.no_grad():
            uv = model(Xte[va_idx]).detach().numpy()
        vol_in = np.exp(uv)
        cal = float(np.sqrt(np.mean(((y[va_idx] / vol_in) ** 2))))
        calibrated = pred_o * cal
        cq, cr = eval_metrics(calibrated, y[split:])
        cfq, cfr = eval_metrics(pred_all*cal, y)
        extras["cal_factor"] = cal
        extras["oos20_qlike_cal"] = cq; extras["oos20_rmse_cal"] = cr
        extras["full_qlike_cal"] = cfq
        log_result(tag + "_cal", cfg, cq, cr, cfq, cfr, epochs_run, extra=extras)
    if ensembler is not None:
        ensembler["pred_o"] = pred_o
        ensembler["calibrated"] = calibrated
        ensembler["y"] = y
        ensembler["cfg"] = cfg
        if calibrate:
            ensembler["uv"] = np.exp(uv)
    log_result(tag, cfg, oq, orr, fq, fr, epochs_run, extra=extras)
    return oq, orr, pred_o, calibrated, pred_all


def add_ewma_local(X, valid_idx, log_ret):
    r = np.asarray(log_ret, float); n = len(r)
    s2 = r[0] ** 2; ew = np.zeros(n)
    for t in range(n):
        s2 = 0.94 * s2 + 0.06 * r[t] ** 2
        ew[t] = math.sqrt(s2 * 252.0)
    t_idx = np.asarray(valid_idx, int)
    c1 = ew[t_idx][:, None, None] * np.ones((1, X.shape[1], 1), dtype=np.float32)
    c2 = np.array([ew[max(0, i - 4):i + 1].mean() for i in range(n)])[t_idx][:, None, None] * np.ones((1, X.shape[1], 1), dtype=np.float32)
    return np.concatenate([X, c1, c2], axis=2)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run1"
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    run(cmd, **kw)
