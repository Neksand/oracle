"""Exp3 (idea 2): standardized log-vol target.

z = (log y - mu) / sd from train stats; model predicts z;
vol = exp(z * sd + mu) with z clamped so vol stays in train log range (+margins).
QLIKE loss on vol^2 (exp inside loss). Rest identical to exp2 winner config.
"""
import sys, time, math, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
from experiments.common import load_data, eval_metrics, log_result
from mambapy.mamba import Mamba, MambaConfig
from exp1_mamba import add_ewma_local


def run(tag, d_model=32, n_layers=2, lr=1e-3, epochs_max=100, stride=3, batch=1024,
        seed=7, warmup=5, patience=18):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, valid, split = load_data()
    from app.research.dataset import load_dataset
    ds = load_dataset('SPY')
    X = add_ewma_local(X, valid, ds.log_returns)
    Xte = torch.from_numpy(X).float()
    yte = torch.from_numpy(y).float()
    n = len(y)
    iv = int(split * 0.9)
    tr_idx = np.arange(0, iv, stride)
    va_idx = np.arange(iv, split)

    ly = np.log(np.maximum(y[:split], 1e-6))
    mu, sd = float(ly.mean()), float(ly.std())
    z = (ly - mu) / sd
    ztr = torch.from_numpy(z).float()
    zlo, zhi = float(z.min()), float(z.max())
    margin = (zhi - zlo) * 0.05

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(X.shape[2], d_model)
            self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.fill_(float(np.median(z[:iv])))

        def forward(self, xb):
            zz = self.mamba(self.proj(xb))
            zz = self.ln(zz)
            out = self.head(zz[:, -1, :]).squeeze(-1)
            return torch.clamp(out, zlo - margin, zhi + margin)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1.0, (e + 1) / warmup) if e < warmup else
        0.5 * (1 + math.cos(math.pi * (e - warmup) / max(1, epochs_max - warmup))))

    best_q, best_state, best_ep = float('inf'), None, -1
    t0 = time.time(); epochs_run = 0; hist = []
    for ep in range(epochs_max):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, zb, yb = Xte[idx], ztr[idx], yte[idx]
            opt.zero_grad()
            zz = model(xb)
            vol = torch.exp(zz * sd + mu)
            pv = vol ** 2
            loss = (torch.log(pv) + yb ** 2 / pv - 1.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        epochs_run = ep + 1
        model.eval()
        with torch.no_grad():
            pv_v = torch.exp(model(Xte[va_idx]) * sd + mu).numpy()
        q, r = eval_metrics(np.sqrt(pv_v), y[va_idx])
        hist.append(q)
        if q < best_q:
            best_q, best_ep = q, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep - best_ep > patience or time.time() - t0 > 22 * 60:
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        po = torch.exp(model(Xte[split:]) * sd + mu).numpy()
        pa = torch.exp(model(Xte) * sd + mu).numpy()
    oq, orr = eval_metrics(po, y[split:])
    fq, fr = eval_metrics(pa, y)
    cfg = dict(arch="z-logvol", d_model=d_model, n_layers=n_layers, lr=lr,
               epochs_max=epochs_max, stride=stride, seed=seed, patience=patience,
               use_ewma=True, mu=mu, sd=sd)
    log_result(tag, cfg, oq, orr, fq, fr, epochs_run,
               extra={"inner_val_qlike": best_q, "best_ep": best_ep,
                      "wall_min": round((time.time() - t0) / 60, 1)})
    return oq, orr


if __name__ == "__main__":
    run("exp3_zlogvol_seed7")
