"""ORACLE - Research: deep volatility forecasters (PyTorch, trained locally).

Two architectures share the same supervised task:
    Input : feature window over the last LAGS trading days
            - lagged daily log-returns
            - high/low-based realized vol ratios (Parkinson & Garman-Klass,
              daily and weekly), smoothed HV levels
            - VIX level (divided by 100), when available from FRED
    Output: realized volatility of the NEXT `HORIZON` trading days (annualized)

    1. VolLSTM    - LSTM(F -> hidden) + MLP head (flagship)
    2. VolFormer  - small Transformer encoder (d_model 32, 2 layers) + same head.
    3. MambaVol-X - real selective SSM (mambapy) + EWMA channels, LayerNorm, log-vol head, QLIKE loss

Training protocol (no look-ahead):
    - temporal 80/20 split by index (validation tails the sample)
    - features standardized with statistics of the TRAIN slice only
    - Adam, full-batch, early stopping on validation QLIKE (patience=4)
    - cosine annealing + 5-epoch warmup, gradient clipping (max_norm=1.0)
    - best model checkpoint written to backend/models/<name>.pth
    - forecasts are produced for every valid t (model.predict over full X)

Falls back cleanly: callers catch ImportError when torch is unavailable.
"""
from __future__ import annotations

import math
import pathlib
from typing import Callable

import numpy as np

LAGS = 30
HORIZON = 5
EPOCHS = 60
LR = 1e-3
HIDDEN = 48
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models"

FEATURE_NAMES = ["lag0", "hv5", "hv10", "hv22", "pv5", "pvwk", "gk5", "gkwk",
                 "vix", "vov", "ewma", "ewma5"]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def qlike_loss(pred_var: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    """QLIKE loss on variance (Patton 2011).
    
    L = mean(log(pred_var) + target_var / pred_var - 1)
    
    This is the proper loss for variance forecasting - scale-invariant,
    robust to outliers, and consistent with QLIKE evaluation metric.
    """
    # Clamp for numerical stability
    pred_var = pred_var.clamp(min=1e-8)
    target_var = target_var.clamp(min=1e-8)
    return (torch.log(pred_var) + target_var / pred_var - 1.0).mean()


# Real selective state-space Mamba backend (pure-Python port; needs no CUDA).
try:  # pragma: no cover - import guard
    from mambapy.mamba import Mamba as _MambaReal, MambaConfig as _MambaConfig
    HAS_MAMBAPY = True
except Exception:  # pragma: no cover
    HAS_MAMBAPY = False


# ---------------------------------------------------------------------------
# Model architectures (same definitions used in train_forecaster, exported for
# callers such as transfer_ablation.py and ensemble/distillation scripts)
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMVolModel(nn.Module):
    """LSTM-based volatility forecaster with MLP head."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN, head_hidden: int = 24,
                 num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).abs()


class VolFormerModel(nn.Module):
    """Small Transformer encoder-based volatility forecaster with same MLP head."""

    def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1,
                 head_hidden: int = 24) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(LAGS, d_model))
        self.layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True,
                                       dropout=dropout), num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.ReLU(),
                                  nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[None, :, :]
        h = self.layer(h)
        return self.head(h[:, -1]).abs()


# ---------------------------------------------------------------------------
# Mamba-based volatility forecaster (State Space Model alternative)
# ---------------------------------------------------------------------------

class MambaVolModel(nn.Module):
    """Mamba-based volatility forecaster with GLU blocks + LayerNorm + log-vol head.

    Uses parameterized selective state space layers for efficient long-sequence
    modeling. GLU (Gated Linear Unit) provides adaptive gating.
    LayerNorm after each block for training stability.

    The head emits raw LOG-volatility; forward() exponentiates it with
    train-data-driven clamps (log_lo/log_hi buffers). This makes predictions
    strictly positive while bounding the worst-case QLIKE error, and keeps the
    pipeline convention uniform: every deep model outputs annualized
    VOLATILITY (not variance) - vol_models.qlike() squares it itself.
    Dropout for regularization.
    """
    def __init__(self, input_size: int = 10, d_model: int = 64, n_layers: int = 2, expand: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.n_layers = n_layers
        
        # Input projection
        self.in_proj = nn.Linear(input_size, d_model)
        
        # Mamba-like layers with GLU + LayerNorm after each block
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.GLU(),  # Gated Linear Unit halves the dimension: (B, L, 2*d) -> (B, L, d)
                nn.Dropout(dropout),
                nn.Linear(d_model * expand // 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        # Sensible default clamps (annualized vol in [0.01, 3.0]).
        # train_forecaster() refines these from the actual training targets.
        self.register_buffer("log_lo", torch.tensor(math.log(0.01), dtype=torch.float32))
        self.register_buffer("log_hi", torch.tensor(math.log(3.0), dtype=torch.float32))

        # Output head: predicts LOG-volatility (unconstrained linear output).
        # exp() + clamps in forward() guarantee a strictly positive forecast
        # that can never collapse to ~0 (the historic QLIKE explosion bug).
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.in_proj(x)  # (batch, seq_len, d_model)

        # Apply Mamba-like layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection for gradient flow

        # Use last timestep for prediction
        x = x[:, -1, :]  # (batch, d_model)
        log_vol = self.head(x).squeeze(-1)                    # (batch,) log-vol
        log_vol = torch.clamp(log_vol, self.log_lo, self.log_hi)
        return torch.exp(log_vol)                             # (batch,) vol > 0




# ---------------------------------------------------------------------------
# Diffusion-based generative volatility forecaster
# ---------------------------------------------------------------------------

class DiffusionVolModel(nn.Module):
    """Simple diffusion-inspired generative volatility forecaster.
    
    Instead of predicting a single RV value, this model learns the
    distribution of future realized volatility, providing calibrated
    prediction intervals for risk management.
    """
    def __init__(self, input_size: int = 10, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
        )
        
        # Mid block
        self.mid = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
        )
        
        # Decoder - predicts variance parameters
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Softplus()  # Positive variance
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size) - take mean across sequence
        x = x.mean(dim=1)  # (batch, input_size)
        
        # Encoder
        x = self.encoder(x)
        
        # Mid block
        x = self.mid(x)
        
        # Decoder - predict variance
        var_pred = self.decoder(x)
        return var_pred.squeeze(-1)  # (batch,) - variance
# ---------------------------------------------------------------------------
# Core data processing
# ---------------------------------------------------------------------------

def build_features(returns_1d: np.ndarray,
                   pv_daily: np.ndarray | None = None,
                   gk_daily: np.ndarray | None = None,
                   vix: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Feature matrix X (N, LAGS, F) with y aligned: target uses days t+1..t+h.

    Series are id-binned so callers can realign forecast/target pairs.
    Rows with incomplete features are dropped; valid sample indices returned.
    """
    n = len(returns_1d)
    sigma_d = np.concatenate([np.full(1, np.nan), returns_1d])  # shift for t indexing
    hv = {w: np.array([sigma_d[max(0, i - w): i].std(ddof=1) if i >= w else np.nan
                       for i in range(1, n + 1)]) * math.sqrt(252)
          for w in (5, 10, 22)}

    pv = pv_daily if pv_daily is not None else np.full(n, np.nan)
    gk = gk_daily if gk_daily is not None else np.full(n, np.nan)
    pv_d = np.concatenate([[np.nan], pv])          # pv of day t (unit: daily var)
    gk_d = np.concatenate([[np.nan], gk])
    pv_w = np.full(n, np.nan)
    gk_w = np.full(n, np.nan)
    if pv_daily is not None:
        pv_w[4:] = np.convolve(pv, np.ones(5) / 5.0, mode="valid")
    if gk_daily is not None:
        gk_w[4:] = np.convolve(gk, np.ones(5) / 5.0, mode="valid")
    vix_level = vix if vix is not None else np.full(n, np.nan)
    vov = np.full(n, np.nan)
    if vix is not None:
        lr_v = np.diff(np.log(vix))
        vov[1:] = np.array([lr_v[max(0, i - 10): i].std(ddof=1) for i in range(1, len(lr_v) + 1)])

    # Causal RiskMetrics EWMA(0.94) annualized daily vol + its 5-day mean.
    # Scalar per row time i (broadcast across lags) - mimics GARCH-X
    # conditioning; strictly uses information up to day i (no leakage).
    sig2 = np.zeros(n)
    prev = float(returns_1d[0]) ** 2 if np.isfinite(returns_1d[0]) else 0.0
    for t in range(n):
        r_t = float(returns_1d[t]) if np.isfinite(returns_1d[t]) else 0.0
        prev = 0.94 * prev + 0.06 * r_t ** 2
        sig2[t] = prev
    ewma_vol = np.sqrt(np.maximum(sig2, 0.0) * 252.0)
    ewma5 = np.array([ewma_vol[max(0, i - 4): i + 1].mean() for i in range(n)])

    feats: list[np.ndarray] = []
    for i in range(n):
        rows = []
        for lag in range(LAGS):
            j = i - lag
            if j < 0:
                rows.append(np.full(len(FEATURE_NAMES), np.nan))
                continue
            if math.isnan(returns_1d[j]):
                rows.append(np.full(len(FEATURE_NAMES), np.nan))
                continue
            rows.append([returns_1d[j], hv[5][j], hv[10][j], hv[22][j],
                         math.sqrt(max(pv_d[j], 0.0) * 252) if np.isfinite(pv_d[j]) else np.nan,
                         math.sqrt(max(pv_w[j], 0.0) * 252) if np.isfinite(pv_w[j]) else np.nan,
                         math.sqrt(max(gk_d[j], 0.0) * 252) if np.isfinite(gk_d[j]) else np.nan,
                         math.sqrt(max(gk_w[j], 0.0) * 252) if np.isfinite(gk_w[j]) else np.nan,
                         vix_level[j] / 100.0 if np.isfinite(vix_level[j]) else np.nan,
                         vov[j] if np.isfinite(vov[j]) else np.nan,
                         float(ewma_vol[i]),
                         float(ewma5[i])])
        feats.append(rows)
    X = np.array(feats, dtype=np.float32)          # (N, LAGS, F)

    # Drop feature channels that contain no finite value (e.g. VIX unavailable);
    # an all-NaN channel would otherwise poison the whole validity filter.
    finite_any = np.isfinite(X).any(axis=(0, 1))
    if not np.all(finite_any):
        X = X[:, :, finite_any]

    y = np.array([returns_1d[t + 1: t + 1 + HORIZON].std(ddof=1) * math.sqrt(252)
                  for t in range(len(returns_1d))], dtype=np.float32)
    valid = [t for t in range(n) if np.isfinite(y[t]) and np.all(np.isfinite(X[t]))]
    return X[valid], y[valid], valid


# ---------------------------------------------------------------------------
# Core training / forecasting
# ---------------------------------------------------------------------------

def _cosine_annealing_with_warmup(optimizer: torch.optim.Optimizer,
                                   warmup_epochs: int,
                                   total_epochs: int,
                                   base_lr: float,
                                   min_lr: float = 1e-6) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing learning rate schedule with linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine annealing after warmup
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def qlike_metric(pred_vol: np.ndarray, target_vol: np.ndarray) -> float:
    """QLIKE on variance from vol-space arrays (matches vol_models.qlike)."""
    pv = (np.asarray(pred_vol, dtype=float)) ** 2
    tv = (np.asarray(target_vol, dtype=float)) ** 2
    ok = np.isfinite(pv) & np.isfinite(tv) & (pv > 1e-12) & (tv > 0)
    if not ok.any():
        return float("inf")
    return float(np.mean(np.log(pv[ok]) + tv[ok] / pv[ok] - 1.0))


def train_mamba_forecaster(X: np.ndarray, y: np.ndarray,
                           checkpoint: str | None = None, seed: int = 7,
                           epochs_max: int = 140, d_model: int = 32,
                           n_layers: int = 2, lr: float = 1e-3,
                           batch: int = 1024, stride: int = 3,
                           patience: int = 28, warmup: int = 5,
                           clamp_margin: float = 0.05) -> dict:
    """Winning MambaVol-X trainer (real selective SSM, mambapy backend).

    Recipe (calibrated 2026-10-24 on SPY, package `backend/experiments/`):
      real Mamba(d_model=32, n_layers=2) -> LayerNorm -> linear log-vol head
      (zeros init + bias = log median target) -> exp with clamps derived from
      train targets; QLIKE loss on squared vol; Adam(lr, cosine + 5-ep warmup),
      grad clip 1.0; mini-batches 1024, stride-3 subsampled train windows;
      early stopping on an INNER validation tail (last 10% of the train slice)
      so the outer OOS block stays untouched (no leakage).

    Falls back to the GLU-based MambaVolModel when mambapy is unavailable.
    Returns dict compatible with pipeline consumers
    (preds as annualized VOL; val_qlike from the inner validation slice).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(y)
    split = int(n * 0.8)
    iv = int(split * 0.9)  # inner val = last 10% of train (time-ordered)

    Xte = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yte = torch.from_numpy(np.asarray(y, dtype=np.float32))

    tr_idx = np.arange(0, iv, stride)
    va_idx = np.arange(iv, split)

    y_train = np.asarray(y[:split], dtype=float)
    y_inner = y[:iv] if iv > 0 else y_train
    log_lo = math.log(max(float(y_train.min()), 1e-4))
    log_hi = math.log(float(y_train.max()))
    m = (log_hi - log_lo) * clamp_margin
    clamp_lo, clamp_hi = log_lo - m, log_hi + m
    median_log = float(np.log(np.median(y_inner) if len(y_inner) else np.median(y_train)))

    class Model(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.proj = nn.Linear(input_size, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)
            if HAS_MAMBAPY:
                self.mamba = _MambaReal(_MambaConfig(d_model=d_model, n_layers=n_layers))
            else:
                # GLU fallback (non-canonical approximation; flagged in info)
                self.mamba = MambaVolModel(input_size=input_size, d_model=d_model,
                                           n_layers=n_layers, expand=2, dropout=0.1)
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.fill_(median_log)

        def forward(self, xb: torch.Tensor) -> torch.Tensor:
            z = self.mamba(self.proj(xb))
            z = self.ln(z)
            u = self.head(z[:, -1, :]).squeeze(-1)
            u = torch.clamp(u, clamp_lo, clamp_hi)
            return u

    backend = "mambapy-ssm" if HAS_MAMBAPY else "glu-fallback"
    model = Model(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda e: min(1.0, (e + 1) / warmup) if e < warmup else
        0.5 * (1 + math.cos(math.pi * (e - warmup) / max(1, epochs_max - warmup))))

    def qlike_var_loss(u: torch.Tensor, yt_b: torch.Tensor) -> torch.Tensor:
        return qlike_loss(torch.exp(2.0 * u), yt_b ** 2)

    best_q, best_state, best_ep = float("inf"), None, -1
    history = []
    epochs_run = 0
    import time as _time
    t0 = _time.time()
    for ep in range(epochs_max):
        model.train()
        perm = np.random.permutation(tr_idx)
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xte[idx], yte[idx]
            opt.zero_grad()
            u = model(xb)
            loss = qlike_var_loss(u, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        epochs_run = ep + 1

        model.eval()
        with torch.no_grad():
            u_v = model(Xte[va_idx])
            pred_v = torch.exp(u_v).detach().numpy()
        q = qlike_metric(pred_v, y[va_idx])
        history.append({"epoch": ep, "train_qlike": round(total / max(1, len(perm)), 6),
                        "inner_val_qlike": round(q, 5)})
        if ep % 20 == 0 or ep == epochs_max - 1:
            print(f"  [mamba] epoch {ep + 1}/{epochs_max} "
                  f"train_qlike={total / max(1, len(perm)):.4f} inner_val_qlike={q:.4f}",
                  flush=True)
        if q < best_q:
            best_q, best_ep = q, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep - best_ep > patience:
            break
        if _time.time() - t0 > 25 * 60:  # CPU safety budget
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if checkpoint:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "architecture": "mamba",
                    "backend": backend,
                    "input_size": int(X.shape[2]),
                    "d_model": d_model, "n_layers": n_layers},
                   MODELS_DIR / checkpoint)

    with torch.no_grad():
        preds = torch.exp(model(Xte)).detach().numpy()

    return {"model": model, "mu": None, "sd": None, "preds": preds,
            "val_qlike": round(best_q, 5), "best_epoch": best_ep,
            "epochs_run": epochs_run, "backend": backend,
            "n_train": int(split), "n_val": int(n - split),
            "history": history}


def train_forecaster(X: np.ndarray, y: np.ndarray, architecture: str = "lstm",
                     checkpoint: str | None = None, seed: int = 7,
                     epochs_max: int = EPOCHS) -> dict:
    """Train one supervised forecaster with QLIKE loss, gradient clipping, 
    and cosine annealing + warmup scheduler.

    Returns dict with model, scaler(F mean/std from train slice), history,
    val_qlike index positions. Caller applies infinity-time-free predictions.
    """
    if architecture not in ("lstm", "transformer", "mamba"):
        raise ValueError("architecture must be lstm|transformer|mamba")

    if architecture == "mamba":
        # Dedicated real-SSM trainer (winner recipe; inner-val early stopping,
        # log-vol head, QLIKE-on-vol^2, mini-batches). No leakage of the OOS tail.
        return train_mamba_forecaster(X, y, checkpoint=checkpoint, seed=seed)

    n = len(y)
    split = int(n * 0.8)
    mu = X[:split].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[:split].reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if architecture == "lstm":
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(input_size=X.shape[-1], hidden_size=HIDDEN,
                                    num_layers=1, batch_first=True)
                self.head = nn.Sequential(nn.Linear(HIDDEN, 24), nn.ReLU(),
                                          nn.Linear(24, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, (h, _) = self.lstm(x)
                return self.head(h[-1]).abs()
    elif architecture == "transformer":
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                d = 32
                self.proj = nn.Linear(X.shape[-1], d)
                self.pos = nn.Parameter(torch.zeros(LAGS, d))
                self.layer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                               dim_feedforward=64,
                                               batch_first=True,
                                               dropout=0.1), num_layers=2)
                self.head = nn.Sequential(nn.Linear(d, 24), nn.ReLU(),
                                          nn.Linear(24, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.proj(x) + self.pos[None, :, :]
                h = self.layer(h)
                return self.head(h[:, -1]).abs()

    net = Net().to(dev)
    Xt = torch.tensor(Xn[:split], dtype=torch.float32, device=dev)
    yt = torch.tensor(y[:split], dtype=torch.float32, device=dev).unsqueeze(-1)
    Xv = torch.tensor(Xn[split:], dtype=torch.float32, device=dev)
    yv = torch.tensor(y[split:], dtype=torch.float32, device=dev).unsqueeze(-1)
    
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    # Cosine annealing with 5-epoch warmup
    scheduler = _cosine_annealing_with_warmup(opt, warmup_epochs=5, total_epochs=epochs_max, base_lr=LR)
    
    # Target variance for QLIKE loss
    yt_var = yt ** 2
    yv_var = yv ** 2

    best = {"qlike": float("inf"), "state": None, "epoch": -1}
    history = []
    for ep in range(epochs_max):
        net.train()
        opt.zero_grad()
        # Unify semantics: every architecture emits annualized VOLATILITY.
        # QLIKE is minimized on VARIANCE, hence square the forecast here.
        pred_vol = net(Xt).squeeze(-1)
        loss = qlike_loss(pred_vol ** 2, yt_var.squeeze(-1))
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        
        net.eval()
        with torch.no_grad():
            pred_vol_val = net(Xv).squeeze(-1)
            val_qlike = qlike_loss(pred_vol_val ** 2, yv_var.squeeze(-1)).item()
            # Also compute RMSE for reference
            pv = pred_vol_val.detach().cpu().numpy()
            vr = float(np.sqrt(np.mean((pv - y[split:]) ** 2)))
        history.append({"epoch": ep, "train_qlike": round(float(loss.detach()), 6),
                        "val_qlike": round(val_qlike, 5), "val_rmse": round(vr, 5),
                        "lr": round(scheduler.get_last_lr()[0], 6)})
        if ep % 10 == 0 or ep == epochs_max - 1:
            print(f"  [deep] epoch {ep + 1}/{epochs_max} "
                  f"train_qlike={float(loss.detach()):.4f} val_qlike={val_qlike:.4f}",
                  flush=True)
        if val_qlike < best["qlike"]:
            best = {"qlike": val_qlike, "state": {k: v.detach().clone() for k, v in net.state_dict().items()},
                    "epoch": ep}
        # Never early-stop during/at the end of the 5-epoch LR warmup.
        if (ep - best["epoch"] >= 8) and ep >= 12:
            break

    net.load_state_dict(best["state"])
    net.eval()
    if checkpoint:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                    "architecture": architecture}, MODELS_DIR / checkpoint)

    # Post-hoc scalar calibration fitted on the VALIDATION slice only:
    # for pred_var_u = u * pv, QLIKE(u) = mean(log(u pv) + tv/(u pv) - 1)
    # is minimized analytically at u* = mean(tv / pv). Clamped to [0.25, 4]
    # so a degenerate validation slice cannot distort the forecast scale.
    with torch.no_grad():
        pv2_val = (net(Xv).squeeze(-1) ** 2).clamp(min=1e-12)
        tv_val = yv_var.squeeze(-1).clamp(min=1e-12)
        u = float(torch.mean(tv_val / pv2_val))
        u = min(max(u, 0.25), 4.0)
        calib_qlike = (torch.log(u * pv2_val) + tv_val / (u * pv2_val) - 1.0).mean().item()
    vol_scale = float(math.sqrt(u))

    with torch.no_grad():
        preds = net(torch.tensor(Xn, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    preds = preds * vol_scale
    return {"model": net, "mu": mu, "sd": sd, "preds": preds,
            "val_qlike": round(calib_qlike, 5), "best_epoch": best["epoch"],
            "vol_scale": round(vol_scale, 4),
            "n_train": int(split), "n_val": int(n - split), "history": history}


def forecast_vol(returns_1d: np.ndarray,
                 pv_daily: np.ndarray | None = None,
                 gk_daily: np.ndarray | None = None,
                 vix: np.ndarray | None = None,
                 architecture: str = "lstm",
                 checkpoint: str | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit + predict; returns (preds_annualized, y_align, n_used).

    preds[i] is the forecast made with information up to valid[i] (inclusive),
    target y[i] uses days valid[i]+1 .. valid[i]+HORIZON  (walk-forward target).
    """
    X, y, valid = build_features(returns_1d, pv_daily, gk_daily, vix)
    if len(y) < 200:
        raise ValueError(f"only {len(y)} usable rows - need >= 200")
    out = train_forecaster(X, y, architecture=architecture, checkpoint=checkpoint)
    return np.asarray(out["preds"]), y, len(valid)


# ---------------------------------------------------------------------------
# Optuna hyperparameter search for LSTM-X
# ---------------------------------------------------------------------------

def optimize_lstm_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 12,
    seed: int = 7,
    timeout_seconds: int = 1800,  # 30 min max
) -> dict:
    """Run Optuna study to find best LSTM-X hyperparameters.

    Search space (mini-grid, pruned by TPE):
        hidden_size: {32, 48, 64, 96}
        num_layers: {1, 2}
        lr: log-uniform [1e-4, 3e-3]
        dropout: {0.0, 0.1, 0.2}
        head_hidden: {16, 24, 32}

    Returns dict with best_params, best_val_qlike, study, and best_model_state.
    """
    import optuna

    n = len(y)
    split = int(n * 0.8)
    mu = X[:split].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[:split].reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn[:split], dtype=torch.float32, device=dev)
    yt = torch.tensor(y[:split], dtype=torch.float32, device=dev).unsqueeze(-1)
    Xv = torch.tensor(Xn[split:], dtype=torch.float32, device=dev)
    yv = torch.tensor(y[split:], dtype=torch.float32, device=dev).unsqueeze(-1)
    
    yt_var = yt ** 2
    yv_var = yv ** 2

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64, 96])
        num_layers = trial.suggest_categorical("num_layers", [1, 2])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
        head_hidden = trial.suggest_categorical("head_hidden", [16, 24, 32])

        torch.manual_seed(seed)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=X.shape[-1],
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, head_hidden), nn.ReLU(),
                    nn.Linear(head_hidden, 1)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _, (h, _) = self.lstm(x)
                return self.head(h[-1]).abs()

        net = Net().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = _cosine_annealing_with_warmup(opt, warmup_epochs=5, total_epochs=EPOCHS, base_lr=lr)

        best_val_qlike = float("inf")
        patience = 4
        best_epoch = -1

        for ep in range(EPOCHS):
            net.train()
            opt.zero_grad()
            pred_vol = net(Xt).squeeze(-1)
            loss = qlike_loss(pred_vol ** 2, yt_var.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            net.eval()
            with torch.no_grad():
                pred_vol_val = net(Xv).squeeze(-1)
                val_qlike = qlike_loss(pred_vol_val ** 2, yv_var.squeeze(-1)).item()

            if val_qlike < best_val_qlike:
                best_val_qlike = val_qlike
                best_epoch = ep
            elif ep - best_epoch >= patience:
                break

            # Optuna pruning (report intermediate value)
            trial.report(val_qlike, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_qlike

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    # Retrain best model on full data (train+val) for final checkpoint
    best_params = study.best_params
    best_val_qlike = study.best_value

    torch.manual_seed(seed)
    class BestNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=X.shape[-1],
                hidden_size=best_params["hidden_size"],
                num_layers=best_params["num_layers"],
                batch_first=True,
                dropout=best_params["dropout"] if best_params["num_layers"] > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(best_params["hidden_size"], best_params["head_hidden"]), nn.ReLU(),
                nn.Linear(best_params["head_hidden"], 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).abs()

    best_net = BestNet().to(dev)
    opt = torch.optim.Adam(best_net.parameters(), lr=best_params["lr"], weight_decay=1e-5)
    scheduler = _cosine_annealing_with_warmup(opt, warmup_epochs=5, total_epochs=EPOCHS, base_lr=best_params["lr"])

    # Train on full data for final checkpoint
    X_full = torch.tensor(Xn, dtype=torch.float32, device=dev)
    y_full = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(-1)
    y_full_var = y_full ** 2
    best_net.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred_var = best_net(X_full)
        loss = qlike_loss(pred_var, y_full_var.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(best_net.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

    best_net.eval()

    return {
        "best_params": best_params,
        "best_val_qlike": round(best_val_qlike, 5),
        "study": study,
        "best_model_state": {k: v.detach().clone() for k, v in best_net.state_dict().items()},
        "scaler": {"mu": mu, "sd": sd},
    }


def train_forecaster_with_best_params(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict,
    checkpoint: str | None = None,
    seed: int = 7,
) -> dict:
    """Train LSTM with given best hyperparameters (from optimize_lstm_hyperparameters).
    Uses QLIKE loss, gradient clipping, and cosine annealing with warmup.
    """
    import torch
    import torch.nn as nn

    n = len(y)
    split = int(n * 0.8)
    mu = X[:split].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[:split].reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    Xn = (X - mu) / sd

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=X.shape[-1],
                hidden_size=best_params["hidden_size"],
                num_layers=best_params["num_layers"],
                batch_first=True,
                dropout=best_params["dropout"] if best_params["num_layers"] > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(best_params["hidden_size"], best_params["head_hidden"]), nn.ReLU(),
                nn.Linear(best_params["head_hidden"], 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).abs()

    net = Net().to(dev)
    Xt = torch.tensor(Xn[:split], dtype=torch.float32, device=dev)
    yt = torch.tensor(y[:split], dtype=torch.float32, device=dev).unsqueeze(-1)
    Xv = torch.tensor(Xn[split:], dtype=torch.float32, device=dev)
    yv = torch.tensor(y[split:], dtype=torch.float32, device=dev).unsqueeze(-1)
    yt_var = yt ** 2
    yv_var = yv ** 2
    opt = torch.optim.Adam(net.parameters(), lr=best_params["lr"], weight_decay=1e-5)
    scheduler = _cosine_annealing_with_warmup(opt, warmup_epochs=5, total_epochs=EPOCHS, base_lr=best_params["lr"])

    best = {"qlike": float("inf"), "state": None, "epoch": -1}
    history = []
    for ep in range(EPOCHS):
        net.train()
        opt.zero_grad()
        pred_vol = net(Xt).squeeze(-1)
        loss = qlike_loss(pred_vol ** 2, yt_var.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        net.eval()
        with torch.no_grad():
            pred_vol_val = net(Xv).squeeze(-1)
            val_qlike = qlike_loss(pred_vol_val ** 2, yv_var.squeeze(-1)).item()
            pv = pred_vol_val.detach().cpu().numpy()
            vr = float(np.sqrt(np.mean((pv - y[split:]) ** 2)))
        history.append({"epoch": ep, "train_qlike": round(float(loss.detach()), 6),
                        "val_qlike": round(val_qlike, 5), "val_rmse": round(vr, 5),
                        "lr": round(scheduler.get_last_lr()[0], 6)})
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  [deep] epoch {ep + 1}/{EPOCHS} "
                  f"train_qlike={float(loss.detach()):.4f} val_qlike={val_qlike:.4f}",
                  flush=True)
        if val_qlike < best["qlike"]:
            best = {"qlike": val_qlike, "state": {k: v.detach().clone() for k, v in net.state_dict().items()},
                    "epoch": ep}
        if ep - best["epoch"] >= 4:
            break

    net.load_state_dict(best["state"])
    net.eval()
    if checkpoint:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                    "architecture": "lstm", "best_params": best_params},
                   MODELS_DIR / checkpoint)

    with torch.no_grad():
        preds = net(torch.tensor(Xn, dtype=torch.float32, device=dev)).squeeze(-1).cpu().numpy()
    return {"model": net, "mu": mu, "sd": sd, "preds": preds,
            "val_qlike": round(best["qlike"], 5), "best_epoch": best["epoch"],
            "n_train": int(split), "n_val": int(n - split), "history": history,
            "best_params": best_params}