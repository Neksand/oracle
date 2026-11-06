#!/usr/bin/env python3
"""
Stage 1: Daily Pre-training for Volatility Models.

Trains LSTM-X and VolFormer on 33 years of SPY daily data with QLIKE loss.
Saves checkpoints for later fine-tuning on intraday data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.research.dataset import load_dataset
from app.research.deep_models import LSTMVolModel, VolFormerModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    "lstm": {
        "window": 30,
        "channels": 10,
        "hidden": 48,
        "num_layers": 1,
        "dropout": 0.1,
        "head_hidden": 24,
    },
    "volformer": {
        "window": 30,
        "channels": 10,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 64,
        "dropout": 0.1,
    },
    "training": {
        "epochs": 60,
        "batch_size": 256,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 4,
        "val_split": 0.2,
        "seed": 7,
    },
    "loss": "qlike",  # "qlike" | "smoothl1"
}


# ---------------------------------------------------------------------------
# QLIKE Loss
# ---------------------------------------------------------------------------


class QLIKELoss(nn.Module):
    """
    QLIKE loss for volatility forecasting (Patton 2011).
    
    L = mean( log(σ̂²) + RV / σ̂² - 1 )
    
    Robust to noise in realized volatility proxy.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, sigma2_pred: torch.Tensor, rv_true: torch.Tensor) -> torch.Tensor:
        """
        sigma2_pred: predicted variance (σ̂²), shape (batch,)
        rv_true: realized variance proxy, shape (batch,)
        """
        sigma2_pred = torch.clamp(sigma2_pred, min=self.eps)
        rv_true = torch.clamp(rv_true, min=self.eps)
        loss = torch.log(sigma2_pred) + rv_true / sigma2_pred - 1
        return loss.mean()


class SmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss on volatility (not variance)."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, sigma_pred: torch.Tensor, rv_true: torch.Tensor) -> torch.Tensor:
        sigma_true = torch.sqrt(torch.clamp(rv_true, min=1e-8))
        diff = torch.abs(sigma_pred - sigma_true)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


def get_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "qlike":
        return QLIKELoss()
    elif loss_name == "smoothl1":
        return SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def prepare_daily_data(
    symbol: str = "SPY",
    window: int = 30,
    val_split: float = 0.2,
    seed: int = 7,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load daily dataset and create windowed sequences for training.
    
    Returns (train_loader, val_loader, scaler_stats)
    """
    # Load dataset (returns dict with features and targets)
    data = load_dataset(symbol, refresh=False)
    
    # Features: 10 channels x window
    # Target: 5-day forward RV (annualized)
    features = data["features"]  # (n_samples, window, channels)
    targets = data["targets"]    # (n_samples,)
    
    # Remove any NaN
    valid = ~np.isnan(features).any(axis=(1,2)) & ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    
    # Temporal split (no shuffle!)
    n = len(features)
    split_idx = int(n * (1 - val_split))
    
    train_features = features[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features[split_idx:]
    val_targets = targets[split_idx:]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.FloatTensor(train_targets)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.FloatTensor(val_targets)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=DEFAULT_CONFIG["training"]["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=DEFAULT_CONFIG["training"]["batch_size"], shuffle=False)
    
    # Scaler stats for later use
    scaler_stats = {
        "feature_mean": train_features.mean(axis=(0,1)).tolist(),
        "feature_std": train_features.std(axis=(0,1)).tolist(),
        "target_mean": float(train_targets.mean()),
        "target_std": float(train_targets.std()),
    }
    
    return train_loader, val_loader, scaler_stats


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: Dict,
    device: torch.device,
    model_name: str,
) -> Dict[str, Any]:
    """Train model with early stopping."""
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"  Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, MODELS_DIR / f"{model_name}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break
    
    # Load best model
    checkpoint = torch.load(MODELS_DIR / f"{model_name}_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pretrain_lstm(config: Dict, device: torch.device, seed: int) -> Dict[str, Any]:
    """Pre-train LSTM-X on daily data."""
    print(f"\n{'='*60}")
    print(f"Pre-training LSTM-X (seed={seed})")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    train_loader, val_loader, scaler_stats = prepare_daily_data(
        window=config["lstm"]["window"],
        val_split=config["training"]["val_split"],
        seed=seed,
    )
    
    # Model
    model = LSTMVolModel(
        input_size=config["lstm"]["channels"],
        hidden_size=config["lstm"]["hidden"],
        num_layers=config["lstm"]["num_layers"],
        dropout=config["lstm"]["dropout"],
        head_hidden=config["lstm"]["head_hidden"],
    ).to(device)
    
    # Loss
    loss_fn = get_loss_fn(config["training"]["loss"])
    
    # Train
    result = train_model(
        model, train_loader, val_loader, loss_fn,
        config["training"], device, f"lstm_daily_seed{seed}"
    )
    
    # Save final checkpoint with scaler
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config["lstm"],
        "scaler_stats": scaler_stats,
        "seed": seed,
        "val_loss": result["best_val_loss"],
        "epoch": result["best_epoch"],
    }, MODELS_DIR / f"lstm_daily_pretrained_seed{seed}.pth")
    
    # Save scaler separately
    with open(MODELS_DIR / f"lstm_daily_scaler_seed{seed}.json", "w") as f:
        json.dump(scaler_stats, f, indent=2)
    
    result["model_path"] = str(MODELS_DIR / f"lstm_daily_pretrained_seed{seed}.pth")
    result["scaler_path"] = str(MODELS_DIR / f"lstm_daily_scaler_seed{seed}.json")
    result["seed"] = seed
    
    return result


def pretrain_volformer(config: Dict, device: torch.device, seed: int) -> Dict[str, Any]:
    """Pre-train VolFormer on daily data."""
    print(f"\n{'='*60}")
    print(f"Pre-training VolFormer (seed={seed})")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader, val_loader, scaler_stats = prepare_daily_data(
        window=config["volformer"]["window"],
        val_split=config["training"]["val_split"],
        seed=seed,
    )
    
    model = VolFormerModel(
        input_size=config["volformer"]["channels"],
        d_model=config["volformer"]["d_model"],
        nhead=config["volformer"]["nhead"],
        num_layers=config["volformer"]["num_layers"],
        dim_feedforward=config["volformer"]["dim_feedforward"],
        dropout=config["volformer"]["dropout"],
    ).to(device)
    
    loss_fn = get_loss_fn(config["training"]["loss"])
    
    result = train_model(
        model, train_loader, val_loader, loss_fn,
        config["training"], device, f"volformer_daily_seed{seed}"
    )
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config["volformer"],
        "scaler_stats": scaler_stats,
        "seed": seed,
        "val_loss": result["best_val_loss"],
        "epoch": result["best_epoch"],
    }, MODELS_DIR / f"volformer_daily_pretrained_seed{seed}.pth")
    
    with open(MODELS_DIR / f"volformer_daily_scaler_seed{seed}.json", "w") as f:
        json.dump(scaler_stats, f, indent=2)
    
    result["model_path"] = str(MODELS_DIR / f"volformer_daily_pretrained_seed{seed}.pth")
    result["scaler_path"] = str(MODELS_DIR / f"volformer_daily_scaler_seed{seed}.json")
    result["seed"] = seed
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily pre-training for volatility models")
    parser.add_argument("--models", nargs="+", default=["lstm", "volformer"],
                        choices=["lstm", "volformer"],
                        help="Models to pre-train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123, 456, 789],
                        help="Random seeds for ensemble")
    parser.add_argument("--loss", default="qlike", choices=["qlike", "smoothl1"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", help="Path to config JSON")
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    config["training"]["loss"] = args.loss
    config["training"]["epochs"] = args.epochs
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
            config.update(user_config)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Loss: {args.loss}")
    
    all_results = {}
    
    if "lstm" in args.models:
        lstm_results = []
        for seed in args.seeds:
            try:
                result = pretrain_lstm(config, device, seed)
                lstm_results.append(result)
            except Exception as e:
                print(f"Error training LSTM seed {seed}: {e}")
        all_results["lstm"] = lstm_results
    
    if "volformer" in args.models:
        volformer_results = []
        for seed in args.seeds:
            try:
                result = pretrain_volformer(config, device, seed)
                volformer_results.append(result)
            except Exception as e:
                print(f"Error training VolFormer seed {seed}: {e}")
        all_results["volformer"] = volformer_results
    
    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "device": str(device),
        "results": all_results,
    }
    with open(MODELS_DIR / "pretrain_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        for r in results:
            print(f"  {model_name} seed={r['seed']}: val_loss={r['best_val_loss']:.6f} "
                  f"(epoch {r['best_epoch']+1})")


if __name__ == "__main__":
    main()