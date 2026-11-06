#!/usr/bin/env python3
"""
Stage 2: Fine-tuning on Intraday Realized Variance.

Loads daily pre-trained checkpoints and fine-tunes on Binance 5m RV data.
Supports multiple strategies: random init, frozen encoder, full fine-tune, daily-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.research.dataset import load_binance_rv_dataset
from app.research.deep_models import LSTMVolModel, VolFormerModel
from app.research import deep_models as _dm_models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FINE_TUNE_STRATEGIES = [
    "random_init",      # Train from scratch on RV data
    "frozen_encoder",   # Freeze encoder, train head only
    "full_finetune",    # Unfreeze all, discriminative LR
    "daily_only",       # No fine-tune, evaluate daily pretrained on RV
]


# ---------------------------------------------------------------------------
# Data Preparation for RV
# ---------------------------------------------------------------------------


def prepare_rv_data(
    symbol: str = "BTCUSDT",
    window: int = 30,
    val_split: float = 0.2,
    target_estimator: str = "rv",  # "rv", "bv", "tsrv", "msrv"
    seed: int = 7,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load Binance 5m RV dataset and create windowed sequences.
    """
    data = load_binance_rv_dataset(symbol, target_estimator=target_estimator)
    
    # Features: (n_samples, window, channels) - engineered features from 5m bars
    # Targets: (n_samples,) - 5-day forward RV
    features = data["features"]
    targets = data["target_5d"]
    
    # Remove NaN
    valid = ~np.isnan(features).any(axis=(1,2)) & ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    
    if len(features) == 0:
        raise ValueError(f"No valid data for {symbol} {target_estimator}")
    
    # Normalize targets by their mean to handle tiny RV values (~0.001)
    # This brings them to ~1.0 scale, avoiding NaN losses
    target_mean = targets.mean()
    targets_norm = targets / target_mean
    
    # Temporal split
    n = len(features)
    split_idx = int(n * (1 - val_split))
    
    train_features = features[:split_idx]
    train_targets = targets_norm[:split_idx]  # normalized targets
    val_features = features[split_idx:]
    val_targets = targets_norm[split_idx:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.FloatTensor(train_targets)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.FloatTensor(val_targets)
    )
    
    # Return original mean for metric denormalization
    return train_dataset, val_dataset, {
        "feature_mean": train_features.mean(axis=(0,1)).tolist(),
        "feature_std": train_features.std(axis=(0,1)).tolist(),
        "target_mean_original": float(target_mean),
        "target_std_original": float(targets.norm().item() / len(targets) ** 0.5) if False else float(np.std(targets_norm)),
        "target_mean_norm": float(train_targets.mean()),
        "target_std_norm": float(train_targets.std()),
    }


# ---------------------------------------------------------------------------
# QLIKE Loss (same as pretrain)
# ---------------------------------------------------------------------------


class QLIKELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, sigma2_pred: torch.Tensor, rv_true: torch.Tensor) -> torch.Tensor:
        sigma2_pred = torch.clamp(sigma2_pred, min=self.eps)
        rv_true = torch.clamp(rv_true, min=self.eps)
        loss = torch.log(sigma2_pred) + rv_true / sigma2_pred - 1
        return loss.mean()


# ---------------------------------------------------------------------------
# Model Loading & Freezing Utilities
# ---------------------------------------------------------------------------


def load_pretrained_lstm(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load pre-trained LSTM from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    model = LSTMVolModel(
        input_size=config.get("channels", 10),
        hidden_size=config.get("hidden", 48),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.1),
        head_hidden=config.get("head_hidden", 24),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def load_pretrained_volformer(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load pre-trained VolFormer from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    model = VolFormerModel(
        input_size=config.get("channels", 10),
        d_model=config.get("d_model", 32),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 64),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def freeze_encoder(model: nn.Module, model_type: str) -> None:
    """Freeze encoder layers, keep head trainable."""
    if model_type == "lstm":
        # Freeze LSTM layers
        for param in model.lstm.parameters():
            param.requires_grad = False
        # Head stays trainable
        for param in model.head.parameters():
            param.requires_grad = True
    elif model_type == "volformer":
        # Freeze transformer encoder
        for param in model.transformer.parameters():
            param.requires_grad = False
        # Head and input projection stay trainable
        for param in model.input_proj.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Discriminative LR Optimizer
# ---------------------------------------------------------------------------


def create_discriminative_optimizer(
    model: nn.Module,
    model_type: str,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    """Create optimizer with different LRs for encoder vs head."""
    
    if model_type == "lstm":
        encoder_params = list(model.lstm.parameters())
        head_params = list(model.head.parameters())
    elif model_type == "volformer":
        encoder_params = list(model.transformer.parameters()) + list(model.input_proj.parameters())
        head_params = list(model.head.parameters())
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    param_groups = [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]
    
    return optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    config: Dict,
    device: torch.device,
    model_name: str,
) -> Dict[str, Any]:
    """Fine-tuning training loop."""
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    epochs = config.get("finetune_epochs", 50)
    patience = config.get("finetune_patience", 5)
    
    for epoch in range(epochs):
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
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, MODELS_DIR / f"{model_name}_finetuned_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break
    
    # Load best (compatible with all PyTorch versions)
    checkpoint = torch.load(MODELS_DIR / f"{model_name}_finetuned_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    info: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    losses = []
    preds = []
    targets = []
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            losses.append(loss.item())
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    # Denormalize targets back to original scale
    target_mean_original = 1.0
    if info is not None and "target_mean_original" in info:
        target_mean_original = info["target_mean_original"]
    targets_orig = targets * target_mean_original
    preds_orig = preds * target_mean_original
    
    rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(preds_orig - targets_orig)))
    
    # QLIKE on original scale
    eps = 1e-8
    qlike = float(np.mean(np.log(preds_orig**2 + eps) + targets_orig / (preds_orig**2 + eps) - 1))
    
    return {
        "loss": float(np.mean(losses)),
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
    }


# ---------------------------------------------------------------------------
# Main Fine-tuning Orchestration
# ---------------------------------------------------------------------------


@dataclass
class FinetuneResult:
    strategy: str
    model_type: str
    symbol: str
    target_estimator: str
    seed: int
    trainable_params: int
    total_params: int
    best_val_loss: float
    best_epoch: int
    test_metrics: Dict[str, float]
    model_path: str
    duration_sec: float


def run_finetune_experiment(
    model_type: str,
    strategy: str,
    symbol: str,
    target_estimator: str,
    seed: int,
    pretrained_checkpoint: Optional[str],
    config: Dict,
    device: torch.device,
) -> FinetuneResult:
    """Run a single fine-tuning experiment."""
    
    import time
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_type} | {strategy} | {symbol} | {target_estimator} | seed={seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    train_dataset, val_dataset, scaler_stats = prepare_rv_data(
        symbol=symbol,
        window=config[model_type]["window"],
        val_split=config["training"]["val_split"],
        target_estimator=target_estimator,
        seed=seed,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # Model setup based on strategy
    if strategy == "random_init":
        # Fresh model
        if model_type == "lstm":
            model = LSTMVolModel(
                input_size=config["lstm"]["channels"],
                hidden_size=config["lstm"]["hidden"],
                num_layers=config["lstm"]["num_layers"],
                dropout=config["lstm"]["dropout"],
                head_hidden=config["lstm"]["head_hidden"],
            ).to(device)
        else:
            model = VolFormerModel(
                input_size=config["volformer"]["channels"],
                d_model=config["volformer"]["d_model"],
                nhead=config["volformer"]["nhead"],
                num_layers=config["volformer"]["num_layers"],
                dim_feedforward=config["volformer"]["dim_feedforward"],
                dropout=config["volformer"]["dropout"],
            ).to(device)
        model_name = f"{model_type}_{symbol}_{target_estimator}_{strategy}_seed{seed}"
        
    elif strategy == "daily_only":
        # Load pretrained, no fine-tuning
        if model_type == "lstm":
            model, _ = load_pretrained_lstm(pretrained_checkpoint, device)
        else:
            model, _ = load_pretrained_volformer(pretrained_checkpoint, device)
        model.eval()
        model_name = f"{model_type}_{symbol}_{target_estimator}_{strategy}_seed{seed}"
        
        # Evaluate directly
        loss_fn = QLIKELoss()
        test_metrics = evaluate_model(model, val_loader, loss_fn, device, scaler_stats)
        duration = time.time() - start_time
        
        trainable, total = count_trainable_params(model)
        return FinetuneResult(
            strategy=strategy,
            model_type=model_type,
            symbol=symbol,
            target_estimator=target_estimator,
            seed=seed,
            trainable_params=0,
            total_params=total,
            best_val_loss=test_metrics["loss"],
            best_epoch=0,
            test_metrics=test_metrics,
            model_path=pretrained_checkpoint,
            duration_sec=duration,
        )
        
    else:
        # Load pretrained
        if model_type == "lstm":
            model, _ = load_pretrained_lstm(pretrained_checkpoint, device)
        else:
            model, _ = load_pretrained_volformer(pretrained_checkpoint, device)
        model_name = f"{model_type}_{symbol}_{target_estimator}_{strategy}_seed{seed}"
    
    # Apply strategy
    if strategy == "random_init":
        # All parameters trainable from scratch
        unfreeze_all(model)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["training"]["head_lr"],
            weight_decay=config["training"]["weight_decay"],
        )
    elif strategy == "frozen_encoder":
        freeze_encoder(model, model_type)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["training"]["head_lr"],
            weight_decay=config["training"]["weight_decay"],
        )
    elif strategy == "full_finetune":
        unfreeze_all(model)
        optimizer = create_discriminative_optimizer(
            model, model_type,
            encoder_lr=config["training"]["encoder_lr"],
            head_lr=config["training"]["head_lr"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    trainable, total = count_trainable_params(model)
    print(f"  Trainable params: {trainable:,} / {total:,}")
    
    # Train
    loss_fn = QLIKELoss()
    
    if strategy != "daily_only":
        result = train_finetune(
            model, train_loader, val_loader, loss_fn, optimizer,
            config["training"], device, model_name
        )
        best_val_loss = result["best_val_loss"]
        best_epoch = result["best_epoch"]
    else:
        best_val_loss = 0
        best_epoch = 0
    
    # Final evaluation on validation set
    test_metrics = evaluate_model(model, val_loader, loss_fn, device, scaler_stats)
    
    # Save final model
    final_path = MODELS_DIR / f"{model_name}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config[model_type],
        "scaler_stats": scaler_stats,
        "strategy": strategy,
        "symbol": symbol,
        "target_estimator": target_estimator,
        "seed": seed,
        "test_metrics": test_metrics,
    }, final_path)
    
    duration = time.time() - start_time
    
    return FinetuneResult(
        strategy=strategy,
        model_type=model_type,
        symbol=symbol,
        target_estimator=target_estimator,
        seed=seed,
        trainable_params=trainable,
        total_params=total,
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        model_path=str(final_path),
        duration_sec=duration,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_FINETUNE_CONFIG = {
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
    "mamba": {
        "window": 30,
        "channels": 10,
        "d_model": 32,
        "n_layers": 1,
        "expand": 2,
    },
    "training": {
        "batch_size": 256,
        "val_split": 0.2,
        "encoder_lr": 1e-5,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "finetune_epochs": 50,
        "finetune_patience": 5,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning on intraday RV data")
    parser.add_argument("--model-type", required=True, choices=["lstm", "volformer", "mamba"])
    parser.add_argument("--strategies", nargs="+", default=FINE_TUNE_STRATEGIES,
                        choices=FINE_TUNE_STRATEGIES)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--target-estimators", nargs="+", default=["rv", "tsrv", "msrv"],
                        choices=["rv", "bv", "tsrv", "msrv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--pretrained-checkpoint", help="Path to daily pretrained checkpoint")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    config = DEFAULT_FINETUNE_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
            config.update(user_config)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Model: {args.model_type}")
    print(f"Strategies: {args.strategies}")
    print(f"Symbols: {args.symbols}")
    print(f"Targets: {args.target_estimators}")
    print(f"Seeds: {args.seeds}")
    
    # Find pretrained checkpoint if not provided
    if not args.pretrained_checkpoint:
        pretrained_dir = MODELS_DIR
        pattern = f"{args.model_type}_daily_pretrained_seed*.pth"
        checkpoints = list(pretrained_dir.glob(pattern))
        if checkpoints:
            # Use the one with best validation loss
            best_ckpt = min(checkpoints, key=lambda p: float(p.stem.split("_")[-1].replace("seed", "")))
            args.pretrained_checkpoint = str(best_ckpt)
            print(f"Using pretrained checkpoint: {args.pretrained_checkpoint}")
        else:
            print("Warning: No pretrained checkpoint found, will use random_init for all")
            args.pretrained_checkpoint = None
    
    all_results = []
    
    for symbol in args.symbols:
        for target in args.target_estimators:
            for strategy in args.strategies:
                for seed in args.seeds:
                    # Skip daily_only if no pretrained checkpoint
                    if strategy == "daily_only" and not args.pretrained_checkpoint:
                        continue
                    
                    try:
                        result = run_finetune_experiment(
                            model_type=args.model_type,
                            strategy=strategy,
                            symbol=symbol,
                            target_estimator=target,
                            seed=seed,
                            pretrained_checkpoint=args.pretrained_checkpoint,
                            config=config,
                            device=device,
                        )
                        all_results.append(asdict(result))
                        print(f"  ✓ Done: RMSE={result.test_metrics['rmse']:.6f}, QLIKE={result.test_metrics['qlike']:.6f}")
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Save results
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "model_type": args.model_type,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "results": all_results,
    }
    output_path = MODELS_DIR / f"finetune_results_{args.model_type}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"FINE-TUNING COMPLETE - {len(all_results)} experiments")
    print(f"Results saved to {output_path}")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n{'Strategy':<18} {'Symbol':<10} {'Target':<6} {'Seed':<5} {'Trainable':<10} {'RMSE':<10} {'QLIKE':<10} {'Time':<8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['strategy']:<18} {r['symbol']:<10} {r['target_estimator']:<6} "
              f"{r['seed']:<5} {r['trainable_params']:<10,} "
              f"{r['test_metrics']['rmse']:<10.6f} {r['test_metrics']['qlike']:<10.6f} "
              f"{r['duration_sec']:<8.1f}s")


if __name__ == "__main__":
    main()