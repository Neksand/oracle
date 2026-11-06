#!/usr/bin/env python3
"""
Ensemble & Distillation for Volatility Forecasting.

Combines classical (GARCH, HAR) and deep learning models via:
- Inverse-QLIKE weighting
- Stacking with meta-learner
- Knowledge distillation to small student model for deployment
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.research.deep_models import LSTMVolModel, VolFormerModel
from app.research.vol_models import GARCHModel, HARModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ensemble Methods
# ---------------------------------------------------------------------------


@dataclass
class EnsemblePrediction:
    """Container for ensemble predictions."""
    model_names: List[str]
    weights: np.ndarray
    predictions: np.ndarray  # (n_models, n_samples)
    ensemble_pred: np.ndarray  # (n_samples,)
    target: np.ndarray
    metrics: Dict[str, float]


def inverse_qlike_weights(
    qlike_scores: Dict[str, float],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Compute ensemble weights from QLIKE scores.
    
    Lower QLIKE = better = higher weight.
    weight_i = exp(-QLIKE_i / T) / sum_j exp(-QLIKE_j / T)
    """
    models = list(qlike_scores.keys())
    scores = np.array([qlike_scores[m] for m in models])
    
    # Convert to weights (negative QLIKE because lower is better)
    weights = np.exp(-scores / temperature)
    weights = weights / weights.sum()
    
    return {m: float(w) for m, w in zip(models, weights)}


def optimize_ensemble_weights(
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    metric: str = "qlike",
    method: str = "scipy",
) -> Dict[str, float]:
    """
    Optimize ensemble weights to minimize metric on validation set.
    
    predictions: model_name -> predictions array (n_samples,)
    target: true values (n_samples,)
    """
    from scipy.optimize import minimize
    
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    n_models = len(model_names)
    
    def objective(weights):
        weights = np.clip(weights, 0, 1)
        weights = weights / (weights.sum() + 1e-8)
        ensemble = pred_matrix @ weights
        
        if metric == "qlike":
            # QLIKE on variance scale
            sigma2 = np.clip(ensemble ** 2, 1e-8, None)
            rv = target
            loss = np.mean(np.log(sigma2) + rv / sigma2 - 1)
        elif metric == "rmse":
            loss = np.sqrt(np.mean((ensemble - target) ** 2))
        elif metric == "mae":
            loss = np.mean(np.abs(ensemble - target))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return loss
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    # Constraints: weights >= 0, sum = 1
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    optimal_weights = result.x
    optimal_weights = np.clip(optimal_weights, 0, 1)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    return {m: float(w) for m, w in zip(model_names, optimal_weights)}


def stacking_meta_learner(
    base_predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    val_predictions: Dict[str, np.ndarray],
    val_target: np.ndarray,
    meta_model: str = "ridge",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stacking: train meta-learner on base model predictions.
    
    Returns (train_meta_pred, val_meta_pred)
    """
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    
    model_names = list(base_predictions.keys())
    X_train = np.column_stack([base_predictions[m] for m in model_names])
    X_val = np.column_stack([val_predictions[m] for m in model_names])
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Meta-learner
    if meta_model == "ridge":
        meta = Ridge(alpha=1.0)
    elif meta_model == "linear":
        meta = LinearRegression()
    elif meta_model == "gbr":
        meta = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    else:
        raise ValueError(f"Unknown meta_model: {meta_model}")
    
    meta.fit(X_train_scaled, target)
    
    train_pred = meta.predict(X_train_scaled)
    val_pred = meta.predict(X_val_scaled)
    
    return train_pred, val_pred


def build_ensemble(
    classical_predictions: Dict[str, np.ndarray],
    deep_predictions: Dict[str, np.ndarray],
    target: np.ndarray,
    val_classical: Dict[str, np.ndarray],
    val_deep: Dict[str, np.ndarray],
    val_target: np.ndarray,
    method: str = "inverse_qlike",
) -> EnsemblePrediction:
    """
    Build ensemble from classical and deep model predictions.
    """
    all_predictions = {**classical_predictions, **deep_predictions}
    all_val = {**val_classical, **val_deep}
    
    if method == "inverse_qlike":
        # Compute QLIKE on validation for each model
        qlike_scores = {}
        for name, pred in all_val.items():
            sigma2 = np.clip(pred ** 2, 1e-8, None)
            rv = val_target
            qlike = np.mean(np.log(sigma2) + rv / sigma2 - 1)
            qlike_scores[name] = qlike
        weights = inverse_qlike_weights(qlike_scores)
    
    elif method == "optimize":
        weights = optimize_ensemble_weights(all_val, val_target, metric="qlike")
    
    elif method == "stacking":
        # Use stacking on validation, then apply to test
        train_meta, val_meta = stacking_meta_learner(
            all_predictions, target, all_val, val_target
        )
        # For stacking, weights are implicit in meta-learner
        # We'll use meta-learner predictions directly
        pass
    
    # Compute ensemble prediction
    model_names = list(all_predictions.keys())
    pred_matrix = np.column_stack([all_predictions[m] for m in model_names])
    weight_array = np.array([weights.get(m, 0) for m in model_names])
    
    ensemble_pred = pred_matrix @ weight_array
    
    # Metrics
    sigma2 = np.clip(ensemble_pred ** 2, 1e-8, None)
    metrics = {
        "qlike": float(np.mean(np.log(sigma2) + target / sigma2 - 1)),
        "rmse": float(np.sqrt(np.mean((ensemble_pred - target) ** 2))),
        "mae": float(np.mean(np.abs(ensemble_pred - target))),
    }
    
    return EnsemblePrediction(
        model_names=model_names,
        weights=weight_array,
        predictions=pred_matrix,
        ensemble_pred=ensemble_pred,
        target=target,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------


class StudentModel(nn.Module):
    """Small student model for distillation."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # Ensure positive volatility
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.head(out)
        return out.squeeze(-1)


def distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Knowledge distillation loss.
    
    L = alpha * L_student_target + (1-alpha) * L_student_teacher
    
    For regression, we use MSE for both terms.
    """
    # Student vs true target
    loss_target = nn.MSELoss()(student_pred, target)
    
    # Student vs teacher (soft targets)
    loss_teacher = nn.MSELoss()(student_pred, teacher_pred.detach())
    
    return alpha * loss_target + (1 - alpha) * loss_teacher


def train_student_distillation(
    student: nn.Module,
    teacher_predictions: Dict[str, np.ndarray],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device,
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """
    Train student model to mimic ensemble teacher.
    
    teacher_predictions: model_name -> predictions on training set
    """
    # Create ensemble teacher predictions
    teacher_preds = np.mean(list(teacher_predictions.values()), axis=0)
    
    # Create dataset with teacher predictions as additional target
    # We'll modify the training loop to use distillation loss
    
    optimizer = optim.AdamW(
        student.parameters(),
        lr=config.get("distill_lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    epochs = config.get("distill_epochs", 50)
    patience = config.get("distill_patience", 5)
    
    # Convert teacher predictions to tensor
    teacher_tensor = torch.FloatTensor(teacher_preds).to(device)
    
    for epoch in range(epochs):
        student.train()
        train_losses = []
        
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Get teacher prediction for this batch
            batch_start = i * train_loader.batch_size
            batch_end = batch_start + xb.shape[0]
            batch_teacher = teacher_tensor[batch_start:batch_end]
            
            optimizer.zero_grad()
            student_pred = student(xb).squeeze(-1)
            
            loss = distillation_loss(student_pred, batch_teacher, yb, alpha=alpha)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        student.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                student_pred = student(xb).squeeze(-1)
                loss = nn.MSELoss()(student_pred, yb)
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
                "model_state_dict": student.state_dict(),
                "config": config,
            }, MODELS_DIR / "student_distilled_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                break
    
    # Load best
    checkpoint = torch.load(MODELS_DIR / "student_distilled_best.pth", map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def evaluate_student(
    student: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate student model."""
    student.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = student(xb).squeeze(-1)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    return {
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "mae": float(np.mean(np.abs(preds - targets))),
        "qlike": float(np.mean(np.log(np.clip(preds**2, 1e-8, None)) + 
                               targets / np.clip(preds**2, 1e-8, None) - 1)),
    }


# ---------------------------------------------------------------------------
# Deployment Export (ONNX)
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int, int],  # (batch, seq_len, features)
    output_path: Path,
    dynamic_axes: bool = True,
) -> None:
    """Export model to ONNX for deployment."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch", 1: "sequence"},
            "output": {0: "batch"},
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
    )
    print(f"Exported ONNX model to {output_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Results from ensemble + distillation."""
    method: str
    classical_models: List[str]
    deep_models: List[str]
    weights: Dict[str, float]
    ensemble_metrics: Dict[str, float]
    student_metrics: Optional[Dict[str, float]]
    student_path: Optional[str]
    onnx_path: Optional[str]


def run_ensemble_pipeline(
    classical_preds: Dict[str, np.ndarray],
    deep_preds: Dict[str, np.ndarray],
    val_classical: Dict[str, np.ndarray],
    val_deep: Dict[str, np.ndarray],
    target: np.ndarray,
    val_target: np.ndarray,
    train_features: np.ndarray,  # For student training
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    config: Dict,
    device: torch.device,
    input_shape: Tuple[int, int, int],
) -> EnsembleResult:
    """
    Full ensemble + distillation pipeline.
    """
    print(f"\n{'='*60}")
    print("ENSEMBLE + DISTILLATION PIPELINE")
    print(f"{'='*60}")
    
    # 1. Build ensemble
    print("\n1. Building ensemble...")
    ensemble = build_ensemble(
        classical_preds, deep_preds, target,
        val_classical, val_deep, val_target,
        method=config.get("ensemble_method", "inverse_qlike"),
    )
    
    print(f"   Weights: {ensemble.weights}")
    print(f"   Metrics: {ensemble.ensemble_metrics}")
    
    # 2. Train student model
    print("\n2. Training student model (distillation)...")
    
    # Prepare data loaders for student
    train_dataset = TensorDataset(torch.FloatTensor(train_features), torch.FloatTensor(train_targets))
    val_dataset = TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(val_targets))
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("distill_batch_size", 256), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.get("distill_batch_size", 256), shuffle=False)
    
    # Teacher predictions on training set (ensemble)
    teacher_train_preds = ensemble.ensemble_pred
    
    # We need to organize this by model for distillation
    teacher_predictions = {"ensemble": teacher_train_preds}
    
    # Create student
    student = StudentModel(
        input_size=input_shape[2],
        hidden_size=config.get("student_hidden", 32),
        num_layers=config.get("student_layers", 2),
        dropout=config.get("student_dropout", 0.1),
    ).to(device)
    
    train_student_distillation(
        student, teacher_predictions, train_loader, val_loader,
        config, device, alpha=config.get("distill_alpha", 0.7),
    )
    
    # Evaluate student
    test_dataset = TensorDataset(torch.FloatTensor(val_features), torch.FloatTensor(val_targets))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    student_metrics = evaluate_student(student, test_loader, device)
    
    print(f"   Student metrics: {student_metrics}")
    
    # 3. Export ONNX
    onnx_path = MODELS_DIR / "student_deployment.onnx"
    export_to_onnx(student, input_shape, onnx_path)
    
    # Save student PyTorch
    student_path = MODELS_DIR / "student_distilled.pth"
    torch.save({
        "model_state_dict": student.state_dict(),
        "config": {
            "input_size": input_shape[2],
            "hidden_size": config.get("student_hidden", 32),
            "num_layers": config.get("student_layers", 2),
            "dropout": config.get("student_dropout", 0.1),
        },
        "metrics": student_metrics,
    }, student_path)
    
    return EnsembleResult(
        method=config.get("ensemble_method", "inverse_qlike"),
        classical_models=list(classical_preds.keys()),
        deep_models=list(deep_preds.keys()),
        weights=ensemble.weights,
        ensemble_metrics=ensemble.ensemble_metrics,
        student_metrics=student_metrics,
        student_path=str(student_path),
        onnx_path=str(onnx_path),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_ENSEMBLE_CONFIG = {
    "ensemble_method": "inverse_qlike",  # "inverse_qlike", "optimize", "stacking"
    "distill_alpha": 0.7,
    "distill_lr": 1e-3,
    "distill_epochs": 50,
    "distill_patience": 5,
    "distill_batch_size": 256,
    "student_hidden": 32,
    "student_layers": 2,
    "student_dropout": 0.1,
    "weight_decay": 1e-4,
}


def main():
    parser = argparse.ArgumentParser(description="Ensemble & Distillation")
    parser.add_argument("--classical-preds", required=True, help="Path to classical predictions JSON")
    parser.add_argument("--deep-preds", required=True, help="Path to deep predictions JSON")
    parser.add_argument("--val-classical", required=True, help="Path to validation classical predictions")
    parser.add_argument("--val-deep", required=True, help="Path to validation deep predictions")
    parser.add_argument("--target", required=True, help="Path to target values JSON")
    parser.add_argument("--val-target", required=True, help="Path to validation target values")
    parser.add_argument("--train-features", required=True, help="Path to training features NPY")
    parser.add_argument("--train-targets", required=True, help="Path to training targets NPY")
    parser.add_argument("--val-features", required=True, help="Path to validation features NPY")
    parser.add_argument("--val-targets", required=True, help="Path to validation targets NPY")
    parser.add_argument("--config", help="Config JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    config = DEFAULT_ENSEMBLE_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
            config.update(user_config)
    
    device = torch.device(args.device)
    
    # Load data
    classical_preds = json.load(open(args.classical_preds))
    deep_preds = json.load(open(args.deep_preds))
    val_classical = json.load(open(args.val_classical))
    val_deep = json.load(open(args.val_deep))
    target = np.array(json.load(open(args.target)))
    val_target = np.array(json.load(open(args.val_target)))
    train_features = np.load(args.train_features)
    train_targets = np.load(args.train_targets)
    val_features = np.load(args.val_features)
    val_targets = np.load(args.val_targets)
    
    # Convert predictions from lists to arrays
    for d in [classical_preds, deep_preds, val_classical, val_deep]:
        for k, v in d.items():
            d[k] = np.array(v)
    
    input_shape = (1, train_features.shape[1], train_features.shape[2])
    
    result = run_ensemble_pipeline(
        classical_preds, deep_preds,
        val_classical, val_deep,
        target, val_target,
        train_features, train_targets,
        val_features, val_targets,
        config, device, input_shape,
    )
    
    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "result": asdict(result),
    }
    with open(MODELS_DIR / "ensemble_distillation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("ENSEMBLE + DISTILLATION COMPLETE")
    print(f"{'='*60}")
    print(f"Ensemble metrics: {result.ensemble_metrics}")
    print(f"Student metrics: {result.student_metrics}")
    print(f"Student model: {result.student_path}")
    print(f"ONNX model: {result.onnx_path}")


if __name__ == "__main__":
    main()