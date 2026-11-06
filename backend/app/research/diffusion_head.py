"""
ORACLE - Research: Diffusion Head for Volatility Distribution Forecasting

Implements DDPM v-prediction (faster convergence than noise prediction)
with DDIM sampling for calibrated prediction intervals and CRPS evaluation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Cosine Noise Schedule (Nichol & Dhariwal 2021)
# =============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule as proposed in Nichol & Dhariwal (2021).
    Better for high-resolution data than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# =============================================================================
# Diffusion RV Head - DDPM v-prediction
# =============================================================================

class DiffusionRVHead(nn.Module):
    """
    DDPM v-prediction head for realized volatility distribution forecasting.
    
    Instead of predicting noise (eps), predicts v = alpha_t * noise - sigma_t * x_0
    which provides faster convergence and better sample quality.
    
    Architecture:
    - Time embedding: sinusoidal + MLP
    - Denoiser: MLP with residual connections
    - v-prediction parameterization for stable training
    - DDIM deterministic sampling for prediction intervals
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_timesteps: int = 1000,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_timesteps = n_timesteps
        
        # Cosine noise schedule
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        
        # Denoiser network - predicts v = alpha_t * noise - sigma_t * x_0
        layers = []
        input_dim = d_model + 1  # mamba_out + x_t
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))  # predicts v
        self.denoiser = nn.Sequential(*layers)
        
        # Variance prediction head (for direct variance prediction without sampling)
        self.variance_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
    
    def forward(
        self,
        mamba_out: torch.Tensor,
        t: torch.Tensor,
        target_rv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or sampling.
        
        Args:
            mamba_out: (B, d_model) - Mamba encoder output
            t: (B,) - timestep indices [0, n_timesteps-1]
            target_rv: (B,) - target RV for training (variance scale)
        
        Returns:
            If target_rv provided: v-prediction loss
            Else: sampled RV distribution (B, n_samples)
        """
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())  # (B, d_model)
        
        if target_rv is not None:
            # Training: v-prediction loss
            return self._training_loss(mamba_out, t_emb, target_rv, t)
        else:
            # Sampling: DDIM deterministic sampling
            return self.sample(mamba_out)
    
    def _training_loss(
        self,
        mamba_out: torch.Tensor,
        t_emb: torch.Tensor,
        target_rv: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute v-prediction loss.
        
        v = alpha_t * noise - sigma_t * x_0
        where x_0 is the clean target (RV variance)
        """
        batch_size = mamba_out.shape[0]
        device = mamba_out.device
        
        # Sample noise - ensure correct shape (B, 1)
        noise = torch.randn(batch_size, 1, device=device)
        
        # Get alpha_t and sigma_t for each sample
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)  # (B, 1)
        sigma_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)  # (B, 1)
        
        # Noisy target: x_t = alpha_t * x_0 + sigma_t * noise
        x_t = alpha_t * target_rv.unsqueeze(-1) + sigma_t * noise
        
        # v-target: v = alpha_t * noise - sigma_t * x_0
        v_target = alpha_t * noise - sigma_t * target_rv.unsqueeze(-1)
        
        # Predict v
        denoiser_input = torch.cat([mamba_out, x_t], dim=-1)  # (B, d_model + 1)
        v_pred = self.denoiser(denoiser_input)  # (B, 1)
        
        # MSE loss on v-prediction
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        mamba_out: torch.Tensor,
        n_samples: int = 100,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM deterministic sampling for prediction intervals.
        
        Args:
            mamba_out: (B, d_model) - Mamba encoder output
            n_samples: Number of Monte Carlo samples per input
            n_steps: Number of DDIM sampling steps (less than training steps)
            eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)
        
        Returns:
            samples: (B, n_samples) - sampled RV values (volatility scale)
        """
        batch_size = mamba_out.shape[0]
        device = mamba_out.device
        
        # Repeat mamba_out for n_samples
        mamba_out = mamba_out.unsqueeze(1).repeat(1, n_samples, 1).view(-1, self.d_model)  # (B*n_samples, d_model)
        
        # Time steps for DDIM (skip steps for faster sampling)
        step_ratio = self.n_timesteps // n_steps
        timesteps = torch.arange(0, self.n_timesteps, step_ratio, device=device).flip(0)  # (n_steps,)
        timesteps = torch.cat([timesteps, torch.tensor([0], device=device)])  # add t=0
        
        # Start from pure noise
        x_t = torch.randn(batch_size * n_samples, 1, device=device)
        
        # Time embedding (broadcast to all samples)
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i].repeat(batch_size * n_samples)
            t_next = timesteps[i + 1].repeat(batch_size * n_samples)
            
            t_emb = self.time_embed(t_curr.unsqueeze(-1).float())
            
            # Predict v
            denoiser_input = torch.cat([mamba_out, x_t], dim=-1)
            v_pred = self.denoiser(denoiser_input)
            
            # Compute x_0 from v prediction
            # v = alpha_t * noise - sigma_t * x_0
            # => x_0 = (alpha_t * noise - v) / sigma_t
            # But we don't have noise, so use:
            # x_0 = (x_t - sigma_t * noise) / alpha_t
            # From v = alpha_t * noise - sigma_t * x_0:
            # noise = (v + sigma_t * x_0) / alpha_t
            # Substitute into x_t = alpha_t * x_0 + sigma_t * noise:
            # x_t = alpha_t * x_0 + sigma_t * (v + sigma_t * x_0) / alpha_t
            # x_t * alpha_t = alpha_t^2 * x_0 + sigma_t * v + sigma_t^2 * x_0
            # x_t * alpha_t = (alpha_t^2 + sigma_t^2) * x_0 + sigma_t * v
            # Since alpha_t^2 + sigma_t^2 = 1:
            # x_0 = (alpha_t * x_t - sigma_t * v)
            
            alpha_curr = self.sqrt_alphas_cumprod[t_curr].view(-1, 1)
            sigma_curr = self.sqrt_one_minus_alphas_cumprod[t_curr].view(-1, 1)
            alpha_next = self.sqrt_alphas_cumprod[t_next].view(-1, 1)
            sigma_next = self.sqrt_one_minus_alphas_cumprod[t_next].view(-1, 1)
            
            # Predict x_0
            x_0_pred = alpha_curr * x_t - sigma_curr * v_pred
            
            # DDIM update
            if eta > 0:
                # Stochastic DDIM
                sigma_ddim = eta * torch.sqrt(
                    (1 - alpha_next**2) / (1 - alpha_curr**2) * (1 - alpha_curr**2 / alpha_next**2)
                )
                noise = torch.randn_like(x_t)
                x_t = alpha_next * x_0_pred + torch.sqrt(1 - alpha_next**2 - sigma_ddim**2) * v_pred + sigma_ddim * noise
            else:
                # Deterministic DDIM
                x_t = alpha_next * x_0_pred + torch.sqrt(1 - alpha_next**2) * v_pred
        
        # Final x_0 prediction
        samples = x_t.view(batch_size, n_samples).squeeze(-1)
        
        # Ensure positive (variance scale)
        samples = samples.clamp(min=1e-8)
        
        return samples
    
    def predict_variance(self, mamba_out: torch.Tensor) -> torch.Tensor:
        """Direct variance prediction without sampling (for speed)."""
        return self.variance_head(mamba_out).squeeze(-1)


# =============================================================================
# CRPS (Continuous Ranked Probability Score)
# =============================================================================

def crps_gaussian(pred_mean: np.ndarray, pred_std: np.ndarray, target: np.ndarray) -> float:
    """
    CRPS for Gaussian predictive distribution.
    Closed form: CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ
    """
    from scipy.stats import norm
    
    z = (target - pred_mean) / (pred_std + 1e-8)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    crps = pred_std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_empirical(samples: np.ndarray, target: np.ndarray) -> float:
    """
    CRPS for empirical distribution (Monte Carlo samples).
    CRPS = E|X - y| - 0.5 * E|X - X'|
    """
    n_samples = samples.shape[1]
    
    # E|X - y|
    term1 = np.mean(np.abs(samples - target[:, np.newaxis]), axis=1)
    
    # E|X - X'|
    diff = samples[:, :, np.newaxis] - samples[:, np.newaxis, :]
    term2 = 0.5 * np.mean(np.abs(diff), axis=(1, 2))
    
    crps = term1 - term2
    return float(np.mean(crps))


def crps_quantile(samples: np.ndarray, target: np.ndarray, quantiles: np.ndarray = None) -> float:
    """
    CRPS via quantile approximation (more stable for small sample sizes).
    """
    if quantiles is None:
        quantiles = np.linspace(0.01, 0.99, 99)
    
    sample_quantiles = np.quantile(samples, quantiles, axis=1).T  # (B, n_quantiles)
    
    # Pinball loss at each quantile
    pinball_losses = []
    for i, q in enumerate(quantiles):
        diff = target - sample_quantiles[:, i]
        loss = np.maximum(q * diff, (q - 1) * diff)
        pinball_losses.append(loss)
    
    crps = 2 * np.mean(pinball_losses, axis=0)
    return float(np.mean(crps))


# =============================================================================
# Full Model: Mamba + Diffusion Head
# =============================================================================

class MambaDiffusionVol(nn.Module):
    """
    Full model: Mamba encoder + Diffusion head for RV distribution forecasting.
    """
    
    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        diffusion_timesteps: int = 1000,
    ):
        super().__init__()
        
        # Mamba encoder (import here to avoid circular import)
        try:
            from mamba_ssm import Mamba
            self.encoder = nn.Linear(n_features, d_model)
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(n_layers - 1)
            ]) if n_layers > 1 else nn.ModuleList()
        except ImportError:
            # Fallback to GLU-based MambaVolModel
            from app.research.deep_models import MambaVolModel
            self.mamba_encoder = MambaVolModel(
                input_size=n_features, d_model=d_model, n_layers=n_layers, expand=2
            )
            self.use_fallback = True
        else:
            self.use_fallback = False
        
        # Diffusion head
        self.diffusion_head = DiffusionRVHead(
            d_model=d_model,
            n_timesteps=diffusion_timesteps,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        target_rv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, L, F) - input features
            t: (B,) - timestep for diffusion training
            target_rv: (B,) - target RV for diffusion training
        
        Returns:
            If training: diffusion loss
            Else: (variance_pred, samples) for inference
        """
        if self.use_fallback:
            # Use GLU-based encoder
            mamba_out = self.mamba_encoder.in_proj(x)
            for layer in self.mamba_encoder.layers:
                residual = mamba_out
                mamba_out = layer(mamba_out)
                mamba_out = mamba_out + residual
            mamba_out = mamba_out[:, -1, :]
        else:
            # Real Mamba-SSM
            x = self.encoder(x)
            x = self.mamba(x)
            for layer in self.mamba_layers:
                x = layer(x)
            mamba_out = x[:, -1, :]
        
        if target_rv is not None and t is not None:
            # Training: diffusion loss
            return self.diffusion_head(mamba_out, t, target_rv)
        else:
            # Inference: variance prediction + samples
            var_pred = self.diffusion_head.predict_variance(mamba_out)
            samples = self.diffusion_head.sample(mamba_out)
            return var_pred, samples


# =============================================================================
# Training Helper
# =============================================================================

def train_diffusion_model(
    model: MambaDiffusionVol,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    epochs: int = 30,
    lr: float = 1e-3,
    warmup: int = 5,
    patience: int = 8,
) -> dict:
    """Train the diffusion model with v-prediction loss."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        progress = float(epoch - warmup) / float(max(1, epochs - warmup))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1e-6 / lr + (1.0 - 1e-6 / lr) * cosine_decay
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Random timesteps
            t = torch.randint(0, model.diffusion_head.n_timesteps, (x_batch.shape[0],), device=device)
            
            optimizer.zero_grad()
            loss = model(x_batch, t=t, target_rv=y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        all_crps = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                t = torch.randint(0, model.diffusion_head.n_timesteps, (x_batch.shape[0],), device=device)
                loss = model(x_batch, t=t, target_rv=y_batch)
                val_losses.append(loss.item())
                
                # CRPS evaluation (sample 100 times)
                var_pred, samples = model(x_batch)
                # Convert samples to volatility scale
                samples_vol = torch.sqrt(samples.clamp(min=1e-8))
                target_vol = torch.sqrt(y_batch.clamp(min=1e-8))
                crps = crps_empirical(samples_vol.cpu().numpy(), target_vol.cpu().numpy())
                all_crps.append(crps)
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_crps = np.mean(all_crps)
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'val_crps': avg_crps,
            'lr': scheduler.get_last_lr()[0],
        })
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch:2d}: train={avg_train:.4f}, val={avg_val:.4f}, CRPS={avg_crps:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return {
        'best_val_loss': best_val_loss,
        'best_state': best_state,
        'history': history,
    }