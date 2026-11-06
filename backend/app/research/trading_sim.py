#!/usr/bin/env python3
"""Trading simulation with transaction costs for volatility forecasting strategies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.research.deep_models import MambaVolModel, LSTMVolModel, DiffusionVolModel


# ---------------------------------------------------------------------------
# Execution cost model
# ---------------------------------------------------------------------------

TRANSACTION_COST_BPS = 10  # 10 bps per trade


def compute_transaction_costs(signal_changes: np.ndarray) -> float:
    """Compute total transaction costs in bps from signal changes."""
    n_changes = len(signal_changes[signal_changes != 0])
    return n_changes * TRANSACTION_COST_BPS


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Engine for backtesting volatility-based trading strategies."""
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = "mamba",
        transaction_cost_bps: int = TRANSACTION_COST_BPS,
        lookback: int = 30,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.transaction_cost_bps = transaction_cost_bps
        self.lookback = lookback
        
    def run(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        confidence_level: float = 0.90,
    ) -> Dict[str, float]:
        """Run backtest on return series."""
        self.model.eval()
        n = len(returns)
        
        # Generate volatility-based signals
        signals = np.zeros(n)
        signal_changes = []
        
        # Simple strategy: go long when predicted vol is low,
        # short when predicted vol is high
        with torch.no_grad():
            # Create dataset windows
            for i in range(self.lookback, n):
                window = returns[i - self.lookback:i]
                # Use model if available, otherwise use simple stats
                if self.model_type == "mamba":
                    window_tensor = torch.tensor(
                        [float(returns[j]) for j in range(i - self.lookback, i)],
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(-1)
                    pred_vol = torch.abs(window_tensor).mean().item()
                else:
                    pred_vol = np.std(returns[max(0, i-20):i+1])
                
                # Signal: go long if vol below threshold, short otherwise
                vol_threshold = np.percentile(volatilities, 40)
                if pred_vol < vol_threshold:
                    signals[i] = 1  # Long
                elif pred_vol > vol_threshold * 1.5:
                    signals[i] = -1  # Short
                else:
                    signals[i] = 0  # Flat
                
                # Track signal changes
                if i > 0:
                    if signals[i] != signals[i-1]:
                        signal_changes.append(i)
        
        # Compute strategy returns - align signals with returns
        # signals[i] applies to return at day i (we hold signal from day i through day i)
        # We'll compute returns as signal * return at next day
        strategy_returns = np.zeros(n - 1)
        for i in range(1, n - 1):
            strategy_returns[i - 1] = signals[i] * returns[i + 1]
        
        # Compute metrics
        total_return = np.cumsum(strategy_returns).sum()
        buy_hold_return = np.cumsum(returns[1:]).sum()
        
        # Transaction costs
        n_trades = len(signal_changes)
        total_costs_bps = compute_transaction_costs(np.array(signal_changes))
        total_costs = total_costs_bps / 10000
        
        net_return = total_return - total_costs
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        excess_return = net_return - 0.0  # risk-free = 0
        annualized_return = net_return * 252
        annualized_vol = np.std(strategy_returns) * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max drawdown
        cum_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        return {
            "total_return": float(total_return),
            "buy_hold_return": float(buy_hold_return),
            "net_return": float(net_return),
            "total_costs_bps": int(total_costs_bps),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "n_trades": int(n_trades),
            "n_days": int(n - 1),
        }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0, 0.02, n)
    volatilities = np.abs(np.random.normal(0.015, 0.005, n))
    
    # Initialize model and backtest engine
    model = MambaVolModel(input_size=10, d_model=32, n_layers=1)
    engine = BacktestEngine(model, model_type="mamba", transaction_cost_bps=10)
    
    # Run backtest
    metrics = engine.run(returns, volatilities, confidence_level=0.90)
    
    # Print results
    print("=" * 60)
    print("TRADING STRATEGY BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:           {metrics['total_return']:.4f}")
    print(f"  Buy & Hold Return:      {metrics['buy_hold_return']:.4f}")
    print(f"  Net Return:             {metrics['net_return']:.4f}")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:           {metrics['max_drawdown']:.4f}")
    print(f"  Win Rate:               {metrics['win_rate']:.1%}")
    print(f"  Total Trades:           {metrics['n_trades']}")
    print(f"  Transaction Costs:      {metrics['total_costs_bps']} bps")
    print("=" * 60)
