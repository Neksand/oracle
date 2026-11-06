"""ORACLE - Pydantic Data Models.

Economic Scenario Engine: single-asset GBM, multi-asset portfolio,
option valuation, market calibration. Legacy NEXUS models kept for
backward compatibility with POST /simulate.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Legacy (NEXUS compat)
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Input parameters for stochastic 3D Monte Carlo simulation (legacy)."""

    mu: float = Field(default=0.10, ge=-1.0, le=1.0)
    sigma: float = Field(default=0.45, ge=0.01, le=2.0)
    num_paths: int = Field(default=5000, ge=100, le=50000)
    steps: int = Field(default=50, ge=10, le=200)


class SimulationStatistics(BaseModel):
    mean: List[float] = Field(...)
    std: List[float] = Field(...)
    percentile_5: List[float] = Field(...)
    percentile_95: List[float] = Field(...)
    var_95: float = Field(...)
    var_99: float = Field(...)


class SimulationResponse(BaseModel):
    positions: List[float] = Field(...)
    colors: List[float] = Field(...)
    statistics: SimulationStatistics = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finance domain
# ---------------------------------------------------------------------------

class FinanceSimulateRequest(BaseModel):
    """Single-asset GBM economic scenario request."""

    symbol: str = Field(default="CUSTOM", description="Ticker / instrument label.")
    s0: float = Field(default=100.0, gt=0, le=10_000_000, description="Current price level S0.")
    mu_ann: float = Field(default=0.08, ge=-1.0, le=3.0, description="Expected annual drift (0.08 = 8%/y).")
    sigma_ann: float = Field(default=0.25, ge=0.01, le=5.0, description="Annualized volatility.")
    t_days: int = Field(default=90, ge=5, le=1260, description="Projection horizon in calendar days.")
    trading_days: int = Field(default=252, description="252 stocks / 365 crypto.")
    num_paths: int = Field(default=8000, ge=500, le=50000)
    num_sample_paths: int = Field(default=160, ge=20, le=400, description="Stored trajectories for 2D fan chart.")
    bins: int = Field(default=60, ge=20, le=120)
    risk_free: float = Field(default=0.04, ge=-0.05, le=0.25, description="Risk-free rate for Sharpe/options.")
    strike: Optional[float] = Field(default=None, gt=0, description="Option strike K (optional).")
    option_type: Literal["call", "put"] = Field(default="call")
    seed: Optional[int] = Field(default=None)


class PortfolioAssetInput(BaseModel):
    symbol: str = Field(default="ASSET")
    s0: float = Field(default=100.0, gt=0)
    mu_ann: float = Field(default=0.08, ge=-1.0, le=3.0)
    sigma_ann: float = Field(default=0.25, ge=0.01, le=5.0)
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class PortfolioSimulateRequest(BaseModel):
    assets: List[PortfolioAssetInput] = Field(default_factory=list)
    corr: Optional[List[List[float]]] = Field(default=None, description="Correlation matrix; default uniform corr_level.")
    corr_level: float = Field(default=0.3, ge=-0.9, le=0.95)
    t_days: int = Field(default=90, ge=5, le=1260)
    trading_days: int = Field(default=252)
    num_paths: int = Field(default=8000, ge=500, le=50000)
    num_sample_paths: int = Field(default=160, ge=20, le=400)
    bins: int = Field(default=60, ge=20, le=120)
    risk_free: float = Field(default=0.04, ge=-0.05, le=0.25)
    seed: Optional[int] = Field(default=None)


class MoneyStatistics(BaseModel):
    expected_ST: float
    median_ST: float
    p5_ST: float
    p25_ST: float
    p75_ST: float
    p95_ST: float
    mean_logret: float
    std_logret: float
    var95_loss: float
    cvar95_loss: float
    var99_loss: float
    prob_decline: float
    prob_loss_gt10: float
    prob_gain_gt10: float
    sharpe_like: float
    skew: float
    kurt: float
    call_price_mc: Optional[float] = None
    put_price_mc: Optional[float] = None
    call_price_bs: Optional[float] = None
    put_price_bs: Optional[float] = None


class HistogramPayload(BaseModel):
    centers: List[float]
    counts: List[int]


class FinanceSimulateResponse(BaseModel):
    positions: List[float]
    colors: List[float]
    sample_paths: List[List[float]]
    time_axis_days: List[float]
    histogram: HistogramPayload
    stats: MoneyStatistics
    # legacy-compatible mirror for old 3D panels
    statistics: SimulationStatistics
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PortfolioSimulateResponse(FinanceSimulateResponse):
    per_asset_expected: List[float] = Field(default_factory=list)
    diversification_benefit: float = Field(default=0.0)


class QuoteResponse(BaseModel):
    symbol: str
    price: float
    currency: str = "USD"
    change_24h_pct: Optional[float] = None
    source: str = "fallback"
    as_of: str = ""


class HistoryResponse(BaseModel):
    symbol: str
    closes: List[float]
    dates: List[str] = Field(default_factory=list)
    source: str = "fallback"


class CalibrateRequest(BaseModel):
    symbol: str = Field(default="SPY")
    lookback_days: int = Field(default=252, ge=30, le=1500)


class CalibrateResponse(BaseModel):
    symbol: str
    s0: float
    mu_ann: float
    sigma_ann: float
    n_obs: int
    source: str = "fallback"
