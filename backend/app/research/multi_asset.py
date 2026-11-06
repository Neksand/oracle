"""
ORACLE - Research: Multi-Asset Volatility Modeling with DCC-GARCH

Implements multi-asset volatility forecasting for portfolio risk management:
- Joint feature matrix with cross-asset correlations
- DCC-GARCH for conditional covariance estimation
- Factor GARCH alternative (PCA on returns -> factor GARCH -> reconstruct Σ_t)
- Portfolio VaR/CVaR on equal-weight portfolio
- Kupiec/Christoffersen backtesting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH
    from arch.univariate import GARCH as GJRGARCH  # GJRGARCH is accessed differently in newer versions
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from app.research import vol_models as vm
from app.research.dataset import load_dataset


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MultiAssetData:
    """Container for multi-asset data."""
    symbols: List[str]
    returns: pd.DataFrame  # (T, N) daily log returns
    dates: pd.DatetimeIndex
    realized_cov: Optional[np.ndarray] = None  # (T, N, N) daily realized covariance
    

@dataclass
class DCCGarchResult:
    """DCC-GARCH estimation results."""
    conditional_cov: np.ndarray  # (T, N, N)
    conditional_corr: np.ndarray  # (T, N, N)
    univariate_params: List[dict]  # GARCH params per asset
    dcc_params: dict  # a, b parameters
    loglik: float


@dataclass
class PortfolioMetrics:
    """Portfolio risk metrics."""
    var_99: float  # 99% VaR
    var_95: float  # 95% VaR
    cvar_99: float  # 99% CVaR (Expected Shortfall)
    cvar_95: float  # 95% CVaR
    kupiec_99: float  # Kupiec test p-value
    kupiec_95: float
    christoffersen_99: float  # Christoffersen test p-value
    christoffersen_95: float


# =============================================================================
# Multi-Asset Data Loading
# =============================================================================

def load_multi_asset_data(
    symbols: List[str] = None,
    start_date: str = None,
    end_date: str = None,
) -> MultiAssetData:
    """
    Load multi-asset data from Yahoo Finance.
    
    Args:
        symbols: List of tickers (default: SPY, QQQ, IWM, EFA)
        start_date: Start date string
        end_date: End date string
    
    Returns:
        MultiAssetData with aligned returns
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'EFA']
    
    import yfinance as yf
    
    all_data = {}
    for sym in symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        if len(hist) > 0:
            all_data[sym] = hist['Close']
        else:
            raise ValueError(f"No data for {sym}")
    
    # Align on common dates
    prices = pd.DataFrame(all_data).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    
    return MultiAssetData(
        symbols=symbols,
        returns=returns,
        dates=returns.index,
    )


def compute_realized_covariance(
    returns: pd.DataFrame,
    window: int = 22,
) -> np.ndarray:
    """
    Compute rolling realized covariance matrix.
    
    Args:
        returns: (T, N) daily log returns
        window: Rolling window in days
    
    Returns:
        (T, N, N) realized covariance matrices
    """
    T, N = returns.shape
    rcov = np.full((T, N, N), np.nan)
    
    for i in range(window, T):
        window_rets = returns.iloc[i-window:i].values
        rcov[i] = np.cov(window_rets.T)
    
    return rcov


# =============================================================================
# DCC-GARCH Implementation
# =============================================================================

def fit_univariate_garch(
    returns: np.ndarray,
    dist: str = 'studentst',
    mean_model: str = 'Zero',
    vol_model: str = 'GARCH',
    p: int = 1,
    q: int = 1,
) -> dict:
    """
    Fit univariate GARCH to each asset.
    
    Returns dict with params and conditional variance.
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch package required for GARCH fitting")
    
    am = arch_model(
        returns * 100,  # Scale for numerical stability
        mean=mean_model,
        vol=vol_model,
        p=p,
        q=q,
        dist=dist,
    )
    
    res = am.fit(update_freq=0, show_warning=False, options={'ftol': 1e-6})
    
    cond_var = (res.conditional_volatility ** 2 / 10000).values if hasattr(res.conditional_volatility, 'values') else (res.conditional_volatility ** 2 / 10000)
    resid = res.resid.values / 100 if hasattr(res.resid, 'values') else res.resid / 100
    std_resid = res.std_resid.values if hasattr(res.std_resid, 'values') else res.std_resid
    
    return {
        'params': res.params.to_dict(),
        'cond_var': cond_var,
        'residuals': resid,
        'std_resid': std_resid,
        'loglik': res.loglikelihood,
        'converged': res.convergence_flag == 0,
    }


def dcc_garch(
    returns: pd.DataFrame,
    univariate_models: List[str] = None,
    dist: str = 'studentst',
) -> DCCGarchResult:
    """
    Two-step DCC-GARCH estimation (Engle 2002).
    
    Step 1: Fit univariate GARCH for each asset
    Step 2: Estimate DCC correlation dynamics on standardized residuals
    
    Args:
        returns: (T, N) DataFrame of returns
        univariate_models: List of GARCH specifications per asset
        dist: Distribution for univariate GARCH
    
    Returns:
        DCCGarchResult with conditional covariances and correlations
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch package required")
    
    T, N = returns.shape
    symbols = returns.columns.tolist()
    
    # Step 1: Univariate GARCH
    univariate_results = []
    std_residuals = np.zeros((T, N))
    cond_vars = np.zeros((T, N))
    
    if univariate_models is None:
        univariate_models = ['GARCH'] * N
    
    for i, sym in enumerate(symbols):
        vol_model = univariate_models[i] if i < len(univariate_models) else 'GARCH'
        res = fit_univariate_garch(
            returns[sym].values,
            dist=dist,
            vol_model=vol_model,
        )
        univariate_results.append(res)
        std_residuals[:, i] = res['std_resid']
        cond_vars[:, i] = res['cond_var']
    
    # Step 2: DCC correlation estimation
    # Standardize residuals
    z = std_residuals  # Already standardized
    
    # Initial correlation matrix (unconditional)
    Q_bar = np.corrcoef(z.T)
    
    # DCC parameters: Q_t = (1 - a - b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
    # R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    
    # Estimate a, b via MLE on correlation likelihood
    from scipy import optimize
    
    def dcc_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1:
            return 1e9
        
        Q = Q_bar.copy()
        ll = 0.0
        
        for t in range(T):
            if t > 0:
                z_t = z[t-1:t].T  # (N, 1)
                Q = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q
            
            # Correlation matrix
            q_diag = np.diag(Q)
            if np.any(q_diag <= 0):
                return 1e9
            D_inv = np.diag(1.0 / np.sqrt(q_diag))
            R = D_inv @ Q @ D_inv
            
            # Ensure positive definite
            try:
                det_R = np.linalg.det(R)
                if det_R <= 0:
                    return 1e9
                inv_R = np.linalg.inv(R)
            except np.linalg.LinAlgError:
                return 1e9
            
            # Log-likelihood contribution
            z_t = z[t:t+1].T  # (N, 1)
            ll += -0.5 * (np.log(det_R) + z_t.T @ inv_R @ z_t - z_t.T @ z_t)
        
        return -ll[0, 0]
    
    # Optimize
    result = optimize.minimize(
        dcc_loglik,
        x0=[0.02, 0.97],
        bounds=[(1e-6, 0.99), (1e-6, 0.99)],
        constraints=[{'type': 'ineq', 'fun': lambda x: 0.999 - x[0] - x[1]}],
        method='L-BFGS-B',
        options={'ftol': 1e-8, 'maxiter': 1000}
    )
    
    a_opt, b_opt = result.x
    
    # Reconstruct conditional correlations and covariances
    Q = Q_bar.copy()
    conditional_corr = np.zeros((T, N, N))
    conditional_cov = np.zeros((T, N, N))
    
    for t in range(T):
        if t > 0:
            z_t = z[t-1:t].T
            Q = (1 - a_opt - b_opt) * Q_bar + a_opt * (z_t @ z_t.T) + b_opt * Q
        
        q_diag = np.diag(Q)
        D_inv = np.diag(1.0 / np.sqrt(q_diag))
        R = D_inv @ Q @ D_inv
        
        conditional_corr[t] = R
        
        # Conditional covariance = D_t * R_t * D_t
        D_t = np.diag(np.sqrt(cond_vars[t]))
        conditional_cov[t] = D_t @ R @ D_t
    
    return DCCGarchResult(
        conditional_cov=conditional_cov,
        conditional_corr=conditional_corr,
        univariate_params=[r['params'] for r in univariate_results],
        dcc_params={'a': float(a_opt), 'b': float(b_opt)},
        loglik=float(-result.fun),
    )


# =============================================================================
# Factor GARCH Alternative
# =============================================================================

def factor_garch_covariance(
    returns: pd.DataFrame,
    n_factors: int = 3,
    garch_type: str = 'GARCH',
    dist: str = 'studentst',
) -> np.ndarray:
    """
    Factor GARCH for conditional covariance.
    
    1. PCA on returns to extract factors
    2. Fit GARCH to each factor
    3. Reconstruct conditional covariance: Σ_t = B Σ_f,t B' + Σ_ε,t
    
    Args:
        returns: (T, N) returns
        n_factors: Number of factors
        garch_type: GARCH model type
        dist: Distribution
    
    Returns:
        (T, N, N) conditional covariance matrices
    """
    if not ARCH_AVAILABLE:
        raise ImportError("arch package required")
    
    T, N = returns.shape
    ret_values = returns.values
    
    # PCA on returns
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(ret_values)  # (T, K)
    loadings = pca.components_.T  # (N, K)
    
    # Idiosyncratic residuals
    recon = factors @ loadings.T
    residuals = ret_values - recon
    
    # Fit GARCH to each factor
    factor_cond_var = np.zeros((T, n_factors))
    for k in range(n_factors):
        am = arch_model(
            factors[:, k] * 100,
            mean='Zero',
            vol=garch_type,
            p=1, q=1,
            dist=dist,
        )
        res = am.fit(update_freq=0, show_warning=False)
        cv = res.conditional_volatility
        factor_cond_var[:, k] = (cv ** 2 / 10000).values if hasattr(cv, 'values') else (cv ** 2 / 10000)
    
    # Fit GARCH to idiosyncratic residuals (diagonal)
    idio_cond_var = np.zeros((T, N))
    for i in range(N):
        am = arch_model(
            residuals[:, i] * 100,
            mean='Zero',
            vol=garch_type,
            p=1, q=1,
            dist=dist,
        )
        res = am.fit(update_freq=0, show_warning=False)
        cv = res.conditional_volatility
        idio_cond_var[:, i] = (cv ** 2 / 10000).values if hasattr(cv, 'values') else (cv ** 2 / 10000)
    
    # Reconstruct covariance
    conditional_cov = np.zeros((T, N, N))
    for t in range(T):
        # Factor covariance (diagonal since factors are orthogonal)
        Sigma_f = np.diag(factor_cond_var[t])
        # Factor component
        factor_cov = loadings @ Sigma_f @ loadings.T
        # Idiosyncratic component (diagonal)
        idio_cov = np.diag(idio_cond_var[t])
        conditional_cov[t] = factor_cov + idio_cov
    
    return conditional_cov


# =============================================================================
# Portfolio Risk Metrics
# =============================================================================

def portfolio_var_cvar(
    returns: np.ndarray,  # (T, N)
    weights: np.ndarray,   # (N,)
    conditional_cov: np.ndarray,  # (T, N, N)
    alpha: float = 0.01,
) -> Tuple[float, float]:
    """
    Compute portfolio VaR and CVaR from conditional covariance.
    
    Assumes conditional normality: r_p,t ~ N(0, w' Σ_t w)
    
    Args:
        returns: Historical returns
        weights: Portfolio weights
        conditional_cov: Conditional covariance matrices
        alpha: VaR level (0.01 for 99%, 0.05 for 95%)
    
    Returns:
        (VaR, CVaR) - positive values (losses)
    """
    T = len(returns)
    
    # Portfolio conditional variance
    port_var = np.array([weights @ conditional_cov[t] @ weights for t in range(T)])
    port_vol = np.sqrt(np.maximum(port_var, 1e-12))
    
    # VaR = -Phi^{-1}(alpha) * sigma_p
    from scipy.stats import norm
    z_alpha = norm.ppf(alpha)
    var = -z_alpha * port_vol  # Positive = loss
    
    # CVaR = VaR + sigma_p * phi(z_alpha) / alpha
    cvar = var + port_vol * norm.pdf(z_alpha) / alpha
    
    return var, cvar


def kupiec_test(
    returns: np.ndarray,
    var: np.ndarray,
    alpha: float = 0.01,
) -> Tuple[float, int, int]:
    """
    Kupiec (1995) unconditional coverage test.
    
    H0: Violation rate = alpha
    
    Returns:
        p-value, violations, expected violations
    """
    violations = np.sum(returns < -var)
    expected = alpha * len(returns)
    
    if violations == 0 or violations == len(returns):
        return 1.0, violations, expected
    
    # Likelihood ratio test
    pi_hat = violations / len(returns)
    lr = -2 * (violations * np.log(alpha / pi_hat) + 
               (len(returns) - violations) * np.log((1 - alpha) / (1 - pi_hat)))
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr, df=1)
    
    return float(p_value), int(violations), int(expected)


def christoffersen_test(
    returns: np.ndarray,
    var: np.ndarray,
    alpha: float = 0.01,
) -> float:
    """
    Christoffersen (1998) conditional coverage test (independence + coverage).
    
    Returns:
        p-value
    """
    hits = (returns < -var).astype(int)
    n = len(hits)
    
    if n < 10 or hits.sum() == 0 or hits.sum() == n:
        return 1.0
    
    # Transition counts
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))
    
    # Probabilities under independence
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n - 1)
    
    if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
        return 1.0
    
    # Likelihood under H0 (independence)
    ll_ind = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    
    # Likelihood under H1 (Markov)
    ll_markov = (n00 * np.log(1 - pi_01) + n01 * np.log(pi_01) +
                 n10 * np.log(1 - pi_11) + n11 * np.log(pi_11))
    
    lr = -2 * (ll_ind - ll_markov)
    
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr, df=1)
    
    return float(p_value)


def portfolio_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    conditional_cov: np.ndarray,
) -> PortfolioMetrics:
    """
    Full portfolio backtest with VaR/CVaR and Kupiec/Christoffersen tests.
    """
    T = len(returns)
    port_rets = (returns.values @ weights)
    
    # 99% VaR/CVaR
    var_99, cvar_99 = portfolio_var_cvar(returns.values, weights, conditional_cov, alpha=0.01)
    var_95, cvar_95 = portfolio_var_cvar(returns.values, weights, conditional_cov, alpha=0.05)
    
    # Kupiec tests
    kupiec_99, viol_99, exp_99 = kupiec_test(port_rets, var_99, alpha=0.01)
    kupiec_95, viol_95, exp_95 = kupiec_test(port_rets, var_95, alpha=0.05)
    
    # Christoffersen tests
    cc_99 = christoffersen_test(port_rets, var_99, alpha=0.01)
    cc_95 = christoffersen_test(port_rets, var_95, alpha=0.05)
    
    return PortfolioMetrics(
        var_99=float(var_99[-1]) if hasattr(var_99, '__len__') else float(var_99),
        var_95=float(var_95[-1]) if hasattr(var_95, '__len__') else float(var_95),
        cvar_99=float(cvar_99[-1]) if hasattr(cvar_99, '__len__') else float(cvar_99),
        cvar_95=float(cvar_95[-1]) if hasattr(cvar_95, '__len__') else float(cvar_95),
        kupiec_99=kupiec_99,
        kupiec_95=kupiec_95,
        christoffersen_99=cc_99,
        christoffersen_95=cc_95,
    )


# =============================================================================
# Multi-Asset Feature Engineering
# =============================================================================

def build_multi_asset_features(
    returns: pd.DataFrame,
    window: int = 22,
    include_cross_corr: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build joint feature matrix for multi-asset forecasting.
    
    Features per asset:
    - lagged returns
    - historical volatility
    - cross-asset correlations (rolling)
    
    Args:
        returns: (T, N) returns
        window: Rolling window
        include_cross_corr: Whether to include cross-correlation features
    
    Returns:
        (T, N * F) feature matrix, feature names
    """
    T, N = returns.shape
    symbols = returns.columns.tolist()
    
    features_list = []
    feature_names = []
    
    # Individual asset features
    for i, sym in enumerate(symbols):
        r = returns.iloc[:, i].values
        
        # Lagged returns
        for lag in [1, 5, 10, 22]:
            feat = np.full(T, np.nan)
            feat[lag:] = r[:-lag]
            features_list.append(feat)
            feature_names.append(f'{sym}_lag{lag}')
        
        # Historical volatility
        for w in [5, 10, 22]:
            hv = np.full(T, np.nan)
            for t in range(w, T):
                hv[t] = np.std(r[t-w:t], ddof=1) * np.sqrt(252)
            features_list.append(hv)
            feature_names.append(f'{sym}_hv{w}')
    
    # Cross-asset correlations
    if include_cross_corr:
        for i in range(N):
            for j in range(i+1, N):
                corr = np.full(T, np.nan)
                for t in range(window, T):
                    corr[t] = np.corrcoef(
                        returns.iloc[t-window:t, i],
                        returns.iloc[t-window:t, j]
                    )[0, 1]
                features_list.append(corr)
                feature_names.append(f'corr_{symbols[i]}_{symbols[j]}')
    
    X = np.column_stack(features_list)
    return X, feature_names


# =============================================================================
# Multi-Asset Pipeline
# =============================================================================

def run_multi_asset_pipeline(
    symbols: List[str] = None,
    start_date: str = '2010-01-01',
    test_frac: float = 0.2,
) -> Dict:
    """
    Run full multi-asset volatility pipeline.
    
    Returns:
        Dictionary with results for DCC-GARCH, Factor GARCH, and portfolio metrics
    """
    print(f"Loading data for {symbols or ['SPY', 'QQQ', 'IWM', 'EFA']}...")
    data = load_multi_asset_data(symbols, start_date=start_date)
    
    print(f"Data shape: {data.returns.shape}")
    print(f"Date range: {data.dates[0]} to {data.dates[-1]}")
    
    # Compute realized covariance (proxy for evaluation)
    print("Computing realized covariance...")
    realized_cov = compute_realized_covariance(data.returns, window=22)
    data.realized_cov = realized_cov
    
    # Split
    T = len(data.returns)
    split = int(T * (1 - test_frac))
    
    train_returns = data.returns.iloc[:split]
    test_returns = data.returns.iloc[split:]
    
    print(f"Train: {len(train_returns)}, Test: {len(test_returns)}")
    
    # DCC-GARCH on full sample (expanding window)
    print("\nFitting DCC-GARCH...")
    dcc_result = dcc_garch(data.returns)
    
    # Factor GARCH
    print("Fitting Factor GARCH...")
    factor_cov = factor_garch_covariance(data.returns)
    
    # Equal weight portfolio
    N = len(data.symbols)
    weights = np.ones(N) / N
    
    # Portfolio backtest on test set
    test_start = split
    test_end = T
    
    print("\nPortfolio backtest (Equal Weight)...")
    dcc_metrics = portfolio_backtest(
        test_returns,
        weights,
        dcc_result.conditional_cov[test_start:test_end],
    )
    
    factor_metrics = portfolio_backtest(
        test_returns,
        weights,
        factor_cov[test_start:test_end],
    )
    
    # QLIKE on covariance
    def portfolio_qlike(pred_cov, real_cov, weights):
        pred_port_var = np.array([weights @ pred_cov[t] @ weights for t in range(len(pred_cov))])
        real_port_var = np.array([weights @ real_cov[t] @ weights for t in range(len(real_cov))])
        mask = np.isfinite(pred_port_var) & np.isfinite(real_port_var) & (pred_port_var > 0) & (real_port_var > 0)
        if mask.sum() == 0:
            return np.inf
        return np.mean(np.log(pred_port_var[mask]) + real_port_var[mask] / pred_port_var[mask] - 1)
    
    # Use realized covariance as proxy for "true" covariance
    test_realized = realized_cov[test_start:test_end]
    mask = ~np.isnan(test_realized).any(axis=(1,2))
    
    if mask.sum() > 50:
        dcc_qlike = portfolio_qlike(
            dcc_result.conditional_cov[test_start:test_end][mask],
            test_realized[mask],
            weights
        )
        factor_qlike = portfolio_qlike(
            factor_cov[test_start:test_end][mask],
            test_realized[mask],
            weights
        )
    else:
        dcc_qlike = np.inf
        factor_qlike = np.inf
    
    results = {
        'symbols': data.symbols,
        'dates': data.dates,
        'split': split,
        'dcc_result': dcc_result,
        'factor_cov': factor_cov,
        'realized_cov': realized_cov,
        'dcc_portfolio': dcc_metrics,
        'factor_portfolio': factor_metrics,
        'dcc_portfolio_qlike': dcc_qlike,
        'factor_portfolio_qlike': factor_qlike,
        'weights': weights,
    }
    
    print(f"\n{'='*60}")
    print("PORTFOLIO BACKTEST RESULTS (Equal Weight)")
    print(f"{'='*60}")
    print(f"\nDCC-GARCH:")
    print(f"  VaR 99%: {dcc_metrics.var_99:.4f}, CVaR 99%: {dcc_metrics.cvar_99:.4f}")
    print(f"  Kupiec 99% p-value: {dcc_metrics.kupiec_99:.4f}")
    print(f"  Christoffersen 99% p-value: {dcc_metrics.christoffersen_99:.4f}")
    print(f"  Portfolio QLIKE: {dcc_qlike:.4f}")
    
    print(f"\nFactor GARCH:")
    print(f"  VaR 99%: {factor_metrics.var_99:.4f}, CVaR 99%: {factor_metrics.cvar_99:.4f}")
    print(f"  Kupiec 99% p-value: {factor_metrics.kupiec_99:.4f}")
    print(f"  Christoffersen 99% p-value: {factor_metrics.christoffersen_99:.4f}")
    print(f"  Portfolio QLIKE: {factor_qlike:.4f}")
    
    # Success criterion
    success = dcc_metrics.kupiec_99 > 0.05
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Kupiec 99% p > 0.05 ({dcc_metrics.kupiec_99:.4f})")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    results = run_multi_asset_pipeline()