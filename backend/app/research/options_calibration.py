"""
ORACLE - Research: Options Calibration with Heston & Rough Volatility

Implements options chain data fetching and model calibration:
- SPY options chain from Yahoo Finance (yfinance) or Cboe DataShop
- Heston model calibration via FFT pricing (Lewis 2001)
- Rough Volatility (Bayer et al.) as alternative
- Vega-weighted RMSE metric for calibration quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import optimize, special, integrate
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class OptionChain:
    """Container for options chain data."""
    spot: float
    expiry: pd.Timestamp
    calls: pd.DataFrame  # columns: strike, bid, ask, mid, iv, volume, open_interest
    puts: pd.DataFrame
    risk_free_rate: float
    dividend_yield: float = 0.0


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # initial variance
    kappa: float   # mean reversion speed
    theta: float   # long-term variance
    sigma: float   # vol of vol
    rho: float     # correlation
    lambda_: float = 0.0  # risk premium (affine transform)


@dataclass
class CalibrationResult:
    """Calibration result."""
    params: HestonParams
    rmse: float
    vega_weighted_rmse: float
    max_error: float
    converged: bool
    market_prices: np.ndarray
    model_prices: np.ndarray
    strikes: np.ndarray
    maturities: np.ndarray


# =============================================================================
# Options Data Fetching
# =============================================================================

def fetch_spy_options_chain(
    expiry_days: int = 30,
    moneyness_range: Tuple[float, float] = (0.95, 1.05),
    min_volume: int = 100,
    use_cached: bool = True,
) -> OptionChain:
    """
    Fetch SPY options chain from Yahoo Finance.
    
    Args:
        expiry_days: Target days to expiry
        moneyness_range: (min, max) moneyness K/S
        min_volume: Minimum volume filter
        use_cached: Use cached data if available
    
    Returns:
        OptionChain object
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required for options data")
    
    spy = yf.Ticker("SPY")
    
    # Get spot price
    hist = spy.history(period="1d")
    spot = float(hist['Close'].iloc[-1])
    
    # Get available expiries
    expiries = spy.options
    if not expiries:
        raise ValueError("No options expiries available")
    
    # Find closest expiry to target days
    target_date = pd.Timestamp.now() + pd.Timedelta(days=expiry_days)
    expiry_dates = [pd.Timestamp(e) for e in expiries]
    closest_expiry = min(expiry_dates, key=lambda d: abs(d - target_date))
    
    # Fetch chain
    chain = spy.option_chain(closest_expiry.strftime('%Y-%m-%d'))
    
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    
    # Compute mid price
    calls['mid'] = (calls['bid'] + calls['ask']) / 2
    puts['mid'] = (puts['bid'] + puts['ask']) / 2
    
    # Filter by moneyness
    k_min, k_max = moneyness_range
    calls = calls[(calls['strike'] >= k_min * spot) & (calls['strike'] <= k_max * spot)]
    puts = puts[(puts['strike'] >= k_min * spot) & (puts['strike'] <= k_max * spot)]
    
    # Filter by volume
    calls = calls[calls['volume'] >= min_volume]
    puts = puts[puts['volume'] >= min_volume]
    
    # Risk-free rate (approximate from 3M treasury)
    r = 0.05  # Default, could fetch from FRED
    
    return OptionChain(
        spot=spot,
        expiry=closest_expiry,
        calls=calls,
        puts=puts,
        risk_free_rate=r,
    )


def fetch_multiple_expiries(
    min_days: int = 20,
    max_days: int = 60,
    moneyness_range: Tuple[float, float] = (0.9, 1.1),
    min_volume: int = 50,
) -> List[OptionChain]:
    """Fetch options chains for multiple expiries."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required")
    
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1d")
    spot = float(hist['Close'].iloc[-1])
    
    expiries = spy.options
    chains = []
    
    for exp_str in expiries:
        exp_date = pd.Timestamp(exp_str)
        days = (exp_date - pd.Timestamp.now()).days
        
        if min_days <= days <= max_days:
            try:
                chain = spy.option_chain(exp_str)
                
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                
                calls['mid'] = (calls['bid'] + calls['ask']) / 2
                puts['mid'] = (puts['bid'] + puts['ask']) / 2
                
                k_min, k_max = moneyness_range
                calls = calls[(calls['strike'] >= k_min * spot) & (calls['strike'] <= k_max * spot)]
                puts = puts[(puts['strike'] >= k_min * spot) & (puts['strike'] <= k_max * spot)]
                
                calls = calls[calls['volume'] >= min_volume]
                puts = puts[puts['volume'] >= min_volume]
                
                if len(calls) > 5 or len(puts) > 5:
                    chains.append(OptionChain(
                        spot=spot,
                        expiry=exp_date,
                        calls=calls,
                        puts=puts,
                        risk_free_rate=0.05,
                    ))
            except Exception as e:
                print(f"Failed to fetch {exp_str}: {e}")
    
    return chains


# =============================================================================
# Heston Model: Characteristic Function & FFT Pricing
# =============================================================================

def heston_char_func(
    u: np.ndarray,
    tau: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    q: float = 0.0,
    lambda_: float = 0.0,
    S: float = 1.0,
) -> np.ndarray:
    """
    Heston characteristic function (Lewis 2001 formulation).
    
    phi(u) = exp(C(u, tau) + D(u, tau) * v0 + i*u*log(S) + i*u*(r-q)*tau)
    
    Args:
        u: Complex frequency array
        tau: Time to maturity
        v0, kappa, theta, sigma, rho: Heston parameters
        r: Risk-free rate
        q: Dividend yield
        lambda_: Risk premium
        S: Spot price (needed for correct CF of log(S_T))
    
    Returns:
        Complex characteristic function values
    """
    # Complex sqrt and log
    i = complex(0, 1)
    
    # Adjusted parameters under risk-neutral measure
    kappa_tilde = kappa + lambda_
    
    d = np.sqrt((rho * sigma * i * u - kappa_tilde)**2 + sigma**2 * (i * u + u**2))
    g = (kappa_tilde - rho * sigma * i * u - d) / (kappa_tilde - rho * sigma * i * u + d)
    
    # Avoid division by zero
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    g = np.where(np.abs(g - 1) < 1e-12, 1 - 1e-12, g)
    
    C = (kappa_tilde * theta / sigma**2) * (
        (kappa_tilde - rho * sigma * i * u - d) * tau - 
        2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g))
    )
    
    D = ((kappa_tilde - rho * sigma * i * u - d) / sigma**2) * (
        (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))
    )
    
    # CF of log(S_T): includes log(S) term
    return np.exp(C + D * v0 + i * u * (np.log(S) + (r - q) * tau))


def heston_price_fft(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    option_type: str = 'call',
    N: int = 4096,
    eta: float = 0.25,
    alpha: float = 1.5,
) -> float:
    """
    Heston option price via FFT (Carr-Madan 1999).
    
    Args:
        S: Spot price
        K: Strike
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        params: HestonParams
        option_type: 'call' or 'put'
        N: Number of FFT points (power of 2)
        eta: Grid spacing in log-strike space
        alpha: Damping factor
    
    Returns:
        Option price
    """
    # FFT grid
    lamb = 2 * np.pi / (N * eta)
    b = N * lamb / 2
    u = np.arange(N) * eta
    
    # Characteristic function at u - i*(alpha+1)
    u_cf = u - 1j * (alpha + 1)
    
    cf = heston_char_func(
        u_cf, T, params.v0, params.kappa, params.theta,
        params.sigma, params.rho, r, q, params.lambda_, S
    )
    
    # Integrand for call price: 
    # psi(v) = exp(-rT) * phi(v - i(alpha+1)) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)
    integrand = np.exp(-r * T) * cf / (alpha**2 + alpha - u**2 + 1j * (2*alpha + 1) * u)
    
    # Simpson's rule weights
    w = np.ones(N)
    w[1:N:2] = 4
    w[2:N-1:2] = 2
    w = w * eta / 3
    
    # Carr-Madan requires (-1)^j factor for correct FFT indexing
    j = np.arange(N)
    integrand = integrand * ((-1) ** j)
    
    # FFT
    x = np.fft.fft(w * integrand).real
    
    # Log-strike grid
    k = -b + np.arange(N) * lamb
    
    # Interpolate at log(K)
    logK = np.log(K)
    idx = int(np.round((logK + b) / lamb))
    idx = np.clip(idx, 0, N - 1)
    
    # Price: C = exp(-alpha*k) / pi * Re[FFT] * lamb * S / 2
    # (the lamb/2 factor from the Riemann sum)
    call_price = np.exp(-alpha * logK) / np.pi * x[idx] * lamb * S / 2
    
    if option_type == 'call':
        return max(call_price, 0.0)
    else:
        # Put-call parity
        put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
        return max(put_price, 0.0)


def heston_price_batch(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    option_types: np.ndarray,
) -> np.ndarray:
    """Vectorized Heston pricing for multiple strikes."""
    prices = np.zeros_like(strikes)
    for i, (K, opt_type) in enumerate(zip(strikes, option_types)):
        prices[i] = heston_price_fft(S, K, T, r, q, params, opt_type)
    return prices


# =============================================================================
# Heston Calibration
# =============================================================================

def calibrate_heston(
    chain: OptionChain,
    initial_guess: HestonParams = None,
    bounds: List[Tuple[float, float]] = None,
    use_vega_weighting: bool = True,
) -> CalibrationResult:
    """
    Calibrate Heston model to options chain.
    
    Minimizes squared price error (optionally vega-weighted).
    
    Args:
        chain: OptionChain data
        initial_guess: Starting parameters
        bounds: Parameter bounds
        use_vega_weighting: Weight by vega for better ATM fit
    
    Returns:
        CalibrationResult
    """
    if initial_guess is None:
        initial_guess = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
        )
    
    if bounds is None:
        bounds = [
            (0.001, 1.0),    # v0
            (0.1, 20.0),     # kappa
            (0.001, 1.0),    # theta
            (0.01, 2.0),     # sigma
            (-0.99, 0.99),   # rho
        ]
    
    # Prepare market data
    S = chain.spot
    r = chain.risk_free_rate
    q = chain.dividend_yield
    tau = (chain.expiry - pd.Timestamp.now()).days / 252.0
    
    # Combine calls and puts
    strikes = []
    market_prices = []
    option_types = []
    vegas = []
    
    for _, row in chain.calls.iterrows():
        if row['mid'] > 0:
            strikes.append(row['strike'])
            market_prices.append(row['mid'])
            option_types.append('call')
            # Approximate vega (Black-Scholes)
            iv = row.get('impliedVolatility', row.get('iv', 0.2))
            vegas.append(approx_vega(S, row['strike'], tau, r, q, iv / 100))
    
    for _, row in chain.puts.iterrows():
        if row['mid'] > 0:
            strikes.append(row['strike'])
            market_prices.append(row['mid'])
            option_types.append('put')
            iv = row.get('impliedVolatility', row.get('iv', 0.2))
            vegas.append(approx_vega(S, row['strike'], tau, r, q, iv / 100))
    
    strikes = np.array(strikes)
    market_prices = np.array(market_prices)
    option_types = np.array(option_types)
    vegas = np.array(vegas)
    
    if len(strikes) < 4:
        raise ValueError("Insufficient options for calibration")
    
    # Vega weights
    if use_vega_weighting:
        weights = vegas / np.sum(vegas)
    else:
        weights = np.ones_like(market_prices) / len(market_prices)
    
    def objective(x):
        params = HestonParams(*x)
        model_prices = heston_price_batch(S, strikes, tau, r, q, params, option_types)
        
        # Relative squared error
        errors = (model_prices - market_prices) / (market_prices + 1e-4)
        
        # Weighted MSE
        loss = np.sum(weights * errors**2)
        
        # Penalty for negative prices or arbitrage
        if np.any(model_prices < 0):
            loss += 1e6
        
        return loss
    
    # Run optimization
    result = optimize.minimize(
        objective,
        x0=[initial_guess.v0, initial_guess.kappa, initial_guess.theta,
            initial_guess.sigma, initial_guess.rho],
        bounds=bounds,
        method='L-BFGS-B',
        options={'ftol': 1e-8, 'maxiter': 200}
    )
    
    best_params = HestonParams(*result.x)
    final_prices = heston_price_batch(S, strikes, tau, r, q, best_params, option_types)
    
    # Compute metrics
    errors = final_prices - market_prices
    rmse = np.sqrt(np.mean(errors**2))
    vega_w_rmse = np.sqrt(np.sum(weights * errors**2))
    max_err = np.max(np.abs(errors))
    
    return CalibrationResult(
        params=best_params,
        rmse=float(rmse),
        vega_weighted_rmse=float(vega_w_rmse),
        max_error=float(max_err),
        converged=result.success,
        market_prices=market_prices,
        model_prices=final_prices,
        strikes=strikes,
        maturities=np.full_like(strikes, tau),
    )


def approx_vega(S: float, K: float, T: float, r: float, q: float, vol: float) -> float:
    """Approximate Black-Scholes vega."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)


# =============================================================================
# Rough Volatility (Bayer et al. 2016) - Neural SDE Approximation
# =============================================================================

class RoughVolModel:
    """
    Rough Volatility model using the rBergomi approximation.
    
    dS_t = S_t * sqrt(v_t) dW_t^1
    v_t = xi_0(t) * exp(eta * W_t^H - 0.5 * eta^2 * t^{2H})
    
    Where H < 0.5 gives rough volatility.
    """
    
    def __init__(self, H: float = 0.1, eta: float = 1.5, rho: float = -0.7):
        self.H = H
        self.eta = eta
        self.rho = rho
    
    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        xi_0: Callable[[float], float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rough volatility paths using hybrid scheme.
        
        Returns:
            (S_paths, v_paths) each of shape (n_paths, n_steps+1)
        """
        if xi_0 is None:
            xi_0 = lambda t: 0.04  # Constant forward variance
        
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)
        
        # Generate fractional Brownian motion using circulant embedding
        # Simplified: use Riemann sum approximation
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        
        S_paths[:, 0] = S0
        v_paths[:, 0] = xi_0(0)
        
        for i in range(n_steps):
            # Volatility process
            dW_H = np.random.randn(n_paths) * dt**self.H
            v_paths[:, i+1] = xi_0(t[i+1]) * np.exp(
                self.eta * dW_H.cumsum() - 0.5 * self.eta**2 * t[i+1]**(2*self.H)
            )
            
            # Spot process
            dW1 = np.random.randn(n_paths) * np.sqrt(dt)
            dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.randn(n_paths) * np.sqrt(dt)
            S_paths[:, i+1] = S_paths[:, i] * np.exp(
                -0.5 * v_paths[:, i] * dt + np.sqrt(v_paths[:, i]) * dW1
            )
        
        return S_paths, v_paths
    
    def price_option_mc(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        n_paths: int = 50000,
        n_steps: int = 100,
    ) -> float:
        """Monte Carlo option pricing."""
        S_paths, _ = self.simulate_paths(S0, T, n_steps, n_paths)
        
        payoff = np.maximum(S_paths[:, -1] - K, 0)
        price = np.exp(-r * T) * np.mean(payoff)
        
        return float(price)
    
    def calibrate(
        self,
        chain: OptionChain,
        param_bounds: Dict = None,
    ) -> Dict:
        """Calibrate rough vol parameters to options."""
        if param_bounds is None:
            param_bounds = {
                'H': (0.05, 0.45),
                'eta': (0.5, 3.0),
                'rho': (-0.99, -0.1),
            }
        
        # Use ATM options for calibration
        S = chain.spot
        r = chain.risk_free_rate
        q = chain.dividend_yield
        tau = (chain.expiry - pd.Timestamp.now()).days / 252.0
        
        # Select ATM options
        calls = chain.calls[chain.calls['mid'] > 0].copy()
        puts = chain.puts[chain.puts['mid'] > 0].copy()
        
        moneyness_calls = calls['strike'] / S
        moneyness_puts = puts['strike'] / S
        
        atm_calls = calls[(moneyness_calls >= 0.95) & (moneyness_calls <= 1.05)]
        atm_puts = puts[(moneyness_puts >= 0.95) & (moneyness_puts <= 1.05)]
        
        market_data = []
        for _, row in atm_calls.iterrows():
            market_data.append((row['strike'], row['mid'], 'call'))
        for _, row in atm_puts.iterrows():
            market_data.append((row['strike'], row['mid'], 'put'))
        
        if len(market_data) < 3:
            raise ValueError("Insufficient ATM options")
        
        def objective(params):
            H, eta, rho = params
            if H <= 0 or H >= 0.5 or eta <= 0 or rho <= -1 or rho >= 1:
                return 1e9
            
            self.H = H
            self.eta = eta
            self.rho = rho
            
            errors = []
            for K, mkt_price, opt_type in market_data:
                try:
                    if opt_type == 'call':
                        model_price = self.price_option_mc(S, K, tau, r, q, n_paths=20000, n_steps=50)
                    else:
                        # Put-call parity
                        call_price = self.price_option_mc(S, K, tau, r, q, n_paths=20000, n_steps=50)
                        model_price = call_price - S * np.exp(-q*tau) + K * np.exp(-r*tau)
                    
                    errors.append((model_price - mkt_price) / (mkt_price + 1e-4))
                except:
                    errors.append(1e3)
            
            return np.mean(np.array(errors)**2)
        
        result = optimize.minimize(
            objective,
            x0=[self.H, self.eta, self.rho],
            bounds=[param_bounds['H'], param_bounds['eta'], param_bounds['rho']],
            method='L-BFGS-B',
            options={'maxiter': 50}
        )
        
        return {
            'H': result.x[0],
            'eta': result.x[1],
            'rho': result.x[2],
            'success': result.success,
            'loss': result.fun,
        }


# =============================================================================
# Implied Volatility Surface Fitting
# =============================================================================

def fit_iv_surface(
    chain: OptionChain,
    degree: int = 2,
) -> Tuple[Callable, np.ndarray]:
    """
    Fit polynomial IV surface: IV(K, T) = sum a_ij * m^i * tau^j
    where m = log(K/S) is log-moneyness.
    
    Returns:
        (iv_func, coefficients)
    """
    S = chain.spot
    tau = (chain.expiry - pd.Timestamp.now()).days / 252.0
    
    # Collect data
    moneyness = []
    ivs = []
    
    for _, row in chain.calls.iterrows():
        if row['iv'] > 0 and row['mid'] > 0:
            m = np.log(row['strike'] / S)
            moneyness.append(m)
            ivs.append(row['iv'] / 100)
    
    for _, row in chain.puts.iterrows():
        if row['iv'] > 0 and row['mid'] > 0:
            m = np.log(row['strike'] / S)
            moneyness.append(m)
            ivs.append(row['iv'] / 100)
    
    moneyness = np.array(moneyness)
    ivs = np.array(ivs)
    
    # Polynomial features
    X = np.column_stack([
        np.ones_like(moneyness),
        moneyness,
        moneyness**2,
        moneyness**3,
    ][:degree+1])
    
    coeffs, *_ = np.linalg.lstsq(X, ivs, rcond=None)
    
    def iv_func(K: float) -> float:
        m = np.log(K / S)
        x = np.array([1, m, m**2, m**3][:degree+1])
        return np.dot(x, coeffs)
    
    return iv_func, coeffs


# =============================================================================
# Calibration Pipeline
# =============================================================================

def run_options_calibration(
    expiry_days: int = 30,
    moneyness_range: Tuple[float, float] = (0.95, 1.05),
    min_volume: int = 100,
) -> Dict:
    """
    Run full options calibration pipeline.
    
    Returns:
        Dictionary with Heston and Rough Vol results
    """
    print(f"Fetching SPY options chain (expiry ~{expiry_days}d)...")
    chain = fetch_spy_options_chain(expiry_days, moneyness_range, min_volume)
    
    print(f"Spot: {chain.spot:.2f}")
    print(f"Expiry: {chain.expiry.date()}")
    print(f"Calls: {len(chain.calls)}, Puts: {len(chain.puts)}")
    print(f"Rate: {chain.risk_free_rate:.2%}")
    
    # Heston calibration
    print("\nCalibrating Heston model...")
    heston_result = calibrate_heston(chain)
    
    print(f"Heston params:")
    print(f"  v0={heston_result.params.v0:.4f}, kappa={heston_result.params.kappa:.4f}")
    print(f"  theta={heston_result.params.theta:.4f}, sigma={heston_result.params.sigma:.4f}")
    print(f"  rho={heston_result.params.rho:.4f}")
    print(f"  RMSE: {heston_result.rmse:.4f}")
    print(f"  Vega-weighted RMSE: {heston_result.vega_weighted_rmse:.4f}")
    print(f"  Max error: {heston_result.max_error:.4f}")
    print(f"  Converged: {heston_result.converged}")
    
    # Rough Vol calibration
    print("\nCalibrating Rough Volatility model...")
    rough_model = RoughVolModel()
    rough_result = rough_model.calibrate(chain)
    
    print(f"Rough Vol params:")
    print(f"  H={rough_result['H']:.4f}, eta={rough_result['eta']:.4f}, rho={rough_result['rho']:.4f}")
    print(f"  Loss: {rough_result['loss']:.6f}")
    print(f"  Success: {rough_result['success']}")
    
    # IV surface fit
    print("\nFitting IV surface...")
    iv_func, coeffs = fit_iv_surface(chain)
    print(f"IV surface coefficients: {coeffs}")
    
    # Error metrics
    strikes = np.concatenate([chain.calls['strike'].values, chain.puts['strike'].values])
    ivs_market = np.concatenate([chain.calls['iv'].values, chain.puts['iv'].values]) / 100
    ivs_model = np.array([iv_func(K) for K in strikes])
    iv_rmse = np.sqrt(np.mean((ivs_model - ivs_market)**2))
    print(f"IV surface RMSE: {iv_rmse:.4f}")
    
    results = {
        'chain': chain,
        'heston': heston_result,
        'rough_vol': rough_result,
        'iv_surface': {'func': iv_func, 'coeffs': coeffs},
        'iv_rmse': iv_rmse,
    }
    
    # Success criterion: Option RMSE < 2% of spot
    spot = chain.spot
    success = heston_result.rmse < 0.02 * spot
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Heston RMSE < 2% spot ({heston_result.rmse:.4f} < {0.02*spot:.4f})")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    results = run_options_calibration()