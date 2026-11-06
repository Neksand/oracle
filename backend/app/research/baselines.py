#!/usr/bin/env python3
"""
SOTA Baselines for Realized Variance Forecasting — ORACLE project.

Implements classical econometric baselines for RV forecasting:
  - HAR-RV (Hybrid Autoregressive, Corsi 2009)
  - GARCH(1,1) (Bollerslev 1986)
  - EGARCH(1,1) (Nelson 1991)
  - RVAR (Realized Variance Autoregressive, order 1)

All models predict daily RV from lagged RV/daily variance series.
Used for comparison against deep learning models (LSTM, VolFormer, Mamba, Diffusion).

References:
  - Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
    Journal of Financial Econometrics.
  - Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
    Journal of Econometrics.
  - Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns.
    Econometrica.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# Additional model dependencies - use sklearn's RandomForest directly
try:
    from sklearn.ensemble import RandomForestRegressor as SkRandomForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATS_MODELS = True
except ImportError:
    HAS_STATS_MODELS = False


# ============================================================================
# HAR (Hybrid Autoregressive) Model — Corsi (2009)
# ============================================================================

class HARModel:
    """
    HAR-RV model: Realized Volatility forecasting using daily, weekly,
    and monthly lagged RV components.

    RV_{t+1} = beta_0 + beta_d * RV_t + beta_w * RV_t^(w) + beta_m * RV_t^(m) + eps_t

    where RV_t^(w) = (1/5) sum_{i=1..5} RV_{t-i+1} (weekly)
          RV_t^(m) = (1/22) sum_{i=1..22} RV_{t-i+1} (monthly)
    """

    def __init__(self, lags_daily: int = 1, lags_weekly: int = 5, lags_monthly: int = 22):
        self.lags_daily = lags_daily
        self.lags_weekly = lags_weekly
        self.lags_monthly = lags_monthly
        self.coef_daily = 0.0
        self.coef_weekly = 0.0
        self.coef_monthly = 0.0
        self.intercept = 0.0
        self.fitted = False

    def _compute_lags(self, rv_series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute daily, weekly, monthly lagged RV for each time step."""
        n = len(rv_series)
        daily = np.full(n, np.nan)
        weekly = np.full(n, np.nan)
        monthly = np.full(n, np.nan)

        for t in range(n):
            if t >= self.lags_daily:
                daily[t] = rv_series[t - 1]
            if t >= self.lags_weekly:
                weekly[t] = np.mean(rv_series[t - self.lags_weekly:t])
            if t >= self.lags_monthly:
                monthly[t] = np.mean(rv_series[t - self.lags_monthly:t])

        return daily, weekly, monthly

    def fit(self, rv_series: np.ndarray) -> dict:
        """Fit HAR model via OLS."""
        n = len(rv_series)
        if n < self.lags_monthly + 10:
            raise ValueError(f"Series too short for HAR: n={n}, need >{self.lags_monthly + 10}")

        daily, weekly, monthly = self._compute_lags(rv_series)

        # Target: next-day RV
        y = rv_series[1:]
        X_daily = daily[:-1]
        X_weekly = weekly[:-1]
        X_monthly = monthly[:-1]

        # Valid mask
        valid = np.isfinite(X_daily) & np.isfinite(X_weekly) & np.isfinite(X_monthly) & np.isfinite(y)
        if valid.sum() < 10:
            raise ValueError("Too few valid samples for HAR fit")

        X = np.column_stack([np.ones(valid.sum()), X_daily[valid], X_weekly[valid], X_monthly[valid]])
        yv = y[valid]

        # OLS
        beta, residuals, rank, s = np.linalg.lstsq(X, yv, rcond=None)

        self.intercept = beta[0]
        self.coef_daily = beta[1]
        self.coef_weekly = beta[2]
        self.coef_monthly = beta[3]
        self.fitted = True

        # In-sample predictions
        y_pred = X @ beta
        rmse = float(np.sqrt(np.mean((yv - y_pred) ** 2)))
        mae = float(np.mean(np.abs(yv - y_pred)))

        return {
            "intercept": self.intercept,
            "coef_daily": self.coef_daily,
            "coef_weekly": self.coef_weekly,
            "coef_monthly": self.coef_monthly,
            "n_samples": int(valid.sum()),
            "rmse": rmse,
            "mae": mae,
        }

    def predict(self, rv_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Predict next n_steps RV values (walk-forward style).

        Returns array of predictions aligned with rv_series[1:].
        """
        if not self.fitted:
            raise RuntimeError("HAR model not fitted. Call fit() first.")

        n = len(rv_series)
        predictions = np.full(n - 1, np.nan)

        for t in range(1, n):
            # Use data up to t-1 to predict rv[t]
            hist = rv_series[:t]
            daily = hist[-1] if len(hist) >= self.lags_daily else np.mean(hist)
            weekly = np.mean(hist[-self.lags_weekly:]) if len(hist) >= self.lags_weekly else np.mean(hist)
            monthly = np.mean(hist[-self.lags_monthly:]) if len(hist) >= self.lags_monthly else np.mean(hist)

            predictions[t - 1] = (self.intercept + self.coef_daily * daily
                                  + self.coef_weekly * weekly
                                  + self.coef_monthly * monthly)

        return predictions


# ============================================================================
# RVAR (Realized Variance Autoregressive) Model
# ============================================================================

class RVARModel:
    """
    RVAR(p) model: Realized Variance Autoregressive.

    RV_t = c + phi_1 * RV_{t-1} + ... + phi_p * RV_{t-p} + eps_t

    Simple autoregressive model on realized variance.
    """

    def __init__(self, order: int = 1):
        self.order = order
        self.coefs = None
        self.intercept = 0.0
        self.fitted = False

    def fit(self, rv_series: np.ndarray) -> dict:
        """Fit RVAR via OLS."""
        n = len(rv_series)
        if n < self.order + 10:
            raise ValueError(f"Series too short for RVAR: n={n}, need >{self.order + 10}")

        # Build design matrix
        X_cols = [np.ones(n - self.order)]
        for lag in range(1, self.order + 1):
            X_cols.append(rv_series[lag:lag + (n - self.order)])

        X = np.column_stack(X_cols)
        y = rv_series[self.order:]

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.intercept = beta[0]
        self.coefs = beta[1:]
        self.fitted = True

        y_pred = X @ beta
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y - y_pred)))

        return {
            "intercept": self.intercept,
            "coefs": self.coefs.tolist(),
            "order": self.order,
            "n_samples": int(len(y)),
            "rmse": rmse,
            "mae": mae,
        }

    def predict(self, rv_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Predict next n_steps RV values (walk-forward style).

        Returns array of predictions aligned with rv_series[order:].
        """
        if not self.fitted:
            raise RuntimeError("RVAR model not fitted. Call fit() first.")

        n = len(rv_series)
        predictions = np.full(n - self.order, np.nan)

        for t in range(self.order, n):
            hist = rv_series[:t]
            pred = self.intercept
            for lag, coef in enumerate(self.coefs, start=1):
                pred += coef * hist[-lag]
            predictions[t - self.order] = pred

        return predictions


# ============================================================================
# GARCH(1,1) Model — Bollerslev (1986)
# ============================================================================

class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.

    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    Uses arch package for fitting. Forecasts conditional variance.
    """

    def __init__(self, p: int = 1, q: int = 1, vol: str = "GARCH"):
        self.p = p
        self.q = q
        self.vol = vol
        self.model = None
        self.fitted = False
        self.last_variance = 0.0
        self.last_resid = 0.0

    def fit(self, returns: np.ndarray) -> dict:
        """Fit GARCH model to return series."""
        if not HAS_ARCH:
            raise ImportError("arch package required for GARCH. Install: pip install arch")

        n = len(returns)
        if n < 100:
            raise ValueError(f"Series too short for GARCH: n={n}, need >= 100")

        # Convert to percentage returns (arch expects %)
        returns_pct = returns * 100

        am = arch_model(returns_pct, vol=self.vol, p=self.p, q=self.q, dist="normal")
        self.model = am.fit(disp="off", show_warning=False)
        self.fitted = True

        # Extract last variance and residual for forecasting
        params = self.model.params
        self.last_variance = float(self.model.conditional_volatility[-1] ** 2 / 10000)  # back to decimal
        self.last_resid = float(returns[-1])

        # In-sample RMSE (conditional std forecast vs realized)
        cond_vol = self.model.conditional_volatility / 100  # back to decimal
        realized = np.abs(returns)
        rmse = float(np.sqrt(np.mean((cond_vol - realized) ** 2)))
        mae = float(np.mean(np.abs(cond_vol - realized)))

        return {
            "omega": float(params.get("omega", 0)),
            "alpha": float(params.get("alpha[1]", 0)),
            "beta": float(params.get("beta[1]", 0)),
            "n_samples": int(n),
            "loglik": float(self.model.loglikelihood),
            "aic": float(self.model.aic),
            "bic": float(self.model.bic),
            "rmse": rmse,
            "mae": mae,
        }

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        """Forecast variance for horizon steps ahead."""
        if not self.fitted:
            raise RuntimeError("GARCH model not fitted. Call fit() first.")

        if self.model is None:
            raise RuntimeError("GARCH model not fitted.")

        # Use arch's built-in forecast
        forecast = self.model.forecast(horizon=horizon)
        var_forecast = forecast.variance.values[-1, :] / 10000  # back to decimal
        return var_forecast


# ============================================================================
# EGARCH(1,1) Model — Nelson (1991)
# ============================================================================

class EGARCHModel:
    """
    EGARCH(1,1) model: Exponential GARCH allowing asymmetric volatility
    response to positive/negative shocks.

    log(sigma_t^2) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma_{t-1}^2)
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted = False

    def fit(self, returns: np.ndarray) -> dict:
        """Fit EGARCH model to return series."""
        if not HAS_ARCH:
            raise ImportError("arch package required for EGARCH. Install: pip install arch")

        n = len(returns)
        if n < 100:
            raise ValueError(f"Series too short for EGARCH: n={n}, need >= 100")

        returns_pct = returns * 100

        am = arch_model(returns_pct, vol="EGARCH", p=self.p, q=self.q, dist="normal")
        self.model = am.fit(disp="off", show_warning=False)
        self.fitted = True

        cond_vol = self.model.conditional_volatility / 100
        realized = np.abs(returns)
        rmse = float(np.sqrt(np.mean((cond_vol - realized) ** 2)))
        mae = float(np.mean(np.abs(cond_vol - realized)))

        return {
            "n_samples": int(n),
            "loglik": float(self.model.loglikelihood),
            "aic": float(self.model.aic),
            "bic": float(self.model.bic),
            "rmse": rmse,
            "mae": mae,
        }

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        """Forecast variance for horizon steps ahead."""
        if not self.fitted:
            raise RuntimeError("EGARCH model not fitted.")
        forecast = self.model.forecast(horizon=horizon)
        var_forecast = forecast.variance.values[-1, :] / 10000
        return var_forecast


# ============================================================================
# Random Forest Regressor Wrapper
# ============================================================================

class RandomForestRegressor:
    """
    Random Forest Regression model for RV forecasting.

    Ensemble method using multiple decision trees, robust to overfitting
    and capable of capturing non-linear relationships in RV dynamics.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.fitted = False

    def fit(self, rv_series: np.ndarray, returns: Optional[np.ndarray] = None) -> dict:
        """Fit Random Forest model to predict RV."""
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for RandomForest. Install: pip install scikit-learn")

        n = len(rv_series)
        if n < 10:
            split_idx = max(5, int(0.8 * n))
        else:
            split_idx = int(0.8 * n)

        # Create lagged features - use single lag for simplicity
        max_lag = 1
        X = []
        y = []
        for t in range(max_lag, n):
            features = [rv_series[t - lag] for lag in range(1, max_lag + 1)]
            X.append(features)
            y.append(rv_series[t])

        X = np.array(X)
        y = np.array(y)

        # Simple train/test split
        split_idx = int(0.8 * len(X))
        if split_idx < 1:
            split_idx = 1

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Handle edge cases
        if len(X_test) == 0:
            X_test = X_train
            y_test = y_train

        # Fit sklearn's RandomForest
        self.model = SkRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self.fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        if len(y_test) > 0:
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            if np.var(y_test) > 0:
                r2 = float(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            else:
                r2 = float('nan')
        else:
            rmse = mae = r2 = float('nan')

        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "n_samples": int(len(X)),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def predict(self, rv_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Predict next n_steps RV values using walk-forward forecasting."""
        if not self.fitted:
            raise RuntimeError("Random Forest model not fitted. Call fit() first.")

        n = len(rv_series)
        predictions = np.full(n - 1, np.nan)

        # Use data up to current point for prediction
        for t in range(1, n):
            if t < 1:  # Need at least one lag
                predictions[t - 1] = rv_series[t]
                continue

            # Create features from available history
            lag_features = min(10, t)
            X = []
            for lag in range(1, lag_features + 1):
                if t - lag >= 0:
                    X.append(rv_series[t - lag])

            if len(X) < 1:  # Need at least one feature
                predictions[t - 1] = rv_series[t]
                continue

            X = np.array(X).reshape(1, -1)
            pred = self.model.predict(X)[0]
            predictions[t - 1] = pred

        return predictions


# ============================================================================
# XGBoost Regressor
# ============================================================================

class XGBoostRegressor:
    """
    XGBoost regression model for RV forecasting.

    Gradient boosting framework that builds sequential decision trees,
    handling complex non-linear relationships and interactions effectively.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1,
                 subsample: float = 0.8, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.model = None
        self.fitted = False

    def fit(self, rv_series: np.ndarray, returns: Optional[np.ndarray] = None) -> dict:
        """Fit XGBoost model to predict RV."""
        if not HAS_XGBOOST:
            raise ImportError("xgboost required for XGBoost. Install: pip install xgboost")

        n = len(rv_series)
        if n < 10:
            # Use available data if too short
            split_idx = max(5, int(0.8 * n))
        else:
            split_idx = int(0.8 * n)

        # Create lagged features - use fewer lags for smaller datasets
        max_lag = min(3, n // 2 - 1)
        if max_lag < 1:
            max_lag = 1

        X = []
        y = []
        for t in range(max_lag, n):
            features = []
            for lag in range(1, max_lag + 1):
                features.append(rv_series[t - lag])
            X.append(features)
            y.append(rv_series[t])

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Split for validation - ensure we have training data
        if len(X) < 10:
            split_idx = 1
        else:
            split_idx = int(0.8 * len(X))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Handle case where we don't have enough test data
        if len(X_test) < 1:
            # Use all data for training if test set would be too small
            X_train, y_train = X, y
            X_test, y_test = np.array([]).reshape(0, max_lag), np.array([])

        # Fit XGBoost with proper parameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            objective='reg:squarederror',
            n_jobs=1
        )
        self.model.fit(X_train, y_train)
        self.fitted = True

        # Evaluate
        if len(y_test) > 0:
            y_pred = self.model.predict(X_test)
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            if np.var(y_test) > 0:
                r2 = float(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            else:
                r2 = float('nan')
        else:
            rmse = mae = r2 = float('nan')

        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "n_samples": int(len(X)),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def predict(self, rv_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Predict next n_steps RV values using walk-forward forecasting."""
        if not self.fitted:
            raise RuntimeError("XGBoost model not fitted. Call fit() first.")

        n = len(rv_series)
        predictions = np.full(n - 1, np.nan)

        for t in range(1, n):
            if t < 1:
                predictions[t - 1] = rv_series[t]
                continue

            lag_features = min(10, t)
            X = []
            for lag in range(1, lag_features + 1):
                if t - lag >= 0:
                    X.append(rv_series[t - lag])

            if len(X) < 1:
                predictions[t - 1] = rv_series[t]
                continue

            X = np.array(X).reshape(1, -1)
            pred = self.model.predict(X)[0]
            predictions[t - 1] = pred

        return predictions


# ============================================================================
# ARIMA (AutoRegressive Integrated Moving Average) Model
# ============================================================================

class ARIMAModel:
    """
    ARIMA model for RV forecasting.

    Statistical time series model that combines autoregressive, differencing,
    and moving average components for capturing linear time dependencies.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted = False
        self.predictions = None

    def fit(self, rv_series: np.ndarray, returns: Optional[np.ndarray] = None) -> dict:
        """Fit ARIMA model to RV series."""
        if not HAS_STATS_MODELS:
            raise ImportError("statsmodels required for ARIMA. Install: pip install statsmodels")

        n = len(rv_series)
        if n < 30:
            split_idx = max(10, int(0.8 * n))
        else:
            split_idx = int(0.8 * n)

        # Create training and test sets
        train = rv_series[:split_idx]
        test = rv_series[split_idx:]

        # Fit ARIMA model - handle statsmodels API changes
        try:
            # Try newer statsmodels API (2.0+)
            self.model = ARIMA(train, order=self.order)
            self.model_fit = self.model.fit()
            print("ARIMA using newer statsmodels API")
        except Exception as e1:
            try:
                # Fallback to older API
                self.model = ARIMA(train, order=self.order)
                self.model_fit = self.model.fit(disp=False)
                print("ARIMA using older statsmodels API")
            except Exception as e2:
                print(f"ARIMA fit failed with both APIs: {e2}")
                # Try even older API
                try:
                    from statsmodels.tsa.arima_model import ARIMA as ARIMAOri
                    self.model = ARIMAOri(train, order=self.order)
                    self.model_fit = self.model.fit(disp=False)
                    print("ARIMA using very old statsmodels API")
                except Exception as e3:
                    raise ImportError(f"ARIMA fit failed: {e3}")

        self.fitted = True

        # In-sample predictions (training set)
        try:
            train_pred = self.model_fit.predict(start=1, end=len(train) - 1)
        except Exception:
            train_pred = np.full(len(train) - 1, np.nan)

        # Out-of-sample predictions (test set)
        if len(test) > 0:
            try:
                test_pred = self.model_fit.forecast(steps=len(test))
                y_true = np.concatenate([train_pred, test])
                y_pred = np.concatenate([train_pred, test_pred])
            except Exception:
                y_true = np.concatenate([train_pred, test])
                y_pred = np.full(len(train) - 1 + len(test), np.nan)
        else:
            y_true = train_pred
            y_pred = train_pred

        # Calculate metrics
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))
            mae = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
        else:
            rmse = mae = float('inf')

        # AIC and BIC from model fit
        try:
            aic = float(self.model_fit.aic)
            bic = float(self.model_fit.bic)
        except Exception:
            aic = bic = float('inf')

        return {
            "order": self.order,
            "n_samples": int(n),
            "rmse": rmse,
            "mae": mae,
            "aic": aic,
            "bic": bic,
        }

    def predict(self, rv_series: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Predict next n_steps RV values using rolling forecast."""
        if not self.fitted:
            raise RuntimeError("ARIMA model not fitted. Call fit() first.")

        n = len(rv_series)
        predictions = np.full(n - 1, np.nan)

        for t in range(1, n):
            # Use expanding window for prediction
            train_data = rv_series[:t]

            if len(train_data) >= self.order[0] + 10:
                # Refit ARIMA on expanding window
                temp_model = ARIMA(train_data, order=self.order)
                temp_fit = temp_model.fit(disp=False)

                if t == 1:
                    pred = temp_fit.predict(start=1, end=1)[0]
                else:
                    pred = temp_fit.forecast(steps=1)[0]

                predictions[t - 1] = pred
            else:
                # Not enough data for ARIMA
                predictions[t - 1] = np.nan

        return predictions


# ============================================================================
# Comparison runner: evaluate all baselines on RV series
# ============================================================================

@dataclass
class BaselineResult:
    name: str
    rmse: float
    mae: float
    n_samples: int
    params: dict


def run_all_baselines(
    rv_series: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> list[BaselineResult]:
    """
    Run all baseline models on RV series and return results.

    rv_series: daily realized variance series
    returns: daily log returns (for GARCH/EGARCH). If None, derived from sqrt(RV).
    """
    results = []

    # HAR
    try:
        har = HARModel()
        har_fit = har.fit(rv_series)
        har_preds = har.predict(rv_series)  # aligned with rv_series[1:]
        actuals = rv_series[1:]
        pred_vals = har_preds[:len(actuals)]
        valid = np.isfinite(actuals) & np.isfinite(pred_vals)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
            mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
            results.append(BaselineResult("HAR", rmse, mae, int(valid.sum()), har_fit))
    except Exception as e:
        print(f"HAR failed: {e}")

    # RVAR
    try:
        rvar = RVARModel(order=1)
        rvar_fit = rvar.fit(rv_series)
        rvar_preds = rvar.predict(rv_series)  # aligned with rv_series[order:]
        actuals = rv_series[rvar.order:]
        pred_vals = rvar_preds[:len(actuals)]
        valid = np.isfinite(actuals) & np.isfinite(pred_vals)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
            mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
            results.append(BaselineResult("RVAR", rmse, mae, int(valid.sum()), rvar_fit))
    except Exception as e:
        print(f"RVAR failed: {e}")

    # GARCH / EGARCH (on returns)
    if returns is not None:
        for model_cls, name in [(GARCHModel, "GARCH(1,1)"), (EGARCHModel, "EGARCH(1,1)")]:
            try:
                model = model_cls()
                fit_result = model.fit(returns)
                # Forecast one step ahead variance
                var_fc = model.forecast_variance(horizon=1)
                # Use last variance as naive comparison, compare with actual squared return
                actual_var = returns[-1] ** 2 if len(returns) > 0 else 0
                pred_var = float(var_fc[0]) if len(var_fc) > 0 else 0
                rmse = float(np.sqrt((actual_var - pred_var) ** 2))
                mae = float(abs(actual_var - pred_var))
                results.append(BaselineResult(name, rmse, mae, int(len(returns)), fit_result))
            except Exception as e:
                print(f"{name} failed: {e}")

    return results


def run_all_baselines_with_advanced(
    rv_series: np.ndarray,
    returns: Optional[np.ndarray] = None,
) -> list[BaselineResult]:
    """
    Extended baseline runner: includes HAR, RVAR, GARCH, EGARCH, Random Forest, XGBoost, and ARIMA.
    
    Run all baseline models on RV series and return results.
    rv_series: daily realized variance series
    returns: daily log returns (for GARCH/EGARCH). If None, derived from sqrt(RV).
    """
    results = []

    # HAR
    try:
        har = HARModel()
        har_fit = har.fit(rv_series)
        har_preds = har.predict(rv_series)  # aligned with rv_series[1:]
        actuals = rv_series[1:]
        pred_vals = har_preds[:len(actuals)]
        valid = np.isfinite(actuals) & np.isfinite(pred_vals)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
            mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
            results.append(BaselineResult("HAR", rmse, mae, int(valid.sum()), har_fit))
    except Exception as e:
        print(f"HAR failed: {e}")

    # RVAR
    try:
        rvar = RVARModel(order=1)
        rvar_fit = rvar.fit(rv_series)
        rvar_preds = rvar.predict(rv_series)  # aligned with rv_series[order:]
        actuals = rv_series[rvar.order:]
        pred_vals = rvar_preds[:len(actuals)]
        valid = np.isfinite(actuals) & np.isfinite(pred_vals)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
            mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
            results.append(BaselineResult("RVAR", rmse, mae, int(valid.sum()), rvar_fit))
    except Exception as e:
        print(f"RVAR failed: {e}")

    # GARCH / EGARCH (on returns)
    if returns is not None:
        for model_cls, name in [(GARCHModel, "GARCH(1,1)"), (EGARCHModel, "EGARCH(1,1)")]:
            try:
                model = model_cls()
                fit_result = model.fit(returns)
                # Forecast one step ahead variance
                var_fc = model.forecast_variance(horizon=1)
                # Use last variance as naive comparison, compare with actual squared return
                actual_var = returns[-1] ** 2 if len(returns) > 0 else 0
                pred_var = float(var_fc[0]) if len(var_fc) > 0 else 0
                rmse = float(np.sqrt((actual_var - pred_var) ** 2))
                mae = float(abs(actual_var - pred_var))
                results.append(BaselineResult(name, rmse, mae, int(len(returns)), fit_result))
            except Exception as e:
                print(f"{name} failed: {e}")

    # Random Forest (uses RV directly)
    if HAS_SKLEARN:
        try:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10)
            rf_fit = rf.fit(rv_series)
            rf_preds = rf.predict(rv_series)  # aligned with rv_series[1:]
            actuals = rv_series[1:]
            pred_vals = rf_preds[:len(actuals)]
            valid = np.isfinite(actuals) & np.isfinite(pred_vals)
            if valid.sum() > 0:
                rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
                mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
                results.append(BaselineResult("RandomForest", rmse, mae, int(valid.sum()), rf_fit))
        except Exception as e:
            print(f"RandomForest failed: {e}")

    # XGBoost (uses RV directly)
    if HAS_XGBOOST:
        try:
            xgb = XGBoostRegressor(n_estimators=100, max_depth=6)
            xgb_fit = xgb.fit(rv_series)
            xgb_preds = xgb.predict(rv_series)  # aligned with rv_series[1:]
            actuals = rv_series[1:]
            pred_vals = xgb_preds[:len(actuals)]
            valid = np.isfinite(actuals) & np.isfinite(pred_vals)
            if valid.sum() > 0:
                rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
                mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
                results.append(BaselineResult("XGBoost", rmse, mae, int(valid.sum()), xgb_fit))
        except Exception as e:
            print(f"XGBoost failed: {e}")

    # ARIMA (uses RV directly)
    if HAS_STATS_MODELS:
        try:
            arima = ARIMAModel(order=(1, 1, 1))
            arima_fit = arima.fit(rv_series)
            arima_preds = arima.predict(rv_series)  # aligned with rv_series[1:]
            actuals = rv_series[1:]
            pred_vals = arima_preds[:len(actuals)]
            valid = np.isfinite(actuals) & np.isfinite(pred_vals)
            if valid.sum() > 0:
                rmse = float(np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2)))
                mae = float(np.mean(np.abs(actuals[valid] - pred_vals[valid])))
                results.append(BaselineResult("ARIMA", rmse, mae, int(valid.sum()), arima_fit))
        except Exception as e:
            print(f"ARIMA failed: {e}")

    return results


def compare_all_models(
    deep_results: dict,
    baseline_results: list[BaselineResult],
) -> dict:
    """
    Combine deep learning results with baseline results into a comparison table.

    deep_results: dict with keys like {"LSTM": rmse, "Mamba": rmse, ...}
    baseline_results: list of BaselineResult
    """
    comparison = {}
    for name, rmse in deep_results.items():
        comparison[name] = {"RMSE": rmse, "Type": "Deep Learning"}

    for br in baseline_results:
        comparison[br.name] = {"RMSE": br.rmse, "MAE": br.mae, "Type": "Statistical/ML"}

    # Sort by RMSE
    sorted_items = sorted(comparison.items(), key=lambda x: x[1].get("RMSE", float("inf")))
    return dict(sorted_items)


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n = 500
    # Simulate RV series with some persistence
    rv = np.abs(np.random.randn(n) * 0.01 + 0.005)
    # Add persistence
    for t in range(1, n):
        rv[t] = 0.7 * rv[t - 1] + 0.3 * np.abs(np.random.randn() * 0.01)
    returns = np.random.randn(n) * 0.02

    # Test original baselines
    print("=== ORIGINAL BASES (4 models) ===")
    results_orig = run_all_baselines(rv, returns)
    for r in results_orig:
        print(f"{r.name:15s} RMSE={r.rmse:.6f} MAE={r.mae:.6f} n={r.n_samples}")

    # Test enhanced baselines (7 models)
    print("\n=== ENHANCED BASES (7 models) ===")
    results_enh = run_all_baselines_with_advanced(rv, returns)
    for r in results_enh:
        print(f"{r.name:15s} RMSE={r.rmse:.6f} MAE={r.mae:.6f} n={r.n_samples}")

    # Show improvement
    print("\n=== COMPARISON ===")
    print(f"Original baselines: {len(results_orig)} models")
    print(f"Enhanced baselines: {len(results_enh)} models")
    print(f"Added: {len(results_enh) - len(results_orig)} new models")

    # Show which new models worked
    orig_names = {r.name for r in results_orig}
    enh_names = {r.name for r in results_enh}
    new_models = enh_names - orig_names
    if new_models:
        print(f"✅ Successfully added: {', '.join(sorted(new_models))}")
    else:
        print("❌ No new models added")

    # Test RandomForest specifically
    if HAS_SKLEARN:
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5)
            rf_fit = rf.fit(rv)
            rf_preds = rf.predict(rv)
            actuals = rv[1:]
            pred_vals = rf_preds[:len(actuals)]
            valid = np.isfinite(actuals) & np.isfinite(pred_vals)
            if valid.sum() > 0:
                rf_rmse = np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2))
                print(f"\nRandomForest specific test: RMSE={rf_rmse:.6f}")
        except Exception as e:
            print(f"RandomForest test failed: {e}")

    # Test XGBoost specifically
    if HAS_XGBOOST:
        try:
            xgb = XGBoostRegressor(n_estimators=50, max_depth=4)
            xgb_fit = xgb.fit(rv)
            xgb_preds = xgb.predict(rv)
            actuals = rv[1:]
            pred_vals = xgb_preds[:len(actuals)]
            valid = np.isfinite(actuals) & np.isfinite(pred_vals)
            if valid.sum() > 0:
                xgb_rmse = np.sqrt(np.mean((actuals[valid] - pred_vals[valid]) ** 2))
                print(f"XGBoost specific test: RMSE={xgb_rmse:.6f}")
        except Exception as e:
            print(f"XGBoost test failed: {e}")

    # Test ARIMA specifically
    if HAS_STATS_MODELS:
        try:
            arima = ARIMAModel(order=(1, 1, 1))
            arima_fit = arima.fit(rv)
            print(f"ARIMA specific test: RMSE={arima_fit['rmse']:.6f}")
        except Exception as e:
            print(f"ARIMA test failed: {e}")