"""
evaluation/loss_functions.py — Loss functions for volatility forecast evaluation.

Implements:
    - MSE  (Mean Squared Error)
    - MAE  (Mean Absolute Error)
    - QLIKE (Quasi-Likelihood loss) — standard for RV forecasting
    - R²_OOS (Out-of-Sample R²)

QLIKE is the primary metric following Patton (2011), which shows QLIKE is
robust to noise in the volatility proxy (i.e., ranking of forecasts is
consistent regardless of whether we use RV, BPV, or true σ²).
"""

import numpy as np
import pandas as pd
from typing import Union


def mse(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Mean Squared Error.

    MSE = (1/T) * Σ (actual_t - forecast_t)²

    Parameters
    ----------
    actual : array-like
        Realized values (RV).
    forecast : array-like
        Forecasted values.

    Returns
    -------
    float
        MSE value.
    """
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean((actual - forecast) ** 2))


def mae(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Mean Absolute Error.

    MAE = (1/T) * Σ |actual_t - forecast_t|

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Forecasted values.

    Returns
    -------
    float
        MAE value.
    """
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean(np.abs(actual - forecast)))


def qlike(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> float:
    """Quasi-Likelihood loss (QLIKE).

    QLIKE = (1/T) * Σ [actual_t / forecast_t - log(actual_t / forecast_t) - 1]

    This is the robust loss function from Patton (2011). It is the
    standard loss for RV forecasting because forecast rankings under QLIKE
    are invariant to the choice of volatility proxy.

    Parameters
    ----------
    actual : array-like
        Realized values (must be positive).
    forecast : array-like
        Forecasted values (must be positive).

    Returns
    -------
    float
        QLIKE value.
    """
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)

    # Guard against non-positive values
    mask = (actual > 0) & (forecast > 0)
    if not mask.all():
        actual = actual[mask]
        forecast = forecast[mask]

    ratio = actual / forecast
    return float(np.mean(ratio - np.log(ratio) - 1))


def r2_oos(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    benchmark_forecast: Union[np.ndarray, pd.Series, None] = None,
) -> float:
    """Out-of-Sample R² (Campbell & Thompson 2008).

    R²_OOS = 1 - Σ(actual - forecast)² / Σ(actual - benchmark)²

    If no benchmark is provided, uses the historical mean as benchmark
    (equivalent to standard OOS R²).

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Model forecasts.
    benchmark_forecast : array-like, optional
        Benchmark forecasts (default: expanding-window mean).

    Returns
    -------
    float
        R²_OOS. Positive = model beats benchmark.
    """
    actual, forecast = np.asarray(actual), np.asarray(forecast)

    if benchmark_forecast is None:
        # Use expanding mean as benchmark (prevailing mean forecast)
        benchmark_forecast = np.full_like(actual, np.mean(actual))
    else:
        benchmark_forecast = np.asarray(benchmark_forecast)

    ss_model = np.sum((actual - forecast) ** 2)
    ss_benchmark = np.sum((actual - benchmark_forecast) ** 2)

    if ss_benchmark == 0:
        return 0.0

    return float(1 - ss_model / ss_benchmark)


def compute_loss_series(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    loss_type: str = "QLIKE",
) -> np.ndarray:
    """Compute element-wise loss series (for DM test input).

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Forecasted values.
    loss_type : str
        One of: 'MSE', 'MAE', 'QLIKE'.

    Returns
    -------
    np.ndarray
        Per-observation loss values.
    """
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)

    if loss_type == 'MSE':
        return (actual - forecast) ** 2
    elif loss_type == 'MAE':
        return np.abs(actual - forecast)
    elif loss_type == 'QLIKE':
        ratio = actual / np.maximum(forecast, 1e-10)
        return ratio - np.log(np.maximum(ratio, 1e-10)) - 1
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_all_losses(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> dict:
    """Compute all loss functions at once.

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Forecasted values.

    Returns
    -------
    dict
        Keys: 'MSE', 'MAE', 'QLIKE', 'R2OOS'. Values: float.
    """
    return {
        'MSE': mse(actual, forecast),
        'MAE': mae(actual, forecast),
        'QLIKE': qlike(actual, forecast),
        'R2OOS': r2_oos(actual, forecast),
    }
