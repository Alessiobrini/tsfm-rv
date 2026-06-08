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
    scale: str = "var",
    var_floor: float = None,
) -> float:
    """Quasi-Likelihood loss (QLIKE).

    QLIKE = (1/T) * Σ [σ²_t / ĥ_t - log(σ²_t / ĥ_t) - 1]

    QLIKE is intrinsically a *variance* loss (Patton 2011): it compares a variance
    proxy σ²_t to a variance forecast ĥ_t. When the headline modeling scale is
    volatility (``scale="vol"``), the stored actual/forecast are volatilities, so
    we square them back to variances first — the proxy then equals the realized
    variance exactly (no information loss) and QLIKE keeps its proxy-robustness.

    Parameters
    ----------
    actual, forecast : array-like
        Realized and forecast values on the modeling ``scale``.
    scale : str
        "var" (values already variances) or "vol" (values are volatilities; squared
        internally to variances).
    var_floor : float, optional
        Lower bound on the variance forecast denominator (the minimum realized
        variance in the estimation window — Referee 2 minor 4 — replacing the
        "unacceptable" 1e-10 floor). Applied after squaring to the variance scale.

    Returns
    -------
    float
        QLIKE value.
    """
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if scale == "vol":
        actual, forecast = actual ** 2, forecast ** 2  # volatility -> variance
    elif scale != "var":
        raise ValueError(f"scale must be 'var' or 'vol', got {scale!r}")

    if var_floor is not None:
        forecast = np.maximum(forecast, var_floor)

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
    scale: str = "var",
    var_floor: float = None,
) -> np.ndarray:
    """Compute element-wise loss series (for DM test input).

    MSE/MAE are computed on the modeling ``scale`` (e.g. volatility); QLIKE is
    always a variance loss (values squared to variance when ``scale="vol"``).

    Parameters
    ----------
    actual, forecast : array-like
        Realized and forecast values on the modeling ``scale``.
    loss_type : str
        One of: 'MSE', 'MAE', 'QLIKE'.
    scale : str
        "var" or "vol".
    var_floor : float, optional
        Variance-scale floor for the QLIKE denominator (min RV in window).

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
        a, f = (actual ** 2, forecast ** 2) if scale == "vol" else (actual, forecast)
        if var_floor is not None:
            f = np.maximum(f, var_floor)
        ratio = a / f
        return ratio - np.log(ratio) - 1
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_all_losses(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    scale: str = "var",
    var_floor: float = None,
) -> dict:
    """Compute all loss functions at once.

    MSE/MAE/R²_OOS are on the modeling ``scale`` (volatility); QLIKE is on the
    variance scale (squared internally when ``scale="vol"``), floored at
    ``var_floor`` (min RV in the estimation window).

    Returns
    -------
    dict
        Keys: 'MSE', 'MAE', 'QLIKE', 'R2OOS'. Values: float.
    """
    return {
        'MSE': mse(actual, forecast),
        'MAE': mae(actual, forecast),
        'QLIKE': qlike(actual, forecast, scale=scale, var_floor=var_floor),
        'R2OOS': r2_oos(actual, forecast),
    }
