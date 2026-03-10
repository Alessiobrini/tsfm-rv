"""
evaluation/mz_regression.py — Mincer-Zarnowitz forecast efficiency regression.

Regresses actual values on a constant and forecast:
    actual_t = alpha + beta * forecast_t + epsilon_t

Tests:
    H0: alpha = 0, beta = 1 (joint F-test for forecast efficiency)
    H0: alpha = 0 (unbiasedness of intercept)
    H0: beta = 1 (unit slope)

Reference: Mincer & Zarnowitz (1969), Patton & Sheppard (2009).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from dataclasses import dataclass
from typing import Union


@dataclass
class MZResult:
    """Result of a Mincer-Zarnowitz regression.

    Attributes:
        alpha: Estimated intercept.
        beta: Estimated slope.
        alpha_se: Standard error of alpha.
        beta_se: Standard error of beta.
        alpha_pval: p-value for H0: alpha = 0.
        beta_pval: p-value for H0: beta = 1.
        joint_fstat: F-statistic for joint H0: alpha=0, beta=1.
        joint_pval: p-value for joint test.
        r_squared: R-squared of the regression.
        n_obs: Number of observations.
    """
    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    alpha_pval: float
    beta_pval: float
    joint_fstat: float
    joint_pval: float
    r_squared: float
    n_obs: int


def mz_regression(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    hac_lags: int = 22,
) -> MZResult:
    """Run Mincer-Zarnowitz regression with HAC standard errors.

    actual_t = alpha + beta * forecast_t + epsilon_t

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Forecasted values.
    hac_lags : int
        Newey-West HAC lag truncation.

    Returns
    -------
    MZResult
        Regression results and test statistics.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    # Drop any NaN pairs
    mask = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[mask], forecast[mask]
    n = len(actual)

    X = sm.add_constant(forecast)
    ols = sm.OLS(actual, X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})

    alpha = ols.params[0]
    beta = ols.params[1]
    alpha_se = ols.bse[0]
    beta_se = ols.bse[1]

    # Test H0: alpha = 0
    alpha_pval = ols.pvalues[0]

    # Test H0: beta = 1 (two-sided)
    t_beta1 = (beta - 1.0) / beta_se
    beta_pval = 2.0 * (1.0 - stats.norm.cdf(abs(t_beta1)))

    # Joint F-test: H0: alpha=0, beta=1
    # Use Wald test with R*(params - r) formulation
    try:
        r = np.array([0.0, 1.0])
        params = np.asarray(ols.params)
        V = np.asarray(ols.cov_params())
        diff = params - r
        wald_stat = float(diff @ np.linalg.solve(V, diff))
        joint_fstat = wald_stat / 2.0  # 2 restrictions
        from scipy import stats as sp_stats
        joint_pval = float(1.0 - sp_stats.f.cdf(joint_fstat, 2, n - 2))
    except Exception:
        joint_fstat = np.nan
        joint_pval = np.nan

    return MZResult(
        alpha=float(alpha),
        beta=float(beta),
        alpha_se=float(alpha_se),
        beta_se=float(beta_se),
        alpha_pval=float(alpha_pval),
        beta_pval=float(beta_pval),
        joint_fstat=joint_fstat,
        joint_pval=joint_pval,
        r_squared=float(ols.rsquared),
        n_obs=n,
    )


def recursive_mz_correction(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
    min_window: int = 252,
) -> np.ndarray:
    """Apply recursively estimated MZ bias correction to forecasts.

    For each t >= min_window, estimate alpha/beta on data up to t-1,
    then correct forecast_t as: corrected_t = alpha_hat + beta_hat * forecast_t.

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecast : array-like
        Raw forecasted values.
    min_window : int
        Minimum observations before starting correction (default: 252).

    Returns
    -------
    np.ndarray
        Bias-corrected forecasts (length = len(forecast) - min_window).
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    n = len(actual)

    corrected = []
    for t in range(min_window, n):
        # Estimate MZ on data up to t-1
        y_train = actual[:t]
        f_train = forecast[:t]
        X = sm.add_constant(f_train)
        try:
            ols = sm.OLS(y_train, X).fit()
            alpha_hat = ols.params[0]
            beta_hat = ols.params[1]
            corrected.append(alpha_hat + beta_hat * forecast[t])
        except Exception:
            corrected.append(forecast[t])

    return np.array(corrected)


def mz_table(
    actual: Union[np.ndarray, pd.Series],
    forecasts: dict,
    hac_lags: int = 22,
) -> pd.DataFrame:
    """Run MZ regression for multiple models and return summary table.

    Parameters
    ----------
    actual : array-like
        Realized values (common across models).
    forecasts : dict
        {model_name: forecast array}.
    hac_lags : int
        HAC lag truncation.

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = alpha, beta, R2, joint p-value, etc.
    """
    rows = []
    for model_name, fcast in forecasts.items():
        result = mz_regression(actual, fcast, hac_lags=hac_lags)
        rows.append({
            'model': model_name,
            'alpha': result.alpha,
            'alpha_se': result.alpha_se,
            'alpha_pval': result.alpha_pval,
            'beta': result.beta,
            'beta_se': result.beta_se,
            'beta_pval': result.beta_pval,
            'R2': result.r_squared,
            'F_stat': result.joint_fstat,
            'F_pval': result.joint_pval,
            'n_obs': result.n_obs,
        })
    return pd.DataFrame(rows).set_index('model')
