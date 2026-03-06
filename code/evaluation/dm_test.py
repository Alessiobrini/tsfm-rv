"""
evaluation/dm_test.py — Diebold-Mariano test for forecast comparison.

Implements the Diebold-Mariano (1995) test for equal predictive accuracy.
Uses HAC (Newey-West) variance estimation to account for serial correlation
in the loss differential series.

Reference: Diebold & Mariano (1995, JBES).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DMTestResult:
    """Result of a Diebold-Mariano test.

    Attributes:
        statistic: DM test statistic.
        p_value: p-value.
        alternative: Test direction ('two-sided', 'less', 'greater').
        model_1: Name of first model.
        model_2: Name of second model.
        mean_loss_diff: Mean loss differential (L1 - L2).
    """
    statistic: float
    p_value: float
    alternative: str
    model_1: str
    model_2: str
    mean_loss_diff: float


def dm_test(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided",
    hac_lags: Optional[int] = None,
) -> DMTestResult:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H₀: E[d_t] = 0 where d_t = L(e_{1,t}) - L(e_{2,t}).

    Parameters
    ----------
    loss_1 : np.ndarray
        Loss series for model 1.
    loss_2 : np.ndarray
        Loss series for model 2.
    h : int
        Forecast horizon (used for HAC lag truncation if hac_lags is None).
    alternative : str
        'two-sided': H₁: E[d_t] ≠ 0
        'less':      H₁: E[d_t] < 0 (model 1 is better)
        'greater':   H₁: E[d_t] > 0 (model 2 is better)
    hac_lags : int, optional
        Number of lags for Newey-West HAC. Default: max(1, h-1).

    Returns
    -------
    DMTestResult
        Test statistic, p-value, and metadata.
    """
    loss_1, loss_2 = np.asarray(loss_1), np.asarray(loss_2)
    d = loss_1 - loss_2  # Loss differential
    T = len(d)

    if hac_lags is None:
        hac_lags = max(1, h - 1)

    # Mean loss differential
    d_bar = np.mean(d)

    # HAC variance estimator (Newey-West)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, hac_lags + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight = 1 - k / (hac_lags + 1)  # Bartlett kernel
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        # Fallback to simple variance if HAC gives non-positive
        var_d = gamma_0 / T

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d)

    # p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return DMTestResult(
        statistic=float(dm_stat),
        p_value=float(p_value),
        alternative=alternative,
        model_1="model_1",
        model_2="model_2",
        mean_loss_diff=float(d_bar),
    )


def dm_test_matrix(
    losses: dict,
    h: int = 1,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Compute pairwise DM test p-values for all model pairs.

    Parameters
    ----------
    losses : dict
        Keys: model names. Values: np.ndarray of loss series.
    h : int
        Forecast horizon.
    alternative : str
        Test direction.

    Returns
    -------
    pd.DataFrame
        Square matrix of DM test p-values. Entry (i, j) tests
        whether model i has significantly different loss from model j.
    """
    model_names = list(losses.keys())
    n = len(model_names)
    p_values = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                result = dm_test(
                    losses[model_names[i]],
                    losses[model_names[j]],
                    h=h,
                    alternative=alternative,
                )
                p_values[i, j] = result.p_value

    return pd.DataFrame(p_values, index=model_names, columns=model_names)
