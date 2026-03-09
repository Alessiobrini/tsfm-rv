"""
evaluation/gr_fluctuation.py — Giacomini-Rossi (2010) Fluctuation Test.

Computes rolling Diebold-Mariano statistics over a moving window to detect
time-varying relative forecast performance. The test statistic is the
supremum of the absolute rolling DM statistics, compared against critical
values from the Kolmogorov-Smirnov distribution.

Reference: Giacomini & Rossi (2010, Journal of Applied Econometrics).
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Union, Optional, Tuple

from .loss_functions import compute_loss_series


@dataclass
class GRFluctuationResult:
    """Result of the Giacomini-Rossi Fluctuation Test.

    Attributes:
        rolling_dm: Series of rolling DM statistics (indexed by date).
        sup_stat: Supremum of |rolling DM| (the test statistic).
        critical_value_10: 10% critical value.
        critical_value_05: 5% critical value.
        reject_10: Whether H0 (equal predictive ability over time) is rejected at 10%.
        reject_05: Whether rejected at 5%.
        window_size: Rolling window used.
        model_1: Name of model 1.
        model_2: Name of model 2.
    """
    rolling_dm: pd.Series
    sup_stat: float
    critical_value_10: float
    critical_value_05: float
    reject_10: bool
    reject_05: bool
    window_size: int
    model_1: str
    model_2: str


def _nw_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC variance of the mean of d."""
    T = len(d)
    d_bar = np.mean(d)
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight = 1 - k / (max_lag + 1)
        gamma_sum += 2 * weight * gamma_k
    return (gamma_0 + gamma_sum) / T


def gr_fluctuation_test(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    window_fraction: float = 0.3,
    hac_lags: int = 1,
    dates: Optional[pd.DatetimeIndex] = None,
    model_1: str = "model_1",
    model_2: str = "model_2",
) -> GRFluctuationResult:
    """Giacomini-Rossi Fluctuation Test for time-varying predictive ability.

    Computes a sequence of rolling DM statistics using a centered or
    trailing window of size m = floor(window_fraction * T).

    Parameters
    ----------
    loss_1 : np.ndarray
        Loss series for model 1.
    loss_2 : np.ndarray
        Loss series for model 2.
    window_fraction : float
        Fraction of sample for rolling window (typical: 0.3).
    hac_lags : int
        Lags for Newey-West HAC within each window.
    dates : pd.DatetimeIndex, optional
        Date index for the output series.
    model_1, model_2 : str
        Model names for labeling.

    Returns
    -------
    GRFluctuationResult
    """
    loss_1, loss_2 = np.asarray(loss_1), np.asarray(loss_2)
    d = loss_1 - loss_2
    T = len(d)
    m = max(int(np.floor(window_fraction * T)), 30)

    # Rolling DM statistics
    rolling_stats = []
    start_indices = []

    for t in range(m, T + 1):
        d_window = d[t - m:t]
        d_bar = np.mean(d_window)

        # HAC variance within window
        gamma_0 = np.var(d_window, ddof=0)
        gamma_sum = 0.0
        for k in range(1, min(hac_lags, m - 1) + 1):
            gamma_k = np.mean(
                (d_window[k:] - d_bar) * (d_window[:-k] - d_bar)
            )
            weight = 1 - k / (hac_lags + 1)
            gamma_sum += 2 * weight * gamma_k

        var_d = (gamma_0 + gamma_sum) / m
        if var_d > 0:
            dm_stat = d_bar / np.sqrt(var_d)
        else:
            dm_stat = 0.0

        rolling_stats.append(dm_stat)
        start_indices.append(t - 1)  # end index of window

    rolling_stats = np.array(rolling_stats)

    # Create indexed series
    if dates is not None and len(dates) == T:
        idx = dates[start_indices]
    else:
        idx = pd.RangeIndex(start_indices[0], start_indices[-1] + 1)

    rolling_dm = pd.Series(rolling_stats, index=idx, name=f"DM({model_1} vs {model_2})")

    # Supremum statistic
    sup_stat = float(np.max(np.abs(rolling_stats)))

    # Critical values from Giacomini & Rossi (2010), Table 1
    # These depend on mu = m/T (the window fraction)
    # For mu in [0.1, 0.5], approximate critical values:
    mu = m / T
    # Use the asymptotic critical values from the paper (Table 1, two-sided)
    # These are derived from the supremum of a standardized Brownian bridge
    # Approximation based on Giacomini & Rossi (2010) Table 1
    cv_table = {
        # mu: (cv_10, cv_05)
        0.1: (3.17, 3.39),
        0.2: (2.82, 3.05),
        0.3: (2.55, 2.80),
        0.4: (2.32, 2.58),
        0.5: (2.10, 2.38),
    }
    # Interpolate
    mus = np.array(sorted(cv_table.keys()))
    cv10s = np.array([cv_table[m_][0] for m_ in sorted(cv_table.keys())])
    cv05s = np.array([cv_table[m_][1] for m_ in sorted(cv_table.keys())])

    mu_clipped = np.clip(mu, mus[0], mus[-1])
    cv_10 = float(np.interp(mu_clipped, mus, cv10s))
    cv_05 = float(np.interp(mu_clipped, mus, cv05s))

    return GRFluctuationResult(
        rolling_dm=rolling_dm,
        sup_stat=sup_stat,
        critical_value_10=cv_10,
        critical_value_05=cv_05,
        reject_10=sup_stat > cv_10,
        reject_05=sup_stat > cv_05,
        window_size=m,
        model_1=model_1,
        model_2=model_2,
    )


def gr_fluctuation_multiple(
    actual: Union[np.ndarray, pd.Series],
    forecasts: dict,
    benchmark: str,
    loss_type: str = "QLIKE",
    window_fraction: float = 0.3,
    hac_lags: int = 1,
    dates: Optional[pd.DatetimeIndex] = None,
) -> dict:
    """Run GR Fluctuation Test for all models against a benchmark.

    Parameters
    ----------
    actual : array-like
        Realized values.
    forecasts : dict
        {model_name: forecast_array}.
    benchmark : str
        Name of benchmark model (must be in forecasts).
    loss_type : str
        Loss function: 'MSE', 'MAE', 'QLIKE'.
    window_fraction : float
        Rolling window as fraction of sample.
    hac_lags : int
        HAC lags.
    dates : pd.DatetimeIndex, optional
        Date index.

    Returns
    -------
    dict
        {model_name: GRFluctuationResult} for each non-benchmark model.
    """
    actual_arr = np.asarray(actual)
    bench_loss = compute_loss_series(actual_arr, np.asarray(forecasts[benchmark]), loss_type)

    results = {}
    for model_name, fcast in forecasts.items():
        if model_name == benchmark:
            continue
        model_loss = compute_loss_series(actual_arr, np.asarray(fcast), loss_type)
        results[model_name] = gr_fluctuation_test(
            bench_loss, model_loss,
            window_fraction=window_fraction,
            hac_lags=hac_lags,
            dates=dates,
            model_1=benchmark,
            model_2=model_name,
        )
    return results
