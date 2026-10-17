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

    # Critical values depend on mu = m / T, the window fraction.
    mu = m / T
    cv_10 = gr_critical_value(mu, 0.10)
    cv_05 = gr_critical_value(mu, 0.05)

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


# Two-sided critical values of the fluctuation test, Giacomini and Rossi (2010), Table 1, as the
# authors distribute them (GiacominiRossiCV.m in the Journal of Applied Econometrics data archive,
# 2010-v25.4/giacomini-rossi). The limit is the supremum over tau of |B(tau) - B(tau - mu)| / sqrt(mu),
# with B a standard Brownian motion. Keys are mu = m / T; values are (5%, 10%).
GR_CRITICAL_VALUES = {
    0.1: (3.393, 3.170), 0.2: (3.179, 2.948), 0.3: (3.012, 2.766),
    0.4: (2.890, 2.626), 0.5: (2.779, 2.500), 0.6: (2.634, 2.356),
    0.7: (2.560, 2.252), 0.8: (2.433, 2.130), 0.9: (2.248, 1.950),
}


def gr_critical_value(mu: float, alpha: float = 0.05) -> float:
    """Critical value of the fluctuation test at window fraction ``mu`` and level ``alpha`` (0.05 or
    0.10), interpolated linearly between the tabulated values of mu."""
    if alpha not in (0.05, 0.10):
        raise ValueError("alpha must be 0.05 or 0.10")
    mus = np.array(sorted(GR_CRITICAL_VALUES))
    col = 0 if alpha == 0.05 else 1
    vals = np.array([GR_CRITICAL_VALUES[m_][col] for m_ in mus])
    return float(np.interp(np.clip(mu, mus[0], mus[-1]), mus, vals))


def gr_fluctuation_multiple(
    actual: Union[np.ndarray, pd.Series],
    forecasts: dict,
    benchmark: str,
    loss_type: str = "QLIKE",
    window_fraction: float = 0.3,
    hac_lags: int = 1,
    dates: Optional[pd.DatetimeIndex] = None,
    scale: str = "var",
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
    scale : str
        Scale of ``actual`` and the forecasts, "var" or "vol". QLIKE is a variance loss, so
        volatilities are squared before it is computed.

    Returns
    -------
    dict
        {model_name: GRFluctuationResult} for each non-benchmark model.
    """
    actual_arr = np.asarray(actual)
    bench_loss = compute_loss_series(actual_arr, np.asarray(forecasts[benchmark]), loss_type, scale=scale)

    results = {}
    for model_name, fcast in forecasts.items():
        if model_name == benchmark:
            continue
        model_loss = compute_loss_series(actual_arr, np.asarray(fcast), loss_type, scale=scale)
        results[model_name] = gr_fluctuation_test(
            bench_loss, model_loss,
            window_fraction=window_fraction,
            hac_lags=hac_lags,
            dates=dates,
            model_1=benchmark,
            model_2=model_name,
        )
    return results


def cross_asset_average(series: list) -> pd.Series:
    """Average of per-asset rolling statistics on each calendar date.

    Only the dates on which every asset has a value are kept, so each point of the average is over
    the same set of assets.
    """
    if not series:
        raise ValueError("no series to average")
    for s in series:
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("rolling statistics must be indexed by date")
    panel = pd.concat([s.rename(i) for i, s in enumerate(series)], axis=1)
    return panel.dropna().mean(axis=1)
