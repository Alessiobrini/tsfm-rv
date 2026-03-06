"""
evaluation/portfolio.py — Global Minimum Variance portfolio construction and evaluation.

Evaluates the economic value of covariance forecasts by:
1. Building GMV portfolios from forecasted covariance matrices
2. Computing realized portfolio variance using actual covariance matrices
3. Computing Sharpe ratio, turnover, and certainty equivalent return (CER)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from covariance_utils import build_gmv_weights, ensure_psd


def compute_portfolio_performance(
    forecast_matrices: Dict,
    actual_matrices: Dict,
    assets: List[str],
    returns: Optional[pd.DataFrame] = None,
    tc_bps: float = 10.0,
    gamma_values: List[float] = None,
) -> pd.DataFrame:
    """Evaluate GMV portfolio performance across all OOS dates.

    Parameters
    ----------
    forecast_matrices : dict
        Maps date -> np.ndarray of forecasted covariance matrix.
    actual_matrices : dict
        Maps date -> np.ndarray of actual realized covariance matrix.
    assets : list
        Sorted asset names.
    returns : pd.DataFrame, optional
        Daily returns with columns matching assets. If None, only variance metrics computed.
    tc_bps : float
        Proportional transaction cost in basis points.
    gamma_values : list of float
        Risk aversion parameters for CER computation.

    Returns
    -------
    pd.DataFrame
        Daily portfolio metrics: date, weights, realized variance, return, turnover.
    """
    if gamma_values is None:
        gamma_values = [1, 5, 10]

    # Get dates where we have both forecasts and actuals
    common_dates = sorted(set(forecast_matrices.keys()) & set(actual_matrices.keys()))
    if len(common_dates) == 0:
        return pd.DataFrame()

    n = len(assets)
    records = []
    prev_weights = np.ones(n) / n  # start with equal weight

    for date in common_dates:
        cov_hat = forecast_matrices[date]
        cov_actual = actual_matrices[date]

        # GMV weights from forecasted covariance
        w = build_gmv_weights(cov_hat)

        # Realized portfolio variance: w' Sigma_actual w
        realized_var = float(w @ cov_actual @ w)

        # Turnover
        turnover = float(np.sum(np.abs(w - prev_weights)))

        record = {
            'date': date,
            'realized_var': realized_var,
            'turnover': turnover,
            'max_weight': float(np.max(np.abs(w))),
            'min_weight': float(np.min(w)),
            'n_short': int(np.sum(w < -0.01)),
        }

        # Portfolio return (if returns available)
        if returns is not None and date in returns.index:
            r = returns.loc[date, assets].values
            port_return = float(w @ r)
            record['return'] = port_return

            # Transaction costs
            tc = tc_bps / 10000 * turnover
            record['return_net'] = port_return - tc

        records.append(record)
        prev_weights = w.copy()

    return pd.DataFrame(records)


def compute_equal_weight_performance(
    actual_matrices: Dict,
    assets: List[str],
    returns: Optional[pd.DataFrame] = None,
    dates: Optional[List] = None,
) -> pd.DataFrame:
    """Compute 1/N equal-weight portfolio performance as benchmark.

    Parameters
    ----------
    actual_matrices : dict
        Maps date -> actual covariance matrix.
    assets : list
        Asset names.
    returns : pd.DataFrame, optional
        Daily returns.
    dates : list, optional
        Dates to evaluate (default: all dates in actual_matrices).

    Returns
    -------
    pd.DataFrame
        Daily performance metrics.
    """
    if dates is None:
        dates = sorted(actual_matrices.keys())

    n = len(assets)
    w = np.ones(n) / n
    records = []

    for date in dates:
        if date not in actual_matrices:
            continue
        cov_actual = actual_matrices[date]
        realized_var = float(w @ cov_actual @ w)

        record = {
            'date': date,
            'realized_var': realized_var,
            'turnover': 0.0,
        }

        if returns is not None and date in returns.index:
            r = returns.loc[date, assets].values
            record['return'] = float(w @ r)
            record['return_net'] = record['return']

        records.append(record)

    return pd.DataFrame(records)


def summarize_portfolio_metrics(
    perf: pd.DataFrame,
    annualization: int = 252,
) -> dict:
    """Compute summary statistics from daily portfolio performance.

    Parameters
    ----------
    perf : pd.DataFrame
        Output of compute_portfolio_performance().
    annualization : int
        Trading days per year.

    Returns
    -------
    dict
        Summary metrics.
    """
    summary = {
        'avg_realized_var': perf['realized_var'].mean(),
        'std_realized_var': perf['realized_var'].std(),
        'avg_turnover': perf['turnover'].mean(),
        'avg_max_weight': perf['max_weight'].mean() if 'max_weight' in perf else None,
        'n_days': len(perf),
    }

    if 'return' in perf.columns:
        avg_ret = perf['return'].mean() * annualization
        std_ret = perf['return'].std() * np.sqrt(annualization)
        summary['annual_return'] = avg_ret
        summary['annual_vol'] = std_ret
        summary['sharpe'] = avg_ret / std_ret if std_ret > 0 else 0.0

    if 'return_net' in perf.columns:
        avg_ret_net = perf['return_net'].mean() * annualization
        std_ret_net = perf['return_net'].std() * np.sqrt(annualization)
        summary['sharpe_net'] = avg_ret_net / std_ret_net if std_ret_net > 0 else 0.0

        for gamma in [1, 5, 10]:
            var_p = perf['return_net'].var() * annualization
            cer = avg_ret_net - (gamma / 2) * var_p
            summary[f'cer_gamma{gamma}'] = cer

    return summary
