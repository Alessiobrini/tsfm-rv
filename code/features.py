"""
features.py — Construct HAR regressors and model inputs from RV data.

Builds the feature matrices needed by HAR, HAR-J, HAR-RS, HARQ models.
All functions operate on single-asset Series and return aligned DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute backward-looking rolling mean of a series.

    RV^{(w)}_{t} = (1/5) * sum_{i=0}^{4} RV_{t-i}
    RV^{(m)}_{t} = (1/22) * sum_{i=0}^{21} RV_{t-i}

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g., daily RV).
    window : int
        Window length in trading days.

    Returns
    -------
    pd.Series
        Rolling mean, same index as input, NaN where insufficient history.
    """
    return series.rolling(window=window, min_periods=window).mean()


def build_har_features(
    rv: pd.Series,
    daily_lag: int = 1,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
) -> pd.DataFrame:
    """Build standard HAR regressors for a single asset.

    HAR(Corsi 2009): RV_{t+1} = β₀ + β₁·RV_t + β₂·RV^{(w)}_t + β₃·RV^{(m)}_t + ε

    Parameters
    ----------
    rv : pd.Series
        Daily realized variance series with DatetimeIndex.
    daily_lag, weekly_lag, monthly_lag : int
        Lag windows for daily, weekly, monthly components.

    Returns
    -------
    pd.DataFrame
        Columns: ['RV_d', 'RV_w', 'RV_m'] aligned to predict RV_{t+1}.
        Index matches the prediction dates (shifted forward by 1).
    """
    rv_d = rv.shift(daily_lag)                       # RV_{t-1}
    rv_w = rolling_mean(rv, weekly_lag).shift(1)     # RV^{(w)}_{t-1}
    rv_m = rolling_mean(rv, monthly_lag).shift(1)    # RV^{(m)}_{t-1}

    features = pd.DataFrame({
        'RV_d': rv_d,
        'RV_w': rv_w,
        'RV_m': rv_m,
    }, index=rv.index)

    return features


def build_har_j_features(
    rv: pd.Series,
    jump: pd.Series,
    daily_lag: int = 1,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
) -> pd.DataFrame:
    """Build HAR-J regressors (HAR + jump component).

    HAR-J: RV_{t+1} = β₀ + β₁·RV_t + β₂·RV^{(w)}_t + β₃·RV^{(m)}_t + β₄·J_t + ε
    where J_t = max(RV_t - BPV_t, 0)

    Parameters
    ----------
    rv : pd.Series
        Daily realized variance.
    jump : pd.Series
        Jump component = max(RV - BPV, 0).

    Returns
    -------
    pd.DataFrame
        Columns: ['RV_d', 'RV_w', 'RV_m', 'J_d'] for prediction dates.
    """
    har = build_har_features(rv, daily_lag, weekly_lag, monthly_lag)
    har['J_d'] = jump.shift(daily_lag)
    return har


def build_har_rs_features(
    good: pd.Series,
    bad: pd.Series,
    daily_lag: int = 1,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
) -> pd.DataFrame:
    """Build HAR-RS regressors (semivariance decomposition).

    HAR-RS (Patton & Sheppard 2015):
    RV_{t+1} = β₀ + β₁⁺·RS⁺_t + β₁⁻·RS⁻_t
                   + β₂⁺·RS⁺^{(w)}_t + β₂⁻·RS⁻^{(w)}_t
                   + β₃⁺·RS⁺^{(m)}_t + β₃⁻·RS⁻^{(m)}_t + ε

    Parameters
    ----------
    good : pd.Series
        Positive (good) semivariance.
    bad : pd.Series
        Negative (bad) semivariance.

    Returns
    -------
    pd.DataFrame
        Columns: ['RS_pos_d', 'RS_neg_d', 'RS_pos_w', 'RS_neg_w',
                   'RS_pos_m', 'RS_neg_m'].
    """
    features = pd.DataFrame({
        'RS_pos_d': good.shift(daily_lag),
        'RS_neg_d': bad.shift(daily_lag),
        'RS_pos_w': rolling_mean(good, weekly_lag).shift(1),
        'RS_neg_w': rolling_mean(bad, weekly_lag).shift(1),
        'RS_pos_m': rolling_mean(good, monthly_lag).shift(1),
        'RS_neg_m': rolling_mean(bad, monthly_lag).shift(1),
    }, index=good.index)

    return features


def build_harq_features(
    rv: pd.Series,
    rq: pd.Series,
    daily_lag: int = 1,
    weekly_lag: int = 5,
    monthly_lag: int = 22,
) -> pd.DataFrame:
    """Build HARQ regressors (HAR + measurement error adjustment).

    HARQ (Bollerslev, Patton & Quaedvlieg 2016):
    RV_{t+1} = β₀ + β₁·RV_t + β₁Q·RV_t·√RQ_t
                   + β₂·RV^{(w)}_t + β₃·RV^{(m)}_t + ε

    The interaction RV_t * sqrt(RQ_t) captures time-varying measurement error.

    Parameters
    ----------
    rv : pd.Series
        Daily realized variance.
    rq : pd.Series
        Realized quarticity.

    Returns
    -------
    pd.DataFrame
        Columns: ['RV_d', 'RV_w', 'RV_m', 'RV_RQ_interaction'].
    """
    har = build_har_features(rv, daily_lag, weekly_lag, monthly_lag)
    sqrt_rq = np.sqrt(rq).shift(daily_lag)
    har['RV_RQ_interaction'] = rv.shift(daily_lag) * sqrt_rq
    return har


def build_target(
    rv: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """Build forecast target: h-step-ahead RV (or average RV for h > 1).

    Parameters
    ----------
    rv : pd.Series
        Daily realized variance.
    horizon : int
        Forecast horizon. For h=1: RV_{t+1}. For h>1: average RV over next h days.

    Returns
    -------
    pd.Series
        Target variable aligned to the feature dates.
    """
    if horizon == 1:
        return rv  # Features are already shifted; target is current RV
    else:
        # Average RV over next h days (forward-looking)
        return rv.rolling(window=horizon, min_periods=horizon).mean().shift(-(horizon - 1))


def align_features_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop rows where features or target have NaN; return aligned pair.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Target variable.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Aligned features and target with no NaN values.
    """
    combined = features.copy()
    combined['_target'] = target
    combined = combined.dropna()
    X = combined.drop(columns='_target')
    y = combined['_target']
    return X, y
