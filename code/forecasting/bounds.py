"""Winsorization bounds known at the forecast origin.

Every forecast made at origin date p is clipped to the range of the asset's series over the
``window`` observations before p (rows p - window to p - 1), so the bounds use only data observed
before the origin. Near the start of the sample, where fewer than ``window`` rows exist, the range
covers every row before p. The same rule applies to every model. Non-positive values are left out
of the lower bound, so a forecast is never clipped to zero, where QLIKE is undefined.
"""
import numpy as np
import pandas as pd

BOUNDS_WINDOW = 1000


def window_bounds(series: pd.Series, window: int = BOUNDS_WINDOW):
    """Lower and upper bound at each date of ``series``: min and max of the ``window`` rows
    before that date. NaN at the first date, which has no history."""
    s = pd.Series(np.asarray(series, dtype=float), index=series.index)
    lo = s.where(s > 0).rolling(window, min_periods=1).min().shift(1)
    hi = s.rolling(window, min_periods=1).max().shift(1)
    return lo, hi


def clip_to_window(forecast: pd.Series, series: pd.Series, window: int = BOUNDS_WINDOW) -> pd.Series:
    """Clip each forecast to the bounds of ``series`` at its origin date (the forecast's index).

    ``series`` is on the forecast's scale (volatility for volatility forecasts) and its index holds
    every origin date of ``forecast``.
    """
    lo, hi = window_bounds(series, window)
    lo, hi = lo.reindex(forecast.index), hi.reindex(forecast.index)
    if lo.isna().any() or hi.isna().any():
        raise ValueError("a forecast date is missing from the series index or has no history before it")
    return forecast.clip(lower=lo, upper=hi)
