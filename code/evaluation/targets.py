"""Scoring target and common sample shared by the evaluation scripts.

Every forecast file is indexed by the first day of its target: the row dated t holds the
forecast made with information through t - 1 for the target that starts on day t. Under the
averaged target the realized quantity is the mean daily realized variance over the h rows
from t on, which `holding_rv` computes from the raw series. Every model is scored against
that one quantity, whatever its file stores in `actual` (the square of the mean volatility
for most models, the mean variance for the augmented HARs).
"""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def holding_rv(rv: pd.Series, dates: Iterable, h: int) -> np.ndarray:
    """Mean daily RV over the h rows of `rv` that start at each date.

    `rv` is one asset's daily realized variance with missing days dropped, the series the
    forecasting engines index by row. At h = 1 this is RV on the date itself.
    """
    dates = pd.DatetimeIndex(dates)
    pos = rv.index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} dates are not in the RV series")
    if (pos + h > len(rv)).any():
        raise ValueError("a holding period runs past the end of the RV series")
    cs = np.concatenate([[0.0], np.cumsum(rv.to_numpy(dtype=float))])
    return (cs[pos + h] - cs[pos]) / h


def target_dates(index: pd.DatetimeIndex, dates: Iterable, h: int) -> pd.DatetimeIndex:
    """Date of the target of each forecast: the row h - 1 positions after the row of its date in the
    asset's series ``index`` (missing days dropped). At h = 1 it is the date itself."""
    dates = pd.DatetimeIndex(dates)
    pos = index.get_indexer(dates)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} dates are not in the series")
    if (pos + h - 1 >= len(index)).any():
        raise ValueError("a target falls past the end of the series")
    return index[pos + h - 1]


def rv_series_all(tickers: Optional[Iterable[str]] = None) -> Dict[str, pd.Series]:
    """Daily realized variance per asset, missing days dropped, from the three VOLARE files."""
    from config import VOLARE_ALL_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS
    from data_loader import load_data

    wanted = list(tickers) if tickers is not None else list(VOLARE_ALL_TICKERS)
    groups = [("volare", [t for t in wanted if t not in VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS]),
              ("volare_fx", [t for t in wanted if t in VOLARE_FX_TICKERS]),
              ("volare_futures", [t for t in wanted if t in VOLARE_FUTURES_TICKERS])]
    out = {}
    for key, tick in groups:
        if not tick:
            continue
        data = load_data(dataset=key, tickers=tick)
        for t in tick:
            out[t] = data.rv[t].dropna()
    return out


def common_dates(frames: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Dates on which every frame has a forecast, sorted."""
    idx = None
    for df in frames.values():
        k = df["forecast"].dropna().index
        idx = k if idx is None else idx.intersection(k)
    return pd.DatetimeIndex([]) if idx is None else idx.sort_values()
