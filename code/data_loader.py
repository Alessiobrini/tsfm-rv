"""
data_loader.py — Load and preprocess VOLARE realized volatility data.

Pipeline expects a `RVData` container with DatetimeIndex rows and ticker columns
for each volatility measure (RV, BPV, semivariances, RQ).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import (
    VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE,
    data_cfg,
)


@dataclass
class RVData:
    """Container for all realized volatility data used by models.

    Attributes:
        rv: Daily realized variance (primary target).
        log_rv: Log of realized variance.
        bpv: Bipower variation (for jump component).
        good: Positive (good) semivariance.
        bad: Negative (bad) semivariance.
        rq: Realized quarticity.
        jump: Jump component = max(RV - BPV, 0).
        dates: DatetimeIndex of trading dates.
        tickers: List of asset tickers.
    """
    rv: pd.DataFrame
    log_rv: pd.DataFrame
    bpv: pd.DataFrame
    good: pd.DataFrame
    bad: pd.DataFrame
    rq: pd.DataFrame
    jump: pd.DataFrame
    dates: pd.DatetimeIndex
    tickers: List[str]


def load_volare_data(
    filepath: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load VOLARE realized variance data from long-format CSV.

    Parameters
    ----------
    filepath : Path, optional
        Path to a VOLARE CSV. Defaults to VOLARE_STOCKS_FILE.
    tickers : List[str], optional
        Tickers to include. If None, loads all available.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys: 'RV', 'BPV', 'Good', 'Bad', 'RQ'.
        Each DataFrame: rows = dates (DatetimeIndex), columns = tickers.
    """
    if filepath is None:
        filepath = VOLARE_STOCKS_FILE

    col_map = {
        'RV': data_cfg.volare_rv_col,
        'BPV': data_cfg.volare_bpv_col,
        'Good': data_cfg.volare_good_col,
        'Bad': data_cfg.volare_bad_col,
        'RQ': data_cfg.volare_rq_col,
    }

    usecols = ['date', 'symbol'] + list(col_map.values())
    df = pd.read_csv(filepath, usecols=usecols, parse_dates=['date'])

    if tickers is not None:
        df = df[df['symbol'].isin(tickers)]

    raw = {}
    for key, col in col_map.items():
        pivoted = df.pivot(index='date', columns='symbol', values=col)
        pivoted.index = pd.DatetimeIndex(pivoted.index)
        pivoted = pivoted.sort_index()
        raw[key] = pivoted

    return raw


def preprocess(
    raw: Dict[str, pd.DataFrame],
    tickers: Optional[List[str]] = None,
    winsorize: bool = False,
    winsorize_pctile: float = 0.995,
) -> RVData:
    """Subset, optionally winsorize, and compute derived series."""
    if tickers is None:
        tickers = list(raw['RV'].columns)

    available = [t for t in tickers if t in raw['RV'].columns]
    if not available:
        available = list(raw['RV'].columns)
    rv = raw['RV'][available].copy()
    bpv = raw['BPV'][available].copy()
    good = raw['Good'][available].copy()
    bad = raw['Bad'][available].copy()
    rq = raw['RQ'][available].copy()

    if winsorize:
        for col in rv.columns:
            upper = rv[col].quantile(winsorize_pctile)
            rv[col] = rv[col].clip(upper=upper)

    log_rv = np.log(rv)
    jump = (rv - bpv).clip(lower=0)

    return RVData(
        rv=rv,
        log_rv=log_rv,
        bpv=bpv,
        good=good,
        bad=bad,
        rq=rq,
        jump=jump,
        dates=rv.index,
        tickers=available,
    )


def load_data(
    dataset: str = "volare",
    filepath: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> RVData:
    """Main entry point: load and preprocess VOLARE data in one call.

    Parameters
    ----------
    dataset : str
        One of "volare" (stocks), "volare_fx", or "volare_futures".
    filepath : Path, optional
        Override default CSV path for the chosen dataset.
    tickers : List[str], optional
        Subset of tickers.

    Returns
    -------
    RVData
        Ready-to-use data container.
    """
    if dataset == "volare_fx":
        raw = load_volare_data(filepath or VOLARE_FOREX_FILE, tickers=tickers)
    elif dataset == "volare_futures":
        raw = load_volare_data(filepath or VOLARE_FUTURES_FILE, tickers=tickers)
    else:
        raw = load_volare_data(filepath, tickers=tickers)

    return preprocess(
        raw,
        tickers=tickers,
        winsorize=data_cfg.winsorize,
        winsorize_pctile=data_cfg.winsorize_pctile,
    )
