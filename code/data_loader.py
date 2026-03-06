"""
data_loader.py — Load and preprocess realized volatility data.

=== VOLARE SWAP-IN POINT ===
When VOLARE data arrives, modify ONLY the `load_raw_data()` function below.
The rest of the pipeline expects the same output format:
    - Dict[str, pd.DataFrame] where keys are measure names (RV, BPV, Good, Bad, RQ)
    - Each DataFrame has DatetimeIndex (rows) and ticker columns
    - Missing data represented as NaN (not zeros)
================================

All downstream code (features.py, models/, etc.) calls `load_data()` which
returns a clean, preprocessed RVData object.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import (
    DATA_FILE, PROCESSED_DIR, MAIN_TICKERS, REPRESENTATIVE_TICKERS,
    VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE,
    VOLARE_COV_STOCKS_FILE, VOLARE_COV_FOREX_FILE, VOLARE_COV_FUTURES_FILE,
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
    data_cfg, RANDOM_SEED,
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


def load_raw_data(filepath: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load raw data from Excel file and return dict of DataFrames.

    *** VOLARE SWAP-IN POINT ***
    Replace the body of this function to load from Parquet/VOLARE format.
    Output format must remain: Dict[str, DataFrame] with DatetimeIndex rows,
    ticker columns, and NaN for missing values.

    Parameters
    ----------
    filepath : Path, optional
        Path to the data file. Defaults to config.DATA_FILE.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys: 'RV', 'BPV', 'Good', 'Bad', 'RQ' (and optionally 5-min variants).
        Each DataFrame: rows = dates (DatetimeIndex), columns = tickers (str).
    """
    if filepath is None:
        filepath = DATA_FILE

    # --- Load metadata ---
    dates_df = pd.read_excel(filepath, sheet_name='Dates', header=None)
    companies_df = pd.read_excel(filepath, sheet_name='Companies', header=None)
    dates = pd.to_datetime(dates_df.iloc[:, 0])
    tickers = companies_df.iloc[:, 0].tolist()

    # --- Load measure sheets ---
    measure_sheets = {
        'RV': data_cfg.rv_measure,
        'BPV': data_cfg.bpv_measure,
        'Good': data_cfg.good_measure,
        'Bad': data_cfg.bad_measure,
        'RQ': data_cfg.rq_measure,
    }

    raw = {}
    for key, sheet_name in measure_sheets.items():
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
        df.index = dates
        df.columns = tickers
        df = df.apply(pd.to_numeric, errors='coerce')

        # Replace zeros with NaN (missing data convention)
        if data_cfg.zero_is_missing:
            df = df.replace(0.0, np.nan)

        raw[key] = df

    return raw


def load_volare_data(
    filepath: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load VOLARE realized variance data from long-format CSV.

    Parameters
    ----------
    filepath : Path, optional
        Path to the VOLARE stocks CSV. Defaults to VOLARE_STOCKS_FILE.
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

    # Column mapping: internal name -> VOLARE column
    col_map = {
        'RV': data_cfg.volare_rv_col,
        'BPV': data_cfg.volare_bpv_col,
        'Good': data_cfg.volare_good_col,
        'Bad': data_cfg.volare_bad_col,
        'RQ': data_cfg.volare_rq_col,
    }

    # Read only needed columns
    usecols = ['date', 'symbol'] + list(col_map.values())
    df = pd.read_csv(filepath, usecols=usecols, parse_dates=['date'])

    # Filter tickers if requested
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
    """Clean and preprocess raw data into RVData container.

    Parameters
    ----------
    raw : Dict[str, pd.DataFrame]
        Output from load_raw_data().
    tickers : List[str], optional
        Subset of tickers to include. Defaults to MAIN_TICKERS.
    winsorize : bool
        Whether to winsorize extreme RV values.
    winsorize_pctile : float
        Upper percentile for winsorization.

    Returns
    -------
    RVData
        Preprocessed data container ready for model estimation.
    """
    if tickers is None:
        tickers = MAIN_TICKERS

    # Subset to requested tickers; fall back to all available if none match
    available = [t for t in tickers if t in raw['RV'].columns]
    if not available:
        available = list(raw['RV'].columns)
    rv = raw['RV'][available].copy()
    bpv = raw['BPV'][available].copy()
    good = raw['Good'][available].copy()
    bad = raw['Bad'][available].copy()
    rq = raw['RQ'][available].copy()

    # Winsorize if requested
    if winsorize:
        for col in rv.columns:
            upper = rv[col].quantile(winsorize_pctile)
            rv[col] = rv[col].clip(upper=upper)

    # Compute derived quantities
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
    dataset: str = "capire",
    filepath: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
) -> RVData:
    """Main entry point: load and preprocess data in one call.

    Parameters
    ----------
    dataset : str
        Which dataset to load: "capire" or "volare".
    filepath : Path, optional
        Path to data file (overrides default for chosen dataset).
    tickers : List[str], optional
        Subset of tickers.

    Returns
    -------
    RVData
        Ready-to-use data container.
    """
    if dataset == "volare":
        raw = load_volare_data(filepath, tickers=tickers)
    elif dataset == "volare_fx":
        raw = load_volare_data(filepath or VOLARE_FOREX_FILE, tickers=tickers)
    elif dataset == "volare_futures":
        raw = load_volare_data(filepath or VOLARE_FUTURES_FILE, tickers=tickers)
    else:
        raw = load_raw_data(filepath)

    return preprocess(
        raw,
        tickers=tickers,
        winsorize=data_cfg.winsorize,
        winsorize_pctile=data_cfg.winsorize_pctile,
    )


@dataclass
class CovData:
    """Container for realized covariance data.

    Attributes:
        matrices: Dict mapping date -> np.ndarray (N x N covariance matrix).
        assets: Sorted list of asset tickers.
        dates: Sorted list of dates with complete matrices.
        pair_series: Dict mapping (asset1, asset2) -> pd.Series of rcov values.
    """
    matrices: Dict[str, np.ndarray]
    assets: List[str]
    dates: List
    pair_series: Dict


def load_covariance_data(
    asset_class: str = "stocks",
    filepath: Optional[Path] = None,
    cov_col: str = "rcov",
) -> CovData:
    """Load VOLARE realized covariance data and build daily covariance matrices.

    Parameters
    ----------
    asset_class : str
        One of "stocks", "forex", "futures".
    filepath : Path, optional
        Override default CSV path.
    cov_col : str
        Column to use for covariance values (default: "rcov").

    Returns
    -------
    CovData
        Container with daily covariance matrices and per-pair series.
    """
    if filepath is None:
        filepath = {
            "stocks": VOLARE_COV_STOCKS_FILE,
            "forex": VOLARE_COV_FOREX_FILE,
            "futures": VOLARE_COV_FUTURES_FILE,
        }[asset_class]

    df = pd.read_csv(filepath, usecols=['date', 'asset1', 'asset2', cov_col],
                     parse_dates=['date'])

    # Get sorted asset list
    assets = sorted(set(df['asset1'].unique()) | set(df['asset2'].unique()))
    n = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}

    # Build per-pair series
    pair_series = {}
    for (a1, a2), group in df.groupby(['asset1', 'asset2']):
        s = group.set_index('date')[cov_col].sort_index()
        pair_series[(a1, a2)] = s
        if a1 != a2:
            pair_series[(a2, a1)] = s  # symmetric

    # Build daily matrices
    dates_all = sorted(df['date'].unique())
    matrices = {}
    valid_dates = []

    for date in dates_all:
        day_df = df[df['date'] == date]
        mat = np.zeros((n, n))
        for _, row in day_df.iterrows():
            i = asset_idx[row['asset1']]
            j = asset_idx[row['asset2']]
            val = row[cov_col]
            mat[i, j] = val
            mat[j, i] = val  # symmetric
        # Check completeness: diagonal should be nonzero for all assets
        if np.all(np.diag(mat) > 0):
            matrices[date] = mat
            valid_dates.append(date)

    return CovData(
        matrices=matrices,
        assets=assets,
        dates=valid_dates,
        pair_series=pair_series,
    )


def save_processed(data: RVData, output_dir: Optional[Path] = None) -> None:
    """Save processed data to CSV for reproducibility.

    Parameters
    ----------
    data : RVData
        Preprocessed data to save.
    output_dir : Path, optional
        Directory to save CSVs. Defaults to config.PROCESSED_DIR.
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    data.rv.to_csv(output_dir / "rv.csv")
    data.log_rv.to_csv(output_dir / "log_rv.csv")
    data.bpv.to_csv(output_dir / "bpv.csv")
    data.good.to_csv(output_dir / "good_semivar.csv")
    data.bad.to_csv(output_dir / "bad_semivar.csv")
    data.rq.to_csv(output_dir / "rq.csv")
    data.jump.to_csv(output_dir / "jump.csv")
    print(f"Saved processed data to {output_dir}")
