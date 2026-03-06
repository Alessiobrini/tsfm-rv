"""
utils.py — Helper functions for logging, timing, and IO.
"""

import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from functools import wraps
from typing import Optional

from config import RESULTS_DIR


def setup_logger(
    name: str = "rv_forecast",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str, optional
        Path to log file.
    level : int
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.1f}s")
        return result
    return wrapper


def save_forecasts(
    actual: pd.Series,
    forecasts: dict,
    ticker: str,
    horizon: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save forecast results to CSV.

    Parameters
    ----------
    actual : pd.Series
        Actual RV values.
    forecasts : dict
        Model name -> forecast Series.
    ticker : str
        Asset ticker.
    horizon : int
        Forecast horizon.
    output_dir : Path, optional
        Output directory. Defaults to RESULTS_DIR / 'forecasts'.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({'actual': actual})
    for name, fcast in forecasts.items():
        df[name] = fcast

    filepath = output_dir / f"forecasts_{ticker}_h{horizon}.csv"
    df.to_csv(filepath)
    return filepath


def save_metrics(
    metrics: pd.DataFrame,
    filename: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save evaluation metrics to CSV.

    Parameters
    ----------
    metrics : pd.DataFrame
        Metrics table.
    filename : str
        Output filename.
    output_dir : Path, optional
        Output directory. Defaults to RESULTS_DIR / 'metrics'.

    Returns
    -------
    Path
        Path to saved CSV file.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    metrics.to_csv(filepath)
    return filepath
