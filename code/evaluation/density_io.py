"""
evaluation/density_io.py — Shared loaders for the density results layout.

Used by the three Phase-2/3/4 runners:
    run_density_evaluation.py
    run_mz_at_distribution.py
    run_density_recalibration.py

The on-disk layout is uniform across HAR-density and TSFM-density:
    results/volare/density/<model>/<ticker>_h<H>.csv
    columns: actual, q_0025, q_0050, ..., q_0975
             (column name = round(level * 1000), zero-padded to 4 digits)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.density import DEFAULT_QUANTILE_LEVELS, density_summary, DensityScores

_DENSITY_FILE_RE = re.compile(r"^(?P<ticker>[A-Z0-9_.+-]+)_h(?P<horizon>\d+)(?:_ctx(?P<ctx>\d+))?\.csv$")


@dataclass(frozen=True)
class DensityFile:
    """One persisted density forecast file."""

    model: str
    ticker: str
    horizon: int
    context_length: Optional[int]
    path: Path


# ----------------------------------------------------------------------
# Column / file conventions
# ----------------------------------------------------------------------

def q_columns(levels: np.ndarray = DEFAULT_QUANTILE_LEVELS) -> List[str]:
    """Column names corresponding to a quantile level grid."""
    return [f"q_{int(round(level * 1000)):04d}" for level in levels]


def density_path(
    root: Path,
    model: str,
    ticker: str,
    horizon: int,
    context_length: Optional[int] = None,
) -> Path:
    """Canonical path for a density CSV under results/volare/density/<model>/."""
    suffix = "" if context_length is None or context_length == 512 else f"_ctx{context_length}"
    return root / model / f"{ticker}_h{horizon}{suffix}.csv"


# ----------------------------------------------------------------------
# Discovery and IO
# ----------------------------------------------------------------------

def discover_density_files(
    root: Path,
    models: Optional[Sequence[str]] = None,
    tickers: Optional[Sequence[str]] = None,
    horizons: Optional[Sequence[int]] = None,
) -> List[DensityFile]:
    """Walk results/volare/density/ and enumerate every per-asset density CSV.

    Filters by model / ticker / horizon when those arguments are passed.
    Skips files whose names do not match the canonical pattern (incl. any
    `_summary.csv` aggregates).
    """
    root = Path(root)
    if not root.exists():
        return []

    model_filter = set(models) if models else None
    ticker_filter = set(tickers) if tickers else None
    horizon_filter = set(int(h) for h in horizons) if horizons else None

    found: List[DensityFile] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        if model_filter is not None and model not in model_filter:
            continue
        for csv_path in sorted(model_dir.glob("*.csv")):
            match = _DENSITY_FILE_RE.match(csv_path.name)
            if match is None:
                continue
            ticker = match.group("ticker")
            horizon = int(match.group("horizon"))
            ctx = int(match.group("ctx")) if match.group("ctx") else None
            if ticker_filter is not None and ticker not in ticker_filter:
                continue
            if horizon_filter is not None and horizon not in horizon_filter:
                continue
            found.append(DensityFile(model=model, ticker=ticker, horizon=horizon,
                                     context_length=ctx, path=csv_path))
    return found


def read_density_csv(path: Path) -> pd.DataFrame:
    """Read a density CSV with date index parsed."""
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def split_actual_and_grid(
    df: pd.DataFrame,
    levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (actuals, quantile_grid) arrays for downstream scoring."""
    cols = q_columns(levels)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Density frame is missing expected quantile columns: {missing[:5]}..."
        )
    actuals = df["actual"].to_numpy(dtype=float)
    q_grid = df[cols].to_numpy(dtype=float)
    return actuals, q_grid


def write_density_csv(
    df: pd.DataFrame,
    path: Path,
) -> Path:
    """Persist a density DataFrame in the canonical CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def grid_to_density_frame(
    dates: Iterable,
    actuals: np.ndarray,
    q_grid: np.ndarray,
    levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> pd.DataFrame:
    """Inverse of split_actual_and_grid: build a canonical density frame."""
    cols = q_columns(levels)
    df = pd.DataFrame(q_grid, index=pd.Index(list(dates), name="date"), columns=cols)
    df.insert(0, "actual", np.asarray(actuals, dtype=float))
    return df


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def score_density_grid(
    actuals: np.ndarray,
    q_grid: np.ndarray,
    levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> Dict[str, float]:
    """Run density_summary in BOTH log and level space, return flat dict.

    Log-space is primary (Brini's reply: PIT / coverage scale-invariant
    under monotone transform; level-CRPS dominated by COVID spikes).
    Level-space reported as secondary so readers expecting absolute RV
    numbers can cross-check.
    """
    log_actual = np.log(np.clip(actuals, 1e-30, None))
    log_grid = np.log(np.clip(q_grid, 1e-30, None))

    log_summary = density_summary(log_actual, log_grid, levels)
    lvl_summary = density_summary(actuals, q_grid, levels)

    out = {"n_obs": int(len(actuals))}
    out.update({f"log_{k}": v for k, v in log_summary.to_dict().items() if k != "n_obs"})
    out.update({f"lvl_{k}": v for k, v in lvl_summary.to_dict().items() if k != "n_obs"})
    return out


def score_density_frame(
    df: pd.DataFrame,
    levels: np.ndarray = DEFAULT_QUANTILE_LEVELS,
) -> Dict[str, float]:
    """Convenience wrapper around score_density_grid for a DataFrame."""
    actuals, q_grid = split_actual_and_grid(df, levels)
    return score_density_grid(actuals, q_grid, levels)
