"""
run_har_density_baseline.py — HAR + log-normal density baseline on VOLARE.

Produces the econometric density anchor for the TSFM density-evaluation
phase. For every (ticker, horizon) and both residual modes
{"gaussian", "empirical"}, walks forward with the paper's HAR convention
(252-day train, 126-day test, 126-day step), fits HARDensityModel on the
training fold, and persists the predictive quantile grid on the common
DEFAULT_QUANTILE_LEVELS for every test date.

Output (CSV — matches the codebase convention used by run_baselines_volare):
    results/volare/density/har_logn_<mode>/<ticker>_h<h>.csv
    columns: actual, q_0025, q_0050, q_0100, ..., q_0975
             (column name = round(level * 1000), zero-padded to 4 digits)

Also writes results/volare/density/har_logn_<mode>/_summary.csv with the
per-asset CRPS / KS / coverage / width.

CPU only; runs locally end-to-end.

Usage:
    python run_har_density_baseline.py                            # all 40 stocks
    python run_har_density_baseline.py --tickers AAPL JPM
    python run_har_density_baseline.py --horizons 1 22 --modes gaussian
    python run_har_density_baseline.py --asset-class fx --all-tickers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REPRESENTATIVE_TICKERS,
    VOLARE_FUTURES_TICKERS,
    VOLARE_FX_TICKERS,
    VOLARE_RESULTS_DIR,
    VOLARE_STOCK_TICKERS,
    forecast_cfg,
)
from data_loader import load_data
from evaluation.density import DEFAULT_QUANTILE_LEVELS, density_summary
from features import align_features_target, build_har_features, build_target
from forecasting.rolling_forecast import generate_walk_forward_folds
from models.har_density import HARDensityModel
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
DATASET_KEY = {"stocks": "volare", "fx": "volare_fx", "futures": "volare_futures"}


def _q_columns(levels: np.ndarray) -> List[str]:
    """Column naming convention: q_<level*1000 zero-padded to 4 digits>."""
    return [f"q_{int(round(level * 1000)):04d}" for level in levels]


def _output_dir(mode: str) -> Path:
    return DENSITY_DIR / f"har_logn_{mode}"


def _output_path(mode: str, ticker: str, horizon: int) -> Path:
    return _output_dir(mode) / f"{ticker}_h{horizon}.csv"


def _write_density(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _read_density(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def walk_forward_har_density(
    rv_series: pd.Series,
    horizon: int,
    residual_mode: str,
    train_window: int,
    test_window: int,
    step_size: int,
    levels: np.ndarray,
) -> pd.DataFrame:
    """Walk-forward Log-HAR predictive distributions over the common grid.

    Returns a DataFrame indexed by date with columns [actual, q_0025, ..., q_0975]
    holding level-space (positive) quantiles for every test date.
    """
    X = build_har_features(rv_series)
    y = build_target(rv_series, horizon=horizon)
    X, y = align_features_target(X, y)
    folds = generate_walk_forward_folds(
        n_obs=len(X),
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    )
    if not folds:
        raise ValueError(
            f"No folds: n_obs={len(X)}, train_window={train_window}, "
            f"test_window={test_window}"
        )

    q_cols = _q_columns(levels)
    rows: List[dict] = []
    seen: set = set()

    for ts, te, vs, ve in folds:
        X_train, y_train = X.iloc[ts:te], y.iloc[ts:te]
        X_test, y_test = X.iloc[vs:ve], y.iloc[vs:ve]

        model = HARDensityModel(residual_mode=residual_mode, levels=levels)
        model.fit(X_train, y_train)
        fc = model.predict_density(X_test)

        for i, date in enumerate(X_test.index):
            if date in seen:
                continue
            row: dict = {"date": date, "actual": float(y_test.iloc[i])}
            for k, col in enumerate(q_cols):
                row[col] = float(fc.level_quantiles[i, k])
            rows.append(row)
            seen.add(date)

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


def score_density_frame(df: pd.DataFrame, levels: np.ndarray) -> dict:
    """Compute CRPS / PIT-KS / coverage / width on both log and level scale."""
    actual = df["actual"].to_numpy()
    q_grid_level = df[_q_columns(levels)].to_numpy()
    q_grid_log = np.log(np.clip(q_grid_level, 1e-30, None))
    log_actual = np.log(np.clip(actual, 1e-30, None))

    log_summary = density_summary(log_actual, q_grid_log, levels)
    lvl_summary = density_summary(actual, q_grid_level, levels)

    out = {"n_obs": int(len(actual))}
    out.update({f"log_{k}": v for k, v in log_summary.to_dict().items() if k != "n_obs"})
    out.update({f"lvl_{k}": v for k, v in lvl_summary.to_dict().items() if k != "n_obs"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward HAR + log-normal density baseline on VOLARE"
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument(
        "--modes", nargs="+", default=["gaussian", "empirical"],
        choices=["gaussian", "empirical"],
    )
    parser.add_argument(
        "--asset-class", default="stocks",
        choices=["stocks", "fx", "futures"],
    )
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--train-window", type=int, default=None)
    parser.add_argument("--test-window", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip (ticker, horizon, mode) triples whose density file already exists.",
    )
    args = parser.parse_args()

    if args.all_tickers:
        tickers: Sequence[str] = {
            "stocks": VOLARE_STOCK_TICKERS,
            "fx": VOLARE_FX_TICKERS,
            "futures": VOLARE_FUTURES_TICKERS,
        }[args.asset_class]
    else:
        tickers = args.tickers or REPRESENTATIVE_TICKERS

    horizons = args.horizons or forecast_cfg.horizons
    train_window = args.train_window or forecast_cfg.train_window
    test_window = args.test_window or forecast_cfg.test_window
    step_size = args.step_size or forecast_cfg.step_size

    logger = setup_logger("har_density_baseline")
    logger.info("=== HAR + log-normal density baseline ===")
    logger.info(f"Tickers: {list(tickers)}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Modes: {args.modes}")
    logger.info(f"Window: train={train_window}, test={test_window}, step={step_size}")

    data = load_data(dataset=DATASET_KEY[args.asset_class], tickers=list(tickers))
    levels = DEFAULT_QUANTILE_LEVELS

    summary_rows: dict[str, list] = {m: [] for m in args.modes}

    total = len(tickers) * len(horizons) * len(args.modes)
    done = 0
    for mode in args.modes:
        for ticker in tickers:
            for horizon in horizons:
                done += 1
                tag = f"[{done}/{total}] HAR-density {mode} | {ticker} | h={horizon}"
                out_path = _output_path(mode, ticker, horizon)
                if args.skip_existing and out_path.exists():
                    logger.info(f"  Skipping {tag}: {out_path.name} exists")
                    df_cached = _read_density(out_path)
                    metrics = score_density_frame(df_cached, levels)
                    summary_rows[mode].append(
                        {"ticker": ticker, "horizon": horizon, **metrics}
                    )
                    continue

                logger.info(f"Running {tag}")
                t0 = time.time()
                try:
                    rv = data.rv[ticker].dropna()
                    df = walk_forward_har_density(
                        rv_series=rv,
                        horizon=horizon,
                        residual_mode=mode,
                        train_window=train_window,
                        test_window=test_window,
                        step_size=step_size,
                        levels=levels,
                    )
                    _write_density(df, out_path)
                    metrics = score_density_frame(df, levels)
                    summary_rows[mode].append(
                        {"ticker": ticker, "horizon": horizon, **metrics}
                    )
                    logger.info(
                        f"  Done {tag} in {time.time() - t0:.1f}s | "
                        f"n={metrics['n_obs']} | "
                        f"log CRPS={metrics['log_crps_mean']:.4f} | "
                        f"log KS={metrics['log_pit_ks_stat']:.3f} | "
                        f"cov80={metrics['log_coverage_80']:.2f}"
                    )
                except Exception as exc:
                    logger.error(f"  FAILED {tag}: {exc}")

        # Per-mode summary CSV
        if summary_rows[mode]:
            out_csv = _output_dir(mode) / "_summary.csv"
            _output_dir(mode).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_rows[mode]).to_csv(out_csv, index=False)
            logger.info(f"Wrote summary: {out_csv}")

    # Cross-mode comparison
    if any(summary_rows[m] for m in args.modes):
        logger.info("\n=== Cross-mode summary (mean across assets) ===")
        for mode in args.modes:
            df = pd.DataFrame(summary_rows[mode])
            if df.empty:
                continue
            for h in sorted(df["horizon"].unique()):
                sub = df[df["horizon"] == h]
                logger.info(
                    f"  {mode:>9s} h={int(h):>2d}  "
                    f"log CRPS={sub['log_crps_mean'].mean():.4f}  "
                    f"log KS={sub['log_pit_ks_stat'].mean():.3f}  "
                    f"cov50/80/95={sub['log_coverage_50'].mean():.2f}/"
                    f"{sub['log_coverage_80'].mean():.2f}/"
                    f"{sub['log_coverage_95'].mean():.2f}  "
                    f"width80={sub['log_width_80'].mean():.3f}"
                )

    logger.info("HAR-density baseline complete.")


if __name__ == "__main__":
    main()
