"""
run_mz_at_distribution.py — Phase 3: apply the median's MZ to every quantile.

For each persisted density forecast, fit recursive Mincer-Zarnowitz on
(actual, median forecast) and apply the resulting (alpha_hat_t, beta_hat_t)
SAME pair uniformly to every quantile of the predictive distribution. Re-score
the corrected grid and write a side-by-side comparison against the raw
density scores.

Headline question: does the affine correction that fixes the median's bias
also fix the rest of the distribution, or does it leave the intervals
miscovered?

Inputs:
    results/volare/density/<model>/<ticker>_h<h>.csv

Outputs:
    results/volare/density_mz/<model>/<ticker>_h<h>.csv     corrected grids
    results/volare/density_mz/mz_at_distribution.xlsx        comparison workbook
        sheet 'by_asset': per (model, ticker, horizon) raw vs MZ-corrected
        sheet 'summary':  cross-asset means by (model, horizon)
        sheet 'beta_warnings': blocks where beta_hat <= 0 fired the guard

Usage:
    python run_mz_at_distribution.py
    python run_mz_at_distribution.py --models chronos_bolt_base sundial
    python run_mz_at_distribution.py --min-window 252 --output ...
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR
from evaluation.density import (
    DEFAULT_QUANTILE_LEVELS,
    apply_affine_to_quantiles,
)
from evaluation.density_io import (
    DensityFile,
    discover_density_files,
    grid_to_density_frame,
    q_columns,
    read_density_csv,
    score_density_grid,
    split_actual_and_grid,
    write_density_csv,
)
from evaluation.mz_regression import recursive_mz_coefficients
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
MZ_DIR = VOLARE_RESULTS_DIR / "density_mz"


def _median_column(levels: np.ndarray) -> str:
    cols = q_columns(levels)
    idx = int(np.argmin(np.abs(levels - 0.5)))
    return cols[idx]


def correct_one_file(
    f: DensityFile,
    levels: np.ndarray,
    min_window: int,
    out_root: Path,
) -> Optional[Dict[str, object]]:
    """Apply recursive MZ to one density file. Returns (metrics) dict or None.

    Side effect: writes the MZ-corrected grid under
    results/volare/density_mz/<model>/<ticker>_h<h>.csv.
    """
    df = read_density_csv(f.path)
    actuals, q_grid = split_actual_and_grid(df, levels)

    median_col = _median_column(levels)
    median_forecast = df[median_col].to_numpy(dtype=float)

    alphas, betas = recursive_mz_coefficients(
        actuals, median_forecast, min_window=min_window
    )
    if len(alphas) == 0:
        return None

    bad_beta_mask = ~np.isfinite(betas) | (betas <= 0)
    bad_beta_count = int(bad_beta_mask.sum())
    if bad_beta_count == len(betas):
        return None  # No usable correction.

    # Apply per-date (alpha_t, beta_t) to every quantile of forecast at t.
    test_actuals = actuals[min_window:]
    test_grid = q_grid[min_window:]
    test_dates = df.index[min_window:]

    # Suppress the per-row warning storm; we surface the count separately.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corrected_grid = apply_affine_to_quantiles(
            test_grid,
            alpha=alphas,
            beta=betas,
            on_nonpositive_beta="ignore",
        )

    out_path = out_root / f.model / f.path.name
    write_density_csv(
        grid_to_density_frame(test_dates, test_actuals, corrected_grid, levels),
        out_path,
    )

    raw_metrics = score_density_grid(test_actuals, test_grid, levels)
    mz_metrics = score_density_grid(test_actuals, corrected_grid, levels)

    row: Dict[str, object] = {
        "model": f.model,
        "ticker": f.ticker,
        "horizon": f.horizon,
        "context_length": f.context_length or 512,
        "n_corrected": int(len(test_actuals)),
        "beta_min": float(np.nanmin(betas)) if np.any(np.isfinite(betas)) else float("nan"),
        "beta_mean": float(np.nanmean(betas)) if np.any(np.isfinite(betas)) else float("nan"),
        "beta_max": float(np.nanmax(betas)) if np.any(np.isfinite(betas)) else float("nan"),
        "alpha_mean": float(np.nanmean(alphas)) if np.any(np.isfinite(alphas)) else float("nan"),
        "n_beta_nonpositive": bad_beta_count,
        "beta_nonpositive_pct": 100.0 * bad_beta_count / max(len(betas), 1),
    }
    # Drop bin-count strings: they appear in the Phase-2 harness, not here.
    raw_metrics.pop("log_pit_bin_counts", None)
    raw_metrics.pop("lvl_pit_bin_counts", None)
    mz_metrics.pop("log_pit_bin_counts", None)
    mz_metrics.pop("lvl_pit_bin_counts", None)

    row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
    row.update({f"mz_{k}": v for k, v in mz_metrics.items()})

    # Convenience deltas on the most-watched metrics.
    for key in ("log_crps_mean", "log_pit_ks_stat", "log_coverage_50",
                "log_coverage_80", "log_coverage_95"):
        if f"raw_{key}" in row and f"mz_{key}" in row:
            row[f"delta_{key}"] = row[f"mz_{key}"] - row[f"raw_{key}"]
    return row


def cross_asset_summary(by_asset: pd.DataFrame) -> pd.DataFrame:
    """Mean across assets per (model, horizon, context_length) on headline columns."""
    if by_asset.empty:
        return by_asset

    numeric_cols = [c for c in by_asset.columns
                    if pd.api.types.is_numeric_dtype(by_asset[c])
                    and c not in {"horizon", "context_length"}]
    grouped = by_asset.groupby(["model", "horizon", "context_length"], dropna=False)
    return grouped[numeric_cols].mean().reset_index().sort_values(
        ["model", "horizon", "context_length"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply recursive MZ to every quantile of the predictive density")
    parser.add_argument("--root", default=str(DENSITY_DIR))
    parser.add_argument("--out-root", default=str(MZ_DIR))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--min-window", type=int, default=252,
                        help="Min observations before MZ correction starts (default 252)")
    parser.add_argument("--output", default=None, help="Excel workbook path")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    output = Path(args.output) if args.output else out_root / "mz_at_distribution.xlsx"

    logger = setup_logger("mz_at_distribution")
    logger.info("=== Phase 3: MZ at the distribution level ===")
    logger.info(f"Source root: {root}")
    logger.info(f"Output root: {out_root}")
    logger.info(f"Min window:  {args.min_window}")

    files = discover_density_files(
        root, models=args.models, tickers=args.tickers, horizons=args.horizons,
    )
    if not files:
        logger.warning("No density CSVs found.")
        return
    logger.info(f"Found {len(files)} density files.")

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            row = correct_one_file(f, DEFAULT_QUANTILE_LEVELS, args.min_window, out_root)
        except Exception as exc:
            logger.error(f"  [{i}/{len(files)}] FAILED {f.model}/{f.ticker} h={f.horizon}: {exc}")
            continue
        if row is None:
            logger.warning(f"  [{i}/{len(files)}] skipped {f.model}/{f.ticker} h={f.horizon}: insufficient correction")
            continue
        rows.append(row)
        if i % 25 == 0 or i == len(files):
            logger.info(f"  processed {i}/{len(files)} ({time.time() - t0:.1f}s elapsed)")

    if not rows:
        logger.warning("No files corrected; nothing to write.")
        return

    by_asset = pd.DataFrame(rows)
    summary = cross_asset_summary(by_asset)

    # Beta-guard summary: count of asset-level runs with any non-positive beta.
    beta_warnings = by_asset[by_asset["n_beta_nonpositive"] > 0][
        ["model", "ticker", "horizon", "context_length", "n_beta_nonpositive",
         "beta_nonpositive_pct", "beta_min", "beta_mean"]
    ].sort_values(["model", "n_beta_nonpositive"], ascending=[True, False])

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_asset.sort_values(["model", "ticker", "horizon"]).to_excel(
            writer, sheet_name="by_asset", index=False
        )
        beta_warnings.to_excel(writer, sheet_name="beta_warnings", index=False)
    logger.info(f"Wrote workbook: {output}")

    logger.info("\n=== Headline raw vs MZ-corrected (log space, cross-asset means) ===")
    for _, row in summary.iterrows():
        cov_raw = "/".join(f"{row.get(f'raw_log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        cov_mz = "/".join(f"{row.get(f'mz_log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        crps_raw = row.get("raw_log_crps_mean", float("nan"))
        crps_mz = row.get("mz_log_crps_mean", float("nan"))
        logger.info(
            f"  {row['model']:>22s} h={int(row['horizon']):>2d}  "
            f"CRPS raw={crps_raw:.4f} -> MZ={crps_mz:.4f}  "
            f"cov50/80/95 raw={cov_raw} -> MZ={cov_mz}  "
            f"beta(mean={row.get('beta_mean', float('nan')):.3f}, "
            f"<=0 in {row.get('beta_nonpositive_pct', 0):.1f}% of blocks)"
        )
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
