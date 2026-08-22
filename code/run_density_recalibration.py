"""
run_density_recalibration.py — Phase 4: distributional recalibration.

For every persisted density forecast, walk a rolling 756/126/126
(min_train, validation, test) block schedule. Inside each block:

  Isotonic (Kuleshov, Fenner & Ermon 2018)
    - Compute PIT on the validation window using the RAW grid.
    - fit_isotonic_recalibration() learns H_hat (empirical CDF of cal PIT).
    - apply_isotonic_recalibration_to_quantiles() relabels the test block's
      quantile grid to make PIT uniform.

  Split-conformal CQR (Romano, Patterson & Candes 2019)
    - For each nominal level in {0.50, 0.80, 0.95}:
        offset = split_conformal_offset(cal_actuals, cal_lower, cal_upper, nl)
        test_lower = test_q_low  - offset
        test_upper = test_q_high + offset

CRITICAL: calibration is fit PER BLOCK, not globally. Exchangeability is
violated by volatility clustering and regime drift, so a single global
fit would assume stationarity we know is broken. Treat "does conformal
still help under drift" as the empirical question (Brini's note).

Inputs:
    results/volare/density/<model>/<ticker>_h<h>.csv

Outputs:
    results/volare/density_iso/<model>/<ticker>_h<h>.csv
        full isotonic-recalibrated grid (stitched across test blocks)
    results/volare/density_cqr/<model>/<ticker>_h<h>.csv
        per-date {lower_50, upper_50, lower_80, upper_80, lower_95, upper_95}
    results/volare/density_recal/recalibration_comparison.xlsx
        raw vs isotonic vs CQR (and MZ if Phase 3 output is present)

Usage:
    python run_density_recalibration.py
    python run_density_recalibration.py --models chronos_bolt_base
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR
from evaluation.density import (
    DEFAULT_QUANTILE_LEVELS,
    apply_conformal_to_quantiles,
    apply_isotonic_recalibration_to_quantiles,
    fit_isotonic_recalibration,
    interval_coverage,
    interval_width,
    pit_values,
    tail_exceedance,
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
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
MZ_DIR = VOLARE_RESULTS_DIR / "density_mz"
ISO_DIR = VOLARE_RESULTS_DIR / "density_iso"
CQR_DIR = VOLARE_RESULTS_DIR / "density_cqr"
RECAL_DIR = VOLARE_RESULTS_DIR / "density_recal"

# Block schedule shared with the FT-Head archive: 756 / 126 / 126.
DEFAULT_MIN_TRAIN = 756
DEFAULT_VAL_SIZE = 126
DEFAULT_TEST_SIZE = 126
DEFAULT_STEP_SIZE = 126

NOMINAL_LEVELS = (0.50, 0.80, 0.95)


@dataclass(frozen=True)
class Block:
    """One rolling validation/test block of the recalibration schedule."""

    refit_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def generate_blocks(
    n_obs: int,
    min_train: int = DEFAULT_MIN_TRAIN,
    val_size: int = DEFAULT_VAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
) -> List[Block]:
    """Expanding-train rolling blocks aligned with the FT-Head schedule."""
    blocks: List[Block] = []
    train_start = 0
    train_end = min_train
    refit_id = 0
    while True:
        val_start = train_end
        val_end = val_start + val_size
        test_start = val_end
        test_end = test_start + test_size
        if test_end > n_obs:
            break
        blocks.append(Block(
            refit_id=refit_id,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            test_start=test_start, test_end=test_end,
        ))
        refit_id += 1
        train_end += step_size
    return blocks


# ----------------------------------------------------------------------
# Recalibration over the block schedule
# ----------------------------------------------------------------------

def recalibrate_one_file(
    f: DensityFile,
    levels: np.ndarray,
    blocks_cfg: Dict[str, int],
    iso_root: Path,
    cqr_root: Path,
    mz_root: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """Apply isotonic + CQR per block to one density file. Returns metrics row."""
    df = read_density_csv(f.path)
    actuals, q_grid = split_actual_and_grid(df, levels)
    n = len(actuals)

    blocks = generate_blocks(n, **blocks_cfg)
    if not blocks:
        return None

    iso_test_actuals: List[np.ndarray] = []
    iso_test_grid: List[np.ndarray] = []
    iso_test_dates: List[pd.DatetimeIndex] = []

    cqr_rows: List[Dict[str, object]] = []
    cqr_coverage: Dict[float, List[bool]] = {nl: [] for nl in NOMINAL_LEVELS}
    cqr_widths: Dict[float, List[float]] = {nl: [] for nl in NOMINAL_LEVELS}
    raw_coverage_test_only: Dict[float, List[bool]] = {nl: [] for nl in NOMINAL_LEVELS}
    raw_widths_test_only: Dict[float, List[float]] = {nl: [] for nl in NOMINAL_LEVELS}

    for blk in blocks:
        cal_actuals = actuals[blk.val_start:blk.val_end]
        cal_grid = q_grid[blk.val_start:blk.val_end]
        test_actuals = actuals[blk.test_start:blk.test_end]
        test_grid = q_grid[blk.test_start:blk.test_end]
        test_dates = df.index[blk.test_start:blk.test_end]

        # ---- Isotonic ---------------------------------------------------
        cal_pit = pit_values(cal_actuals, cal_grid, levels, enforce_monotone=False)
        cal_pit_sorted, cal_h = fit_isotonic_recalibration(cal_pit)
        iso_grid, _ = apply_isotonic_recalibration_to_quantiles(
            test_grid, levels, cal_pit_sorted, cal_h
        )

        iso_test_actuals.append(test_actuals)
        iso_test_grid.append(iso_grid)
        iso_test_dates.append(test_dates)

        # ---- CQR per nominal level -------------------------------------
        cqr_blocks = apply_conformal_to_quantiles(
            test_grid, levels, cal_actuals, cal_grid, nominal_levels=NOMINAL_LEVELS,
        )
        block_row: Dict[str, object] = {
            "refit_id": blk.refit_id,
            "test_start": str(df.index[blk.test_start].date()),
            "test_end": str(df.index[blk.test_end - 1].date()),
            "n_test": int(len(test_actuals)),
        }
        for nl in NOMINAL_LEVELS:
            lower = cqr_blocks[nl][:, 0]
            upper = cqr_blocks[nl][:, 1]
            cov = (test_actuals >= lower) & (test_actuals <= upper)
            width = upper - lower
            cqr_coverage[nl].extend(cov.tolist())
            cqr_widths[nl].extend(width.tolist())

            raw_cov = interval_coverage(test_actuals, test_grid, nl, levels)
            raw_w = interval_width(test_grid, nl, levels)
            raw_coverage_test_only[nl].extend([raw_cov] * len(test_actuals))
            raw_widths_test_only[nl].extend([raw_w] * len(test_actuals))

            tag = int(round(nl * 100))
            block_row[f"cqr_lower_{tag}_mean"] = float(np.mean(lower))
            block_row[f"cqr_upper_{tag}_mean"] = float(np.mean(upper))
            block_row[f"cqr_cov_{tag}"] = float(np.mean(cov))
            block_row[f"cqr_width_{tag}"] = float(np.mean(width))
        cqr_rows.append(block_row)

    # Stitch isotonic test blocks into a single time series, save, score.
    iso_actuals_all = np.concatenate(iso_test_actuals)
    iso_grid_all = np.concatenate(iso_test_grid, axis=0)
    iso_dates_all = pd.DatetimeIndex(np.concatenate([d.to_numpy() for d in iso_test_dates]))

    iso_path = iso_root / f.model / f.path.name
    write_density_csv(
        grid_to_density_frame(iso_dates_all, iso_actuals_all, iso_grid_all, levels),
        iso_path,
    )

    # Save the per-block CQR intervals (date-resolved).
    cqr_per_date_rows: List[Dict[str, object]] = []
    cursor = 0
    for blk in blocks:
        cqr_blocks = apply_conformal_to_quantiles(
            q_grid[blk.test_start:blk.test_end], levels,
            actuals[blk.val_start:blk.val_end],
            q_grid[blk.val_start:blk.val_end],
            nominal_levels=NOMINAL_LEVELS,
        )
        for t, date in enumerate(df.index[blk.test_start:blk.test_end]):
            row: Dict[str, object] = {"date": date, "actual": float(actuals[blk.test_start + t])}
            for nl in NOMINAL_LEVELS:
                tag = int(round(nl * 100))
                row[f"lower_{tag}"] = float(cqr_blocks[nl][t, 0])
                row[f"upper_{tag}"] = float(cqr_blocks[nl][t, 1])
            cqr_per_date_rows.append(row)
        cursor += blk.test_end - blk.test_start
    cqr_df = pd.DataFrame(cqr_per_date_rows).set_index("date").sort_index()
    (cqr_root / f.model).mkdir(parents=True, exist_ok=True)
    cqr_df.to_csv(cqr_root / f.model / f.path.name)

    # Score raw vs isotonic on the SAME stitched window for apples-to-apples.
    raw_metrics_window = score_density_grid(
        iso_actuals_all,
        np.concatenate([q_grid[blk.test_start:blk.test_end] for blk in blocks], axis=0),
        levels,
    )
    iso_metrics = score_density_grid(iso_actuals_all, iso_grid_all, levels)

    for d in (raw_metrics_window, iso_metrics):
        d.pop("log_pit_bin_counts", None)
        d.pop("lvl_pit_bin_counts", None)

    row: Dict[str, object] = {
        "model": f.model,
        "ticker": f.ticker,
        "horizon": f.horizon,
        "context_length": f.context_length or 512,
        "n_blocks": len(blocks),
        "n_test_total": int(len(iso_actuals_all)),
    }
    row.update({f"raw_{k}": v for k, v in raw_metrics_window.items()})
    row.update({f"iso_{k}": v for k, v in iso_metrics.items()})

    # CQR aggregates: realised coverage and mean width per nominal level.
    for nl in NOMINAL_LEVELS:
        tag = int(round(nl * 100))
        row[f"cqr_coverage_{tag}"] = float(np.mean(cqr_coverage[nl]))
        row[f"cqr_width_{tag}"] = float(np.mean(cqr_widths[nl]))

    # Pull MZ-corrected metrics from Phase 3's output if it ran for this file.
    if mz_root is not None:
        mz_path = mz_root / f.model / f.path.name
        if mz_path.exists():
            try:
                mz_df = read_density_csv(mz_path)
                mz_actuals, mz_grid = split_actual_and_grid(mz_df, levels)
                mz_metrics = score_density_grid(mz_actuals, mz_grid, levels)
                mz_metrics.pop("log_pit_bin_counts", None)
                mz_metrics.pop("lvl_pit_bin_counts", None)
                row.update({f"mz_{k}": v for k, v in mz_metrics.items()})
            except Exception:
                pass

    # Convenience deltas vs raw for headline columns.
    for key in ("log_crps_mean", "log_pit_ks_stat", "log_coverage_80", "log_coverage_95"):
        if f"raw_{key}" in row and f"iso_{key}" in row:
            row[f"delta_iso_{key}"] = row[f"iso_{key}"] - row[f"raw_{key}"]

    return row


def cross_asset_summary(by_asset: pd.DataFrame) -> pd.DataFrame:
    if by_asset.empty:
        return by_asset
    numeric_cols = [c for c in by_asset.columns
                    if pd.api.types.is_numeric_dtype(by_asset[c])
                    and c not in {"horizon", "context_length"}]
    return (by_asset.groupby(["model", "horizon", "context_length"], dropna=False)[numeric_cols]
            .mean().reset_index()
            .sort_values(["model", "horizon", "context_length"])
            .reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: distributional recalibration")
    parser.add_argument("--root", default=str(DENSITY_DIR))
    parser.add_argument("--iso-root", default=str(ISO_DIR))
    parser.add_argument("--cqr-root", default=str(CQR_DIR))
    parser.add_argument("--mz-root", default=str(MZ_DIR),
                        help="Path to Phase-3 output; if present, MZ metrics are merged in.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    iso_root = Path(args.iso_root)
    cqr_root = Path(args.cqr_root)
    mz_root = Path(args.mz_root) if args.mz_root else None
    output = Path(args.output) if args.output else RECAL_DIR / "recalibration_comparison.xlsx"

    logger = setup_logger("density_recalibration")
    logger.info("=== Phase 4: Distributional recalibration ===")
    logger.info(f"Block schedule: train>={args.min_train}, val={args.val_size}, test={args.test_size}, step={args.step_size}")

    files = discover_density_files(
        root, models=args.models, tickers=args.tickers, horizons=args.horizons,
    )
    if not files:
        logger.warning("No density CSVs found.")
        return
    logger.info(f"Found {len(files)} density files.")

    blocks_cfg = dict(
        min_train=args.min_train,
        val_size=args.val_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            row = recalibrate_one_file(
                f, DEFAULT_QUANTILE_LEVELS, blocks_cfg, iso_root, cqr_root, mz_root,
            )
        except Exception as exc:
            logger.error(f"  [{i}/{len(files)}] FAILED {f.model}/{f.ticker} h={f.horizon}: {exc}")
            continue
        if row is None:
            logger.warning(f"  [{i}/{len(files)}] skipped (no blocks): {f.model}/{f.ticker} h={f.horizon}")
            continue
        rows.append(row)
        if i % 10 == 0 or i == len(files):
            logger.info(f"  processed {i}/{len(files)} ({time.time() - t0:.1f}s elapsed)")

    if not rows:
        logger.warning("No files recalibrated; nothing to write.")
        return

    by_asset = pd.DataFrame(rows)
    summary = cross_asset_summary(by_asset)

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_asset.sort_values(["model", "ticker", "horizon"]).to_excel(
            writer, sheet_name="by_asset", index=False
        )
    logger.info(f"Wrote workbook: {output}")

    logger.info("\n=== Cross-asset headline (log space) ===")
    have_mz = any(c.startswith("mz_") for c in summary.columns)
    for _, row in summary.iterrows():
        crps_raw = row.get("raw_log_crps_mean", float("nan"))
        crps_iso = row.get("iso_log_crps_mean", float("nan"))
        cov_raw = "/".join(f"{row.get(f'raw_log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        cov_iso = "/".join(f"{row.get(f'iso_log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        cov_cqr = "/".join(f"{row.get(f'cqr_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        msg = (
            f"  {row['model']:>22s} h={int(row['horizon']):>2d}  "
            f"CRPS raw={crps_raw:.4f} iso={crps_iso:.4f}  "
            f"cov 50/80/95 raw={cov_raw} iso={cov_iso} cqr={cov_cqr}"
        )
        if have_mz:
            crps_mz = row.get("mz_log_crps_mean", float("nan"))
            cov_mz = "/".join(f"{row.get(f'mz_log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
            msg += f"  || mz: CRPS={crps_mz:.4f} cov={cov_mz}"
        logger.info(msg)
    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
