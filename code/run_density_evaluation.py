"""
run_density_evaluation.py — Aggregate density-evaluation harness.

Walks results/volare/density/ and scores every persisted density CSV
(both HAR baselines and TSFM forecasts) on the same Q3-Q5 diagnostic
set:
    log-space and level-space CRPS (mean + median)
    PIT KS, Anderson-Darling, Berkowitz LR, Ljung-Box, histogram shape
    coverage + left/right tail breach rates @ 50% / 80% / 95%
    mean interval widths @ 50% / 80% / 95%

Outputs an Excel workbook with three sheets:
    summary       cross-asset means by (model, horizon)
    by_asset      flat per-(model, ticker, horizon) row of all metrics
    pit_bins      per-row PIT bin counts for downstream plotting

Q1 caveat: for TimesFM 2.5 and Moirai 2.0-S, Q2.5 / Q5 / Q95 / Q97.5
columns are log-space-extrapolated from native deciles -- 95% coverage
and tail-95 breach rates should be read as diagnostics rather than as
the model's true tail behaviour. Other models query the tails natively
or from MC samples and do not carry this caveat.

Usage:
    python run_density_evaluation.py
    python run_density_evaluation.py --models chronos_bolt_base sundial
    python run_density_evaluation.py --output results/volare/density/eval_phase2.xlsx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR
from evaluation.density import DEFAULT_QUANTILE_LEVELS
from evaluation.density_io import (
    DensityFile,
    discover_density_files,
    read_density_csv,
    score_density_frame,
    split_actual_and_grid,
)
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"

# Chronos-Bolt (S/B), TimesFM 2.5, and Moirai 2.0-S all only emit native
# deciles {Q10..Q90}. Their Q2.5 / Q5 / Q95 / Q97.5 are log-space
# extrapolations from those deciles -- flag them so the tail-95 columns
# carry the Q1 caveat into the workbook.
DECILE_ONLY_MODELS = {
    "chronos_bolt_small",
    "chronos_bolt_base",
    "timesfm_2_5",
    "moirai_2_0_small",
}


def _starify_tail_columns_for_model(row: Dict[str, object], model: str) -> Dict[str, object]:
    """Suffix tail-95 columns with '*' for decile-only models so the Q1 caveat
    travels with the data into the Excel output."""
    if model not in DECILE_ONLY_MODELS:
        return row
    flagged: Dict[str, object] = {}
    for key, val in row.items():
        if any(tag in key for tag in ("coverage_95", "tail_left_95", "tail_right_95", "width_95")):
            flagged[f"{key}*"] = val
        else:
            flagged[key] = val
    return flagged


def evaluate_files(
    files: Sequence[DensityFile],
    levels: np.ndarray,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every density file. Returns (by_asset_df, pit_bins_df)."""
    by_asset_rows: List[Dict[str, object]] = []
    pit_rows: List[Dict[str, object]] = []

    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            df = read_density_csv(f.path)
            metrics = score_density_frame(df, levels)
        except Exception as exc:
            logger.error(f"  [{i}/{len(files)}] FAILED {f.model}/{f.ticker} h={f.horizon}: {exc}")
            continue

        # Extract PIT histogram bins into a separate sheet for plotting.
        for space in ("log", "lvl"):
            bin_field = metrics.pop(f"{space}_pit_bin_counts", "")
            if bin_field:
                pit_rows.append({
                    "model": f.model,
                    "ticker": f.ticker,
                    "horizon": f.horizon,
                    "context_length": f.context_length or 512,
                    "space": space,
                    "bin_counts": bin_field,
                })

        row: Dict[str, object] = {
            "model": f.model,
            "ticker": f.ticker,
            "horizon": f.horizon,
            "context_length": f.context_length or 512,
            **metrics,
        }
        row = _starify_tail_columns_for_model(row, f.model)
        by_asset_rows.append(row)

        if i % 25 == 0 or i == len(files):
            logger.info(f"  scored {i}/{len(files)} ({time.time() - t0:.1f}s elapsed)")

    by_asset = pd.DataFrame(by_asset_rows)
    pit_bins = pd.DataFrame(pit_rows)
    return by_asset, pit_bins


def cross_asset_summary(by_asset: pd.DataFrame) -> pd.DataFrame:
    """Mean across assets per (model, horizon, context_length) on the headline metrics."""
    if by_asset.empty:
        return by_asset

    # Collect numeric columns (skip the shape labels and bin-count strings).
    headline = [c for c in by_asset.columns
                if any(c.startswith(prefix) for prefix in (
                    "n_obs", "log_crps_", "lvl_crps_",
                    "log_pit_ks_", "log_pit_berkowitz_",
                    "log_pit_ad_", "log_pit_lb_", "log_pit_u_stat", "log_pit_skewness",
                    "log_coverage_", "log_tail_left_", "log_tail_right_", "log_width_",
                    "lvl_coverage_", "lvl_tail_left_", "lvl_tail_right_", "lvl_width_",
                ))
                or c.endswith("*")]
    headline = [c for c in headline if pd.api.types.is_numeric_dtype(by_asset[c])]
    # Add rejection-rate columns @5%.
    rejection_cols: Dict[str, List[bool]] = {}
    for test in ("ks", "ad", "berkowitz", "lb"):
        p_col = f"log_pit_{test}_pvalue"
        if p_col in by_asset.columns:
            rejection_cols[f"reject_{test}_pct"] = list((by_asset[p_col] < 0.05))

    grouped = by_asset.groupby(["model", "horizon", "context_length"], dropna=False)
    means = grouped[headline].mean().reset_index()
    if rejection_cols:
        for col, mask in rejection_cols.items():
            tmp = by_asset.assign(**{col: mask}).groupby(
                ["model", "horizon", "context_length"], dropna=False
            )[col].mean().reset_index()
            tmp[col] = (tmp[col] * 100).round(1)
            means = means.merge(tmp, on=["model", "horizon", "context_length"])

    # Modal PIT shape per group for a quick at-a-glance read.
    if "log_pit_shape" in by_asset.columns:
        modal = by_asset.groupby(
            ["model", "horizon", "context_length"], dropna=False
        )["log_pit_shape"].agg(lambda s: s.mode().iloc[0] if len(s) else "")
        means = means.merge(modal.reset_index(), on=["model", "horizon", "context_length"])

    return means.sort_values(["model", "horizon", "context_length"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every persisted density forecast")
    parser.add_argument("--root", default=str(DENSITY_DIR), help="Density results root")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--output", default=None, help="Excel output path")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output) if args.output else root / "density_evaluation.xlsx"

    logger = setup_logger("density_evaluation")
    logger.info("=== Density evaluation harness ===")
    logger.info(f"Root: {root}")
    logger.info(f"Filters: models={args.models}, tickers={args.tickers}, horizons={args.horizons}")

    files = discover_density_files(
        root,
        models=args.models,
        tickers=args.tickers,
        horizons=args.horizons,
    )
    if not files:
        logger.warning("No density CSVs found. Did you run the producer?")
        return

    by_model = pd.Series([f.model for f in files]).value_counts()
    logger.info(f"Discovered {len(files)} density files across {len(by_model)} models:")
    for m, n in by_model.items():
        logger.info(f"  {m:>22s}  {int(n):>4d} files")

    by_asset, pit_bins = evaluate_files(files, DEFAULT_QUANTILE_LEVELS, logger)
    if by_asset.empty:
        logger.warning("No files scored successfully; nothing to write.")
        return

    summary = cross_asset_summary(by_asset)

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_asset.sort_values(["model", "ticker", "horizon"]).to_excel(
            writer, sheet_name="by_asset", index=False
        )
        if not pit_bins.empty:
            pit_bins.to_excel(writer, sheet_name="pit_bins", index=False)

    logger.info(f"Wrote workbook: {output}")
    logger.info("\n=== Cross-asset headline (log-space, h breakout) ===")
    for _, row in summary.iterrows():
        cov = "/".join(f"{row.get(f'log_coverage_{n}', float('nan')):.2f}" for n in (50, 80, 95))
        shape = row.get("log_pit_shape", "?")
        rejects = "/".join(
            f"{row.get(f'reject_{t}_pct', float('nan')):.0f}"
            for t in ("ks", "ad", "berkowitz", "lb")
        )
        logger.info(
            f"  {row['model']:>22s} ctx={int(row['context_length']):>3d} h={int(row['horizon']):>2d}  "
            f"CRPS={row.get('log_crps_mean', float('nan')):.4f}  cov50/80/95={cov}  "
            f"shape={shape:<12s}  reject@5% KS/AD/Berk/LB={rejects}"
        )
    logger.info("Density evaluation complete.")


if __name__ == "__main__":
    main()
