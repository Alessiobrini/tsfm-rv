"""
merge_cov_batches.py — Merge batched TSFM covariance pair forecasts into full matrices.

When run_cov_foundation.py runs with --pair-start/--pair-end (SLURM parallelism),
it saves per-batch .npz files with element-wise forecasts. This script merges
all batch files for a given model/horizon into the full-matrix format expected
by run_portfolio_eval.py.

Usage:
    python merge_cov_batches.py --asset-class stocks
    python merge_cov_batches.py --asset-class stocks --models chronos_bolt_small moirai_2_0_small
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval

sys.path.insert(0, str(Path(__file__).parent))

from config import COV_RESULTS_DIR, forecast_cfg
from data_loader import load_covariance_data
from covariance_utils import ensure_psd
from utils import setup_logger


def merge_batches(asset_class, model_name, horizon, forecast_dir, cov_data, logger):
    """Merge batch .npz files into a single full-matrix .npz file."""
    pattern = f"{model_name}_h{horizon}_pairs*.npz"
    batch_files = sorted(forecast_dir.glob(pattern))

    if not batch_files:
        logger.warning(f"  No batch files for {model_name} h={horizon}")
        return False

    # Check if merged file already exists
    merged_path = forecast_dir / f"{model_name}_h{horizon}.npz"
    if merged_path.exists():
        logger.info(f"  {merged_path.name} already exists, skipping")
        return True

    logger.info(f"  Merging {len(batch_files)} batch files for {model_name} h={horizon}")

    assets = cov_data.assets
    n = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}

    # Collect all pair forecasts: (i, j) -> {date: value}
    pair_forecasts = {}
    total_pairs = 0

    for bf in batch_files:
        data = np.load(bf, allow_pickle=True)
        pairs_arr = data['pairs']
        dates_dict = data['dates'].item()
        values_dict = data['values'].item()

        for row in pairs_arr:
            a1, a2 = row[0], row[1]
            key = str((a1, a2))
            if key not in dates_dict:
                continue

            dates_list = dates_dict[key]
            vals_list = values_dict[key]
            i, j = asset_idx[a1], asset_idx[a2]
            pair_forecasts[(i, j)] = dict(zip(dates_list, vals_list))
            total_pairs += 1

    expected_pairs = n * (n + 1) // 2
    logger.info(f"    Loaded {total_pairs} pairs (expected {expected_pairs})")

    if total_pairs < expected_pairs:
        logger.warning(f"    Incomplete: {total_pairs}/{expected_pairs} pairs")
        return False

    # Get common dates across all pairs
    date_sets = [set(v.keys()) for v in pair_forecasts.values()]
    common_dates = sorted(set.intersection(*date_sets))
    logger.info(f"    Common forecast dates: {len(common_dates)}")

    if not common_dates:
        logger.error("    No common dates found!")
        return False

    # Build forecast matrices
    forecast_mats = []
    for date in common_dates:
        mat = np.zeros((n, n))
        for (i, j), forecasts in pair_forecasts.items():
            val = forecasts[date]
            mat[i, j] = val
            mat[j, i] = val
        mat = ensure_psd(mat)
        forecast_mats.append(mat)

    # Build actual matrices
    actual_mats = []
    valid_dates = []
    cov_dates = cov_data.dates

    for date in common_dates:
        if horizon == 1:
            actual_mat = cov_data.matrices.get(date)
        else:
            if date in cov_dates:
                date_idx = cov_dates.index(date)
                end_idx = min(date_idx + horizon, len(cov_dates))
                mats = []
                for k in range(date_idx, end_idx):
                    m = cov_data.matrices.get(cov_dates[k])
                    if m is not None:
                        mats.append(m)
                actual_mat = np.mean(mats, axis=0) if mats else None
            else:
                actual_mat = None

        if actual_mat is not None:
            actual_mats.append(actual_mat)
            valid_dates.append(date)

    logger.info(f"    Valid dates with actuals: {len(valid_dates)}")

    # Save merged file
    np.savez_compressed(
        merged_path,
        dates=np.array(valid_dates),
        forecasts=np.array([forecast_mats[common_dates.index(d)] for d in valid_dates]),
        actuals=np.array(actual_mats),
        assets=np.array(assets),
    )
    logger.info(f"    Saved {merged_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge batched TSFM cov forecasts")
    parser.add_argument('--asset-class', default='stocks',
                        choices=['stocks', 'forex', 'futures'])
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model names (underscore form, e.g. chronos_bolt_small)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons
    logger = setup_logger("merge_cov")
    forecast_dir = COV_RESULTS_DIR / args.asset_class / "forecasts"

    # Auto-detect models from batch files if not specified
    if args.models:
        models = args.models
    else:
        batch_files = list(forecast_dir.glob("*_pairs*.npz"))
        models = sorted(set(
            f.name.split('_h')[0] for f in batch_files
        ))
        logger.info(f"Auto-detected models: {models}")

    if not models:
        logger.info("No batched models found. Nothing to merge.")
        return

    logger.info(f"=== Merging Covariance Batches ===")
    logger.info(f"Asset class: {args.asset_class}, Models: {models}, Horizons: {horizons}")

    # Load covariance data (needed for actuals)
    logger.info("Loading covariance data...")
    cov_data = load_covariance_data(asset_class=args.asset_class)
    logger.info(f"Loaded {len(cov_data.assets)} assets, {len(cov_data.dates)} dates")

    for model in models:
        logger.info(f"\nModel: {model}")
        for h in horizons:
            merge_batches(args.asset_class, model, h, forecast_dir, cov_data, logger)

    logger.info("\nMerge complete.")


if __name__ == "__main__":
    main()
