"""
run_cov_foundation.py — Walk-forward covariance forecasting with TSFMs.

Strategy: element-wise zero-shot forecasting of each covariance pair
using Chronos-Bolt and Moirai 2.0, then reconstruct and PSD-project.

Usage:
    python run_cov_foundation.py --asset-class forex --models chronos-bolt-small
    python run_cov_foundation.py --asset-class stocks --pair-start 0 --pair-end 100
"""

import sys
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import COV_RESULTS_DIR, forecast_cfg, fm_cfg
from data_loader import load_covariance_data
from covariance_utils import get_pair_list, ensure_psd
from models.foundation import get_foundation_model
from utils import setup_logger

AVAILABLE_MODELS = ['chronos-bolt-small', 'chronos-bolt-base', 'moirai-2.0-small', 'toto', 'sundial', 'moirai-moe-small', 'timesfm-2.5']


def run_covariance_tsfm(
    asset_class: str,
    model_name: str,
    horizons: list,
    context_length: int,
    device: str,
    pair_start: int,
    pair_end: int,
    logger,
    skip_existing: bool = False,
):
    """Run element-wise zero-shot TSFM forecasting for covariance matrices."""
    logger.info(f"Loading covariance data: {asset_class}")
    cov_data = load_covariance_data(asset_class=asset_class)
    assets = cov_data.assets
    n = len(assets)
    all_pairs = get_pair_list(assets)
    dates = cov_data.dates
    n_dates = len(dates)

    # Subset pairs for SLURM array parallelism
    pair_end = min(pair_end, len(all_pairs))
    pairs = all_pairs[pair_start:pair_end]
    logger.info(f"Assets: {n}, Total pairs: {len(all_pairs)}, "
                f"Processing pairs {pair_start}-{pair_end} ({len(pairs)} pairs)")

    # Load TSFM model
    logger.info(f"Loading model: {model_name}")
    t_load = time.time()
    model = get_foundation_model(model_name, device=device)
    model.load_model()
    logger.info(f"Model loaded in {time.time() - t_load:.1f}s")

    out_dir = COV_RESULTS_DIR / asset_class / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace('-', '_').replace('.', '_')

    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon h={horizon}")

        # Skip if full matrix NPZ already exists
        if skip_existing:
            full_npz = out_dir / f"{safe_model}_h{horizon}.npz"
            if full_npz.exists():
                logger.info(f"  Full matrix NPZ exists for {model_name} h={horizon}, skipping.")
                continue

        # For each pair, run zero-shot forecasting over the full series
        pair_forecasts = {}  # (a1, a2) -> {date: value}

        for p_idx, pair in enumerate(pairs):
            a1, a2 = pair
            series = cov_data.pair_series.get((a1, a2))
            if series is None:
                logger.warning(f"  No series for pair ({a1}, {a2}), skipping")
                continue

            series = series.dropna().sort_index()
            if len(series) < context_length + 10:
                logger.warning(f"  Pair ({a1}, {a2}): only {len(series)} obs, skipping")
                continue

            if p_idx % 50 == 0:
                logger.info(f"  Pair {p_idx+1}/{len(pairs)}: ({a1}, {a2})")

            values = series.values
            pair_dates = series.index
            pair_forecasts[pair] = {}

            for i in range(context_length, len(values)):
                ctx = values[i - context_length:i]
                result = model.predict(ctx, horizon)
                point = result.point

                if horizon == 1:
                    pred_val = float(point[0])
                else:
                    pred_val = float(np.mean(point[:horizon]))

                pair_forecasts[pair][pair_dates[i]] = pred_val

        # Save per-pair forecasts
        fname = f"{safe_model}_h{horizon}_pairs{pair_start}_{pair_end}.npz"
        np.savez_compressed(
            out_dir / fname,
            pairs=np.array([(a1, a2) for a1, a2 in pair_forecasts.keys()]),
            dates={str(pair): list(forecasts.keys())
                   for pair, forecasts in pair_forecasts.items()},
            values={str(pair): list(forecasts.values())
                    for pair, forecasts in pair_forecasts.items()},
        )

        # If we have all pairs, reconstruct full matrices
        if pair_start == 0 and pair_end >= len(all_pairs):
            logger.info("  Reconstructing full covariance matrices...")
            reconstruct_and_save_matrices(
                pair_forecasts, cov_data, assets, all_pairs,
                horizon, safe_model, out_dir, logger
            )

        logger.info(f"  Horizon h={horizon} done. {len(pair_forecasts)} pairs forecasted.")


def reconstruct_and_save_matrices(
    pair_forecasts, cov_data, assets, all_pairs,
    horizon, safe_model, out_dir, logger
):
    """Reconstruct full covariance matrices from element-wise forecasts."""
    n = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}
    dates = cov_data.dates
    n_dates = len(dates)

    all_forecast_mats = {}
    all_actual_mats = {}

    # Get forecast dates (dates where all pairs have forecasts)
    first_pair = list(pair_forecasts.keys())[0]
    forecast_dates = sorted(pair_forecasts[first_pair].keys())

    for date in forecast_dates:
        mat = np.zeros((n, n))
        complete = True
        for pair in all_pairs:
            if pair not in pair_forecasts or date not in pair_forecasts[pair]:
                complete = False
                break
            val = pair_forecasts[pair][date]
            i, j = asset_idx[pair[0]], asset_idx[pair[1]]
            mat[i, j] = val
            mat[j, i] = val

        if not complete:
            continue

        mat = ensure_psd(mat)
        all_forecast_mats[date] = mat

        # Get actual matrix
        actual_date_idx = dates.index(date) if date in dates else None
        if actual_date_idx is not None:
            if horizon == 1:
                actual_mat = cov_data.matrices.get(date)
            else:
                end_idx = min(actual_date_idx + horizon, n_dates)
                mats = []
                for k in range(actual_date_idx, end_idx):
                    m = cov_data.matrices.get(dates[k])
                    if m is not None:
                        mats.append(m)
                actual_mat = np.mean(mats, axis=0) if mats else None

            if actual_mat is not None:
                all_actual_mats[date] = actual_mat

    # Save complete matrices
    common_dates = sorted(set(all_forecast_mats.keys()) & set(all_actual_mats.keys()))
    if common_dates:
        np.savez_compressed(
            out_dir / f"{safe_model}_h{horizon}.npz",
            dates=np.array(common_dates),
            forecasts=np.array([all_forecast_mats[d] for d in common_dates]),
            actuals=np.array([all_actual_mats[d] for d in common_dates]),
            assets=np.array(assets),
        )
        logger.info(f"  Saved {len(common_dates)} complete matrices")


def main():
    parser = argparse.ArgumentParser(description="Run TSFM covariance forecasting")
    parser.add_argument('--asset-class', default='forex',
                        choices=['stocks', 'forex', 'futures'])
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'TSFM models. Options: {AVAILABLE_MODELS}')
    parser.add_argument('--device', default=fm_cfg.device)
    parser.add_argument('--context-length', type=int,
                        default=forecast_cfg.tsfm_context_length)
    parser.add_argument('--pair-start', type=int, default=0,
                        help='Start index for pair range (SLURM parallelism)')
    parser.add_argument('--pair-end', type=int, default=99999,
                        help='End index for pair range (exclusive)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip model/horizon combos where full NPZ exists')
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or ['chronos-bolt-small']

    logger = setup_logger("cov_foundation")
    logger.info("=== Covariance Forecasting — TSFMs (Element-wise Zero-Shot) ===")
    logger.info(f"Asset class: {args.asset_class}, Models: {model_names}")
    logger.info(f"Horizons: {horizons}, Device: {args.device}")
    logger.info(f"Pair range: [{args.pair_start}, {args.pair_end})")

    for model_name in model_names:
        run_covariance_tsfm(
            asset_class=args.asset_class,
            model_name=model_name,
            horizons=horizons,
            context_length=args.context_length,
            device=args.device,
            pair_start=args.pair_start,
            pair_end=args.pair_end,
            logger=logger,
            skip_existing=args.skip_existing,
        )

    logger.info("Covariance TSFM forecasting complete.")


if __name__ == "__main__":
    main()
