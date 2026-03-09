"""
run_cov_metrics.py — Compute covariance forecast accuracy metrics from merged .npz files.

Reads merged forecast .npz files (dates, forecasts, actuals, assets) and computes
Frobenius norm and diagonal QLIKE metrics. Saves cov_metrics_*.csv files.

Usage:
    python run_cov_metrics.py --asset-class stocks
    python run_cov_metrics.py --asset-class forex --models chronos_bolt_small moirai_2_0_small
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import COV_RESULTS_DIR, forecast_cfg
from utils import setup_logger


def compute_cov_metrics(forecasts, actuals, n):
    """Compute Frobenius norm and diagonal QLIKE from forecast/actual arrays."""
    frob_losses = []
    diag_qlikes = []

    for f_mat, a_mat in zip(forecasts, actuals):
        frob = np.sqrt(np.sum((f_mat - a_mat) ** 2)) / n
        frob_losses.append(frob)

        for k in range(n):
            actual_v = a_mat[k, k]
            forecast_v = f_mat[k, k]
            if actual_v > 0 and forecast_v > 0:
                ratio = actual_v / forecast_v
                diag_qlikes.append(ratio - np.log(ratio) - 1)

    return {
        'avg_frobenius': np.mean(frob_losses),
        'avg_diag_qlike': np.mean(diag_qlikes) if diag_qlikes else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute cov forecast accuracy metrics")
    parser.add_argument('--asset-class', default='stocks',
                        choices=['stocks', 'forex', 'futures'])
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model names (e.g. chronos_bolt_small). Auto-detects if omitted.')
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons
    logger = setup_logger("cov_metrics")

    forecast_dir = COV_RESULTS_DIR / args.asset_class / "forecasts"
    metrics_dir = COV_RESULTS_DIR / args.asset_class / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect models from merged .npz files (exclude batch files with _pairs)
    if args.models:
        models = args.models
    else:
        npz_files = [f for f in forecast_dir.glob("*_h*.npz") if 'pairs' not in f.name]
        models = sorted(set(f.name.split('_h')[0] for f in npz_files))
        logger.info(f"Auto-detected models: {models}")

    logger.info(f"=== Covariance Forecast Metrics ===")
    logger.info(f"Asset class: {args.asset_class}, Models: {models}, Horizons: {horizons}")

    for model in models:
        for h in horizons:
            fpath = forecast_dir / f"{model}_h{h}.npz"
            out_path = metrics_dir / f"cov_metrics_{model}_h{h}.csv"

            if not fpath.exists():
                logger.warning(f"  {fpath.name} not found, skipping")
                continue

            data = np.load(fpath, allow_pickle=True)
            forecasts = data['forecasts']
            actuals = data['actuals']
            n = len(data['assets'])
            n_dates = len(data['dates'])

            metrics = compute_cov_metrics(forecasts, actuals, n)
            metrics['model'] = model
            metrics['horizon'] = h
            metrics['n_dates'] = n_dates

            logger.info(f"  {model} h={h}: Frobenius={metrics['avg_frobenius']:.6f}, "
                         f"Diag QLIKE={metrics['avg_diag_qlike']:.6f} ({n_dates} dates)")

            pd.DataFrame([metrics]).to_csv(out_path, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
