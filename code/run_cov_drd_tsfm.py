"""
run_cov_drd_tsfm.py — Walk-forward covariance forecasting with DRD-TSFM hybrid.

Applies the DRD decomposition with TSFM-based forecasting of volatilities
and correlations, then reconstructs and PSD-projects the covariance matrix.

Usage:
    python run_cov_drd_tsfm.py --asset-class forex --models chronos-bolt-small
    python run_cov_drd_tsfm.py --asset-class stocks --models moirai-2.0-small
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
from covariance_utils import get_pair_list
from models.foundation import get_foundation_model
from models.drd_tsfm import DRDTSFMModel
from utils import setup_logger

AVAILABLE_MODELS = ['chronos-bolt-small', 'chronos-bolt-base', 'moirai-2.0-small', 'toto', 'sundial', 'moirai-moe-small', 'timesfm-2.5']


def run_drd_tsfm(
    asset_class: str,
    model_name: str,
    horizons: list,
    context_length: int,
    device: str,
    logger,
):
    """Run DRD-TSFM walk-forward covariance forecasting."""
    logger.info(f"Loading covariance data: {asset_class}")
    cov_data = load_covariance_data(asset_class=asset_class)
    assets = cov_data.assets
    n = len(assets)
    dates = cov_data.dates
    n_dates = len(dates)

    logger.info(f"Assets ({n}): {assets}")
    logger.info(f"Dates: {n_dates} ({dates[0].date()} to {dates[-1].date()})")

    # Load TSFM model
    logger.info(f"Loading model: {model_name}")
    t_load = time.time()
    tsfm = get_foundation_model(model_name, device=device)
    tsfm.load_model()
    logger.info(f"Model loaded in {time.time() - t_load:.1f}s")

    # Create DRD-TSFM model
    drd_model = DRDTSFMModel(tsfm=tsfm, context_length=context_length)

    out_dir = COV_RESULTS_DIR / asset_class / "forecasts"
    metrics_dir = COV_RESULTS_DIR / asset_class / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace('-', '_').replace('.', '_')
    drd_label = f"drd_{safe_model}"

    for horizon in horizons:
        label = f"DRD-{model_name} | {asset_class} | h={horizon}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {label}")
        t0 = time.time()

        all_forecast_mats = {}
        all_actual_mats = {}

        # Walk-forward: start after context_length days
        start_idx = context_length
        for t_idx in range(start_idx, n_dates):
            forecast_date = dates[t_idx]

            # Actual: h-step ahead realized covariance
            actual_end = t_idx + horizon
            if actual_end > n_dates:
                break

            if t_idx % 500 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Step {t_idx - start_idx}/{n_dates - start_idx} "
                            f"({forecast_date.date()}) [{elapsed:.0f}s]")

            # Forecast
            cov_hat = drd_model.predict_matrix(
                cov_data.pair_series, assets, forecast_date, horizon
            )

            # Actual covariance
            if horizon == 1:
                cov_actual = cov_data.matrices.get(dates[t_idx])
            else:
                mats = []
                for k in range(t_idx, min(t_idx + horizon, n_dates)):
                    m = cov_data.matrices.get(dates[k])
                    if m is not None:
                        mats.append(m)
                cov_actual = np.mean(mats, axis=0) if mats else None

            if cov_actual is not None:
                all_forecast_mats[forecast_date] = cov_hat
                all_actual_mats[forecast_date] = cov_actual

        elapsed = time.time() - t0
        logger.info(f"  Completed {label}: {len(all_forecast_mats)} forecasts in {elapsed:.1f}s")

        # Save forecasts
        common_dates = sorted(
            set(all_forecast_mats.keys()) & set(all_actual_mats.keys())
        )
        if common_dates:
            np.savez_compressed(
                out_dir / f"{drd_label}_h{horizon}.npz",
                dates=np.array(common_dates),
                forecasts=np.array([all_forecast_mats[d] for d in common_dates]),
                actuals=np.array([all_actual_mats[d] for d in common_dates]),
                assets=np.array(assets),
            )
            logger.info(f"  Saved {len(common_dates)} complete matrices")

        # Compute metrics
        if all_forecast_mats:
            compute_and_save_cov_metrics(
                all_forecast_mats, all_actual_mats, assets,
                drd_label, horizon, metrics_dir, logger
            )


def compute_and_save_cov_metrics(
    forecast_mats, actual_mats, assets, model_name, horizon, metrics_dir, logger
):
    """Compute covariance forecast accuracy metrics."""
    dates = sorted(forecast_mats.keys())
    n = len(assets)

    frob_losses = []
    diag_qlikes = []

    for date in dates:
        f_mat = forecast_mats[date]
        a_mat = actual_mats[date]

        # Frobenius norm
        frob = np.sqrt(np.sum((f_mat - a_mat) ** 2)) / n
        frob_losses.append(frob)

        # QLIKE on diagonal elements
        for k in range(n):
            actual_v = a_mat[k, k]
            forecast_v = f_mat[k, k]
            if actual_v > 0 and forecast_v > 0:
                ratio = actual_v / forecast_v
                diag_qlikes.append(ratio - np.log(ratio) - 1)

    metrics = {
        'model': model_name,
        'horizon': horizon,
        'n_dates': len(dates),
        'avg_frobenius': np.mean(frob_losses),
        'avg_diag_qlike': np.mean(diag_qlikes) if diag_qlikes else np.nan,
    }

    logger.info(f"  Frobenius: {metrics['avg_frobenius']:.6f}, "
                f"Diag QLIKE: {metrics['avg_diag_qlike']:.6f}")

    df = pd.DataFrame([metrics])
    fname = f"cov_metrics_{model_name}_h{horizon}.csv"
    df.to_csv(metrics_dir / fname, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run DRD-TSFM covariance forecasting")
    parser.add_argument('--asset-class', default='forex',
                        choices=['stocks', 'forex', 'futures'])
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'TSFM models. Options: {AVAILABLE_MODELS}')
    parser.add_argument('--device', default=fm_cfg.device)
    parser.add_argument('--context-length', type=int,
                        default=forecast_cfg.tsfm_context_length)
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or ['chronos-bolt-small']

    logger = setup_logger("cov_drd_tsfm")
    logger.info("=== Covariance Forecasting — DRD-TSFM Hybrid ===")
    logger.info(f"Asset class: {args.asset_class}, Models: {model_names}")
    logger.info(f"Horizons: {horizons}, Device: {args.device}")
    logger.info(f"Context length: {args.context_length}")

    for model_name in model_names:
        run_drd_tsfm(
            asset_class=args.asset_class,
            model_name=model_name,
            horizons=horizons,
            context_length=args.context_length,
            device=args.device,
            logger=logger,
        )

    logger.info("DRD-TSFM covariance forecasting complete.")


if __name__ == "__main__":
    main()
