"""
run_cov_baselines.py — Walk-forward covariance forecasting with econometric baselines.

Models:
    - Element-wise HAR: independent HAR on each covariance element
    - HAR-DRD: Bollerslev, Patton & Quaedvlieg (2018) decomposition

Usage:
    python run_cov_baselines.py --asset-class stocks --horizons 1 5 22
    python run_cov_baselines.py --asset-class forex --models har-drd
"""

import sys
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import COV_RESULTS_DIR, forecast_cfg
from data_loader import load_covariance_data
from covariance_utils import get_pair_list, ensure_psd
from models.har_cov import ElementwiseHARCov
from models.har_drd import HARDRDModel
from forecasting.rolling_forecast import generate_walk_forward_folds
from utils import setup_logger

AVAILABLE_MODELS = ['element-har', 'har-drd']


def run_covariance_baselines(
    asset_class: str,
    model_names: list,
    horizons: list,
    train_window: int,
    test_window: int,
    step_size: int,
    logger,
):
    """Run walk-forward covariance forecasting."""
    logger.info(f"Loading covariance data: {asset_class}")
    cov_data = load_covariance_data(asset_class=asset_class)
    logger.info(f"Loaded: {len(cov_data.assets)} assets, {len(cov_data.dates)} dates")

    assets = cov_data.assets
    n = len(assets)
    pairs = get_pair_list(assets)
    dates = cov_data.dates
    n_dates = len(dates)

    logger.info(f"Assets: {assets}")
    logger.info(f"Pairs: {len(pairs)}, Dates: {n_dates}")

    # Generate walk-forward folds
    folds = generate_walk_forward_folds(n_dates, train_window, test_window, step_size)
    logger.info(f"Walk-forward folds: {len(folds)}")

    out_dir = COV_RESULTS_DIR / asset_class
    forecast_dir = out_dir / "forecasts"
    metrics_dir = out_dir / "metrics"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        for horizon in horizons:
            label = f"{model_name} | {asset_class} | h={horizon}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {label}")
            t0 = time.time()

            all_forecast_mats = {}
            all_actual_mats = {}

            for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
                train_dates = pd.DatetimeIndex([dates[i] for i in range(tr_s, tr_e)])
                test_dates_idx = range(te_s, te_e)

                logger.info(f"  Fold {fold_idx+1}/{len(folds)}: "
                            f"train {dates[tr_s].date()}-{dates[tr_e-1].date()}, "
                            f"test {dates[te_s].date()}-{dates[te_e-1].date()}")

                # Fit model
                if model_name == 'element-har':
                    model = ElementwiseHARCov()
                    model.fit(cov_data.pair_series, pairs, train_dates)
                elif model_name == 'har-drd':
                    model = HARDRDModel()
                    model.fit(cov_data.pair_series, assets, train_dates)

                # Forecast each test date
                for t_idx in test_dates_idx:
                    forecast_date = dates[t_idx]
                    # Actual: h-step ahead realized covariance
                    actual_idx = t_idx + horizon - 1
                    if actual_idx >= n_dates:
                        break

                    if forecast_date in all_forecast_mats:
                        continue  # skip duplicates

                    if model_name == 'element-har':
                        cov_hat = model.predict_matrix(
                            pairs, cov_data.pair_series, assets, forecast_date
                        )
                    elif model_name == 'har-drd':
                        cov_hat = model.predict_matrix(
                            cov_data.pair_series, assets, forecast_date
                        )

                    if horizon == 1:
                        actual_date = dates[actual_idx]
                        cov_actual = cov_data.matrices.get(actual_date)
                    else:
                        # Average realized covariance over next h days
                        end_idx = min(t_idx + horizon, n_dates)
                        mats = []
                        for k in range(t_idx, end_idx):
                            m = cov_data.matrices.get(dates[k])
                            if m is not None:
                                mats.append(m)
                        cov_actual = np.mean(mats, axis=0) if mats else None

                    if cov_actual is not None:
                        all_forecast_mats[forecast_date] = cov_hat
                        all_actual_mats[forecast_date] = cov_actual

            elapsed = time.time() - t0
            logger.info(f"  Completed {label}: {len(all_forecast_mats)} forecasts in {elapsed:.1f}s")

            # Save forecasts as compressed numpy
            safe_name = model_name.replace('-', '_')
            np.savez_compressed(
                forecast_dir / f"{safe_name}_h{horizon}.npz",
                dates=np.array(list(all_forecast_mats.keys())),
                forecasts=np.array(list(all_forecast_mats.values())),
                actuals=np.array(list(all_actual_mats.values())),
                assets=np.array(assets),
            )

            # Compute aggregate metrics
            if all_forecast_mats:
                compute_and_save_cov_metrics(
                    all_forecast_mats, all_actual_mats, assets,
                    model_name, horizon, metrics_dir, logger
                )


def compute_and_save_cov_metrics(
    forecast_mats, actual_mats, assets, model_name, horizon, metrics_dir, logger
):
    """Compute per-element and aggregate covariance forecast metrics."""
    dates = sorted(forecast_mats.keys())
    n = len(assets)

    # Frobenius norm loss
    frob_losses = []
    # Element-wise QLIKE (on diagonal = variance elements)
    diag_qlikes = []

    for date in dates:
        f_mat = forecast_mats[date]
        a_mat = actual_mats[date]

        # Frobenius
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
    fname = f"cov_metrics_{model_name.replace('-', '_')}_h{horizon}.csv"
    df.to_csv(metrics_dir / fname, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run covariance forecasting baselines")
    parser.add_argument('--asset-class', default='forex',
                        choices=['stocks', 'forex', 'futures'],
                        help='Asset class (default: forex for quick testing)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Forecast horizons (default: 1 5 22)')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Models to run. Options: {AVAILABLE_MODELS}')
    parser.add_argument('--train-window', type=int, default=None)
    parser.add_argument('--test-window', type=int, default=None)
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or AVAILABLE_MODELS
    train_window = args.train_window or forecast_cfg.train_window
    test_window = args.test_window or forecast_cfg.test_window
    step_size = forecast_cfg.step_size

    logger = setup_logger("cov_baselines")
    logger.info("=== Covariance Forecasting — Baselines ===")
    logger.info(f"Asset class: {args.asset_class}")
    logger.info(f"Models: {model_names}, Horizons: {horizons}")
    logger.info(f"Walk-forward: train={train_window}, test={test_window}, step={step_size}")

    run_covariance_baselines(
        asset_class=args.asset_class,
        model_names=model_names,
        horizons=horizons,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        logger=logger,
    )

    logger.info("Covariance baseline forecasting complete.")


if __name__ == "__main__":
    main()
