"""
run_baselines_volare.py — Run econometric baselines on VOLARE dataset.

Mirrors run_baselines.py but loads VOLARE data and saves to results/volare/forecasts/.
All model, feature, and evaluation code is reused from the shared modules.

Usage:
    python run_baselines_volare.py [--tickers AAPL JPM] [--horizons 1 5 22] [--models HAR HAR-J]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REPRESENTATIVE_TICKERS, forecast_cfg, har_cfg,
    arfima_cfg, VOLARE_RESULTS_DIR,
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)
from data_loader import load_data
from features import (
    build_har_features, build_har_j_features, build_har_rs_features,
    build_harq_features, build_target, align_features_target,
)
from models.har import HARModel, HARJModel, HARRSModel, HARQModel
from models.arfima import ARFIMAModel
from forecasting.rolling_forecast import (
    walk_forward_forecast, walk_forward_series_forecast,
)
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger

# Import shared helpers from run_baselines
from run_baselines import build_features_and_target, get_model_factory

AVAILABLE_MODELS = ['HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'Log-HAR', 'ARFIMA']

FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"


def save_single_forecast(actual, forecast, model_name, ticker, horizon):
    """Save one model's forecasts to CSV in VOLARE results dir."""
    out_dir = FORECAST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'date': actual.index,
        'actual': actual.values,
        'forecast': forecast.values,
    })
    df.set_index('date', inplace=True)

    safe_name = model_name.replace('-', '_').replace(' ', '_')
    filepath = out_dir / f"{safe_name}_{ticker}_h{horizon}.csv"
    df.to_csv(filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run econometric baselines on VOLARE data")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers to run (default: AAPL AMZN CAT JPM)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Forecast horizons (default: 1 5 22)')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Model names (default: all). Options: {AVAILABLE_MODELS}')
    parser.add_argument('--train-window', type=int, default=None,
                        help=f'Training window (default: {forecast_cfg.train_window})')
    parser.add_argument('--test-window', type=int, default=None,
                        help=f'Test window (default: {forecast_cfg.test_window})')
    parser.add_argument('--all-tickers', action='store_true',
                        help='Run on all 40 VOLARE stock tickers')
    parser.add_argument('--asset-class', default='stocks',
                        choices=['stocks', 'fx', 'futures'],
                        help='VOLARE asset class (default: stocks)')
    args = parser.parse_args()

    if args.all_tickers:
        if args.asset_class == 'fx':
            tickers = VOLARE_FX_TICKERS
        elif args.asset_class == 'futures':
            tickers = VOLARE_FUTURES_TICKERS
        else:
            tickers = VOLARE_STOCK_TICKERS
    else:
        tickers = args.tickers or REPRESENTATIVE_TICKERS
    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or AVAILABLE_MODELS
    train_window = args.train_window or forecast_cfg.train_window
    test_window = args.test_window or forecast_cfg.test_window
    step_size = forecast_cfg.step_size

    logger = setup_logger("baselines_volare")
    logger.info("=== VOLARE Dataset — Econometric Baselines ===")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Walk-forward: train={train_window}, test={test_window}, step={step_size}")

    # Load VOLARE data
    dataset_key = {"stocks": "volare", "fx": "volare_fx", "futures": "volare_futures"}[args.asset_class]
    data = load_data(dataset=dataset_key, tickers=tickers)
    logger.info(f"Data loaded: {len(data.tickers)} assets, {len(data.dates)} dates")
    logger.info(f"Date range: {data.dates[0].date()} to {data.dates[-1].date()}")

    all_metrics = []
    total_runs = len(tickers) * len(horizons) * len(model_names)
    completed = 0

    for ticker in tickers:
        for horizon in horizons:
            for model_name in model_names:
                completed += 1
                label = f"[{completed}/{total_runs}] {model_name} | {ticker} | h={horizon}"
                logger.info(f"Running {label}")
                t0 = time.time()

                try:
                    X_or_series, y = build_features_and_target(
                        data, ticker, horizon, model_name
                    )
                    factory = get_model_factory(model_name)

                    if model_name == 'ARFIMA':
                        actual, forecast = walk_forward_series_forecast(
                            series=X_or_series,
                            model_factory=factory,
                            train_window=train_window,
                            test_window=test_window,
                            step_size=step_size,
                            horizon=horizon,
                        )
                    else:
                        actual, forecast = walk_forward_forecast(
                            X=X_or_series,
                            y=y,
                            model_factory=factory,
                            train_window=train_window,
                            test_window=test_window,
                            step_size=step_size,
                            reestimate_every=forecast_cfg.reestimate_every,
                        )

                    # VOLARE RV is in decimal squared returns (~0.0002),
                    # min observed ~8e-6; floor at 1e-6 prevents QLIKE blowup
                    forecast = forecast.clip(lower=1e-6)

                    fpath = save_single_forecast(
                        actual, forecast, model_name, ticker, horizon
                    )

                    metrics = compute_all_losses(actual, forecast)
                    metrics['model'] = model_name
                    metrics['ticker'] = ticker
                    metrics['horizon'] = horizon
                    metrics['n_obs'] = len(actual)
                    all_metrics.append(metrics)

                    elapsed = time.time() - t0
                    logger.info(
                        f"  Done {label} in {elapsed:.1f}s | "
                        f"n={len(actual)} | R2={metrics['R2OOS']:.3f} | "
                        f"QLIKE={metrics['QLIKE']:.4f}"
                    )

                except Exception as e:
                    logger.error(f"  FAILED {label}: {e}")
                    continue

    # Summary table
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary_path = VOLARE_RESULTS_DIR / "metrics"
        summary_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path / "baseline_metrics.csv", index=False)

        logger.info("\n=== SUMMARY ===")
        for h in horizons:
            sub = summary[summary['horizon'] == h]
            if len(sub) == 0:
                continue
            pivot = sub.pivot_table(
                values=['QLIKE', 'R2OOS', 'MSE'],
                index='model',
                aggfunc='mean',
            ).round(4)
            logger.info(f"\nHorizon h={h} (avg across assets):")
            logger.info(f"\n{pivot.to_string()}")

    logger.info("VOLARE baseline forecasting complete.")


if __name__ == "__main__":
    main()
