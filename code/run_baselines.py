"""
run_baselines.py — Run all econometric baseline models with walk-forward evaluation.

Orchestrates:
    1. Load data
    2. For each asset x horizon x model: run walk-forward forecast
    3. Save each model's forecasts to individual CSV files
    4. Print summary statistics

Usage:
    python run_baselines.py [--tickers AAPL JPM] [--horizons 1 5 22] [--models HAR HAR-J]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MAIN_TICKERS, REPRESENTATIVE_TICKERS, forecast_cfg, har_cfg,
    arfima_cfg, RESULTS_DIR,
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


AVAILABLE_MODELS = ['HAR', 'HAR-J', 'HAR-RS', 'HARQ', 'Log-HAR', 'ARFIMA']

FORECAST_DIR = RESULTS_DIR / "forecasts"


def build_features_and_target(data, ticker, horizon, model_name):
    """Build features and aligned target for a given model/ticker/horizon.

    Returns (X, y) tuple or (series, None) for series-based models.
    """
    rv = data.rv[ticker].dropna()
    target = build_target(rv, horizon=horizon)

    if model_name == 'HAR':
        features = build_har_features(rv)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HAR-J':
        jump = data.jump[ticker].dropna()
        # Align jump to rv index
        common_idx = rv.index.intersection(jump.index)
        rv_a, jump_a = rv.loc[common_idx], jump.loc[common_idx]
        features = build_har_j_features(rv_a, jump_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HAR-RS':
        good = data.good[ticker].dropna()
        bad = data.bad[ticker].dropna()
        common_idx = rv.index.intersection(good.index).intersection(bad.index)
        good_a, bad_a = good.loc[common_idx], bad.loc[common_idx]
        rv_a = rv.loc[common_idx]
        features = build_har_rs_features(good_a, bad_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'HARQ':
        rq = data.rq[ticker].dropna()
        common_idx = rv.index.intersection(rq.index)
        rv_a, rq_a = rv.loc[common_idx], rq.loc[common_idx]
        features = build_harq_features(rv_a, rq_a)
        target = build_target(rv_a, horizon=horizon)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'Log-HAR':
        features = build_har_features(rv)
        X, y = align_features_target(features, target)
        return X, y

    elif model_name == 'ARFIMA':
        # Series-based: return (series, None)
        return rv, None

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_factory(model_name):
    """Return a callable that creates a fresh model instance."""
    if model_name == 'HAR':
        return lambda: HARModel()
    elif model_name == 'HAR-J':
        return lambda: HARJModel()
    elif model_name == 'HAR-RS':
        return lambda: HARRSModel()
    elif model_name == 'HARQ':
        return lambda: HARQModel()
    elif model_name == 'Log-HAR':
        return lambda: HARModel(use_log=True)
    elif model_name == 'ARFIMA':
        return lambda: ARFIMAModel(
            p=arfima_cfg.max_ar,
            q=arfima_cfg.max_ma,
            use_log=arfima_cfg.use_log_rv,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def save_single_forecast(actual, forecast, model_name, ticker, horizon):
    """Save one model's forecasts to CSV: results/forecasts/{model}_{ticker}_h{horizon}.csv"""
    out_dir = FORECAST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'date': actual.index,
        'actual': actual.values,
        'forecast': forecast.values,
    })
    df.set_index('date', inplace=True)

    # Sanitize model name for filename
    safe_name = model_name.replace('-', '_').replace(' ', '_')
    filepath = out_dir / f"{safe_name}_{ticker}_h{horizon}.csv"
    df.to_csv(filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run econometric baselines")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers to run (default: representative subset)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Forecast horizons (default: 1 5 22)')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Model names (default: all). Options: {AVAILABLE_MODELS}')
    parser.add_argument('--all-tickers', action='store_true',
                        help='Run on all 29 DJIA tickers')
    parser.add_argument('--train-window', type=int, default=None,
                        help=f'Training window (default: {forecast_cfg.train_window})')
    parser.add_argument('--test-window', type=int, default=None,
                        help=f'Test window (default: {forecast_cfg.test_window})')
    args = parser.parse_args()

    if args.all_tickers:
        tickers = MAIN_TICKERS
    else:
        tickers = args.tickers or REPRESENTATIVE_TICKERS
    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or AVAILABLE_MODELS
    train_window = args.train_window or forecast_cfg.train_window
    test_window = args.test_window or forecast_cfg.test_window
    step_size = forecast_cfg.step_size

    logger = setup_logger("baselines")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Walk-forward: train={train_window}, test={test_window}, step={step_size}")

    # Load data
    data = load_data(tickers=tickers)
    logger.info(f"Data loaded: {len(data.tickers)} assets, {len(data.dates)} dates")

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
                        # Series-based walk-forward
                        actual, forecast = walk_forward_series_forecast(
                            series=X_or_series,
                            model_factory=factory,
                            train_window=train_window,
                            test_window=test_window,
                            step_size=step_size,
                            horizon=horizon,
                        )
                    else:
                        # Feature-based walk-forward
                        actual, forecast = walk_forward_forecast(
                            X=X_or_series,
                            y=y,
                            model_factory=factory,
                            train_window=train_window,
                            test_window=test_window,
                            step_size=step_size,
                            reestimate_every=forecast_cfg.reestimate_every,
                        )

                    # Clip negative forecasts (RV must be positive)
                    # Floor of 0.01 prevents QLIKE blowup from near-zero predictions
                    forecast = forecast.clip(lower=0.01)

                    # Save forecasts
                    fpath = save_single_forecast(
                        actual, forecast, model_name, ticker, horizon
                    )

                    # Compute metrics
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
        summary_path = RESULTS_DIR / "metrics"
        summary_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path / "baseline_metrics.csv", index=False)

        # Print pivot table
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

    logger.info("Baseline forecasting complete.")


if __name__ == "__main__":
    main()
