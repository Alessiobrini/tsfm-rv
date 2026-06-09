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
    REPRESENTATIVE_TICKERS, forecast_cfg, har_cfg, data_cfg,
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
    walk_forward_forecast, walk_forward_series_forecast, iterated_har_forecast,
)
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger

# Import shared helpers + model-class routing from run_baselines (single source).
from run_baselines import (
    build_features_and_target, get_model_factory,
    AVAILABLE_MODELS, SERIES_MODELS, ITERATED_HAR_MODELS, DIRECT_HAR_MODELS,
)

FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"


def save_single_forecast(actual, forecast, model_name, ticker, horizon,
                         out_dir=None):
    """Save one model's forecasts to CSV in VOLARE results dir."""
    out_dir = out_dir or FORECAST_DIR
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
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip runs where output CSV already exists')
    parser.add_argument('--target-kind', default=None, choices=['point', 'avg'],
                        help=f'Forecast target (default: {forecast_cfg.target_kind}). '
                             '"point"=RV_{t+h} main; "avg"=h-day-average appendix '
                             '(routed to results/volare_avg/).')
    parser.add_argument('--scale', default=None, choices=['vol', 'var'],
                        help=f'Modeling scale (default: {data_cfg.target_scale}). '
                             '"vol"=forecast volatility (sqrt RV); "var"=variance.')
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
    target_kind = args.target_kind or forecast_cfg.target_kind
    target_scale = args.scale or data_cfg.target_scale

    # Results routing: the h-day-average appendix arm and non-default train
    # windows each get their own directory; the main run (point target,
    # default 1000-day window) writes to results/volare/.
    from config import RESULTS_DIR
    if target_kind == 'avg':
        custom_results_dir = RESULTS_DIR / "volare_avg"
        forecast_out_dir = custom_results_dir / "forecasts"
        metrics_out_dir = custom_results_dir / "metrics"
    elif train_window != forecast_cfg.train_window:
        custom_results_dir = RESULTS_DIR / f"volare_{train_window}"
        forecast_out_dir = custom_results_dir / "forecasts"
        metrics_out_dir = custom_results_dir / "metrics"
    else:
        custom_results_dir = VOLARE_RESULTS_DIR
        forecast_out_dir = FORECAST_DIR
        metrics_out_dir = VOLARE_RESULTS_DIR / "metrics"

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

                if args.skip_existing:
                    safe_name = model_name.replace('-', '_').replace(' ', '_')
                    out_path = forecast_out_dir / f"{safe_name}_{ticker}_h{horizon}.csv"
                    if out_path.exists():
                        logger.info(f"  Skipping {label}: output exists")
                        continue

                logger.info(f"Running {label}")
                t0 = time.time()

                try:
                    X_or_series, y = build_features_and_target(
                        data, ticker, horizon, model_name,
                        target_kind=target_kind, target_scale=target_scale,
                    )
                    factory = get_model_factory(model_name)

                    if model_name in SERIES_MODELS:
                        # ARFIMA / ARMA / MEM: native (iterated) multi-step.
                        actual, forecast = walk_forward_series_forecast(
                            series=X_or_series, model_factory=factory,
                            train_window=train_window, test_window=test_window,
                            step_size=step_size, horizon=horizon,
                            target_kind=target_kind,
                        )
                    elif model_name in ITERATED_HAR_MODELS:
                        # HAR / Log-HAR: iterated recursive plug-in (Referee 1).
                        actual, forecast = iterated_har_forecast(
                            rv_series=X_or_series, model_factory=factory,
                            horizon=horizon, train_window=train_window,
                            test_window=test_window, step_size=step_size,
                            target_kind=target_kind,
                        )
                    else:
                        # HAR-J / HAR-RS / HARQ: direct h-step (auxiliary
                        # regressors cannot be projected; literature-standard).
                        actual, forecast = walk_forward_forecast(
                            X=X_or_series, y=y, model_factory=factory,
                            train_window=train_window, test_window=test_window,
                            step_size=step_size,
                            reestimate_every=forecast_cfg.reestimate_every,
                        )

                    # Winsorize forecasts to the in-sample realized-vol support
                    # [min, max]: the floor replaces the "unacceptable" 1e-10
                    # (Referee 2 minor 4); the symmetric cap guards against
                    # pathological high spikes (e.g. heavy-tailed TSFM draws).
                    rv_clean = data.rv[ticker].dropna()
                    var_floor = float(rv_clean.min())
                    var_cap = float(rv_clean.max())

                    if target_scale == "vol":
                        if model_name in DIRECT_HAR_MODELS:
                            # Augmented variants are fit on variance -> map to
                            # volatility (floor variance before sqrt to avoid NaNs).
                            actual = np.sqrt(actual.clip(lower=0.0))
                            forecast = np.sqrt(forecast.clip(lower=var_floor))
                        store_floor = float(np.sqrt(var_floor))
                        store_cap = float(np.sqrt(var_cap))
                    else:
                        store_floor = var_floor
                        store_cap = var_cap

                    forecast = forecast.clip(lower=store_floor, upper=store_cap)

                    fpath = save_single_forecast(
                        actual, forecast, model_name, ticker, horizon,
                        out_dir=forecast_out_dir,
                    )

                    metrics = compute_all_losses(
                        actual, forecast, scale=target_scale, var_floor=var_floor
                    )
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
        metrics_out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(metrics_out_dir / "baseline_metrics.csv", index=False)

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
