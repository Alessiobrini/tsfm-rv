"""
run_foundation.py — Run zero-shot time series foundation model evaluation.

Orchestrates:
    1. Load data
    2. For each TSFM model (load once):
       For each asset x horizon: run zero-shot forecast over full sample
    3. Save each model's forecasts to individual CSV files
    4. Print summary statistics

Usage:
    python run_foundation.py [--tickers AAPL JPM] [--horizons 1 5 22] [--models chronos-bolt-small]
    python run_foundation.py --device cuda --models chronos-bolt-base timesfm-2.5
"""

import sys
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MAIN_TICKERS, REPRESENTATIVE_TICKERS, forecast_cfg, fm_cfg, RESULTS_DIR,
)
from data_loader import load_data
from models.foundation import get_foundation_model
from forecasting.rolling_forecast import zero_shot_forecast
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger


AVAILABLE_MODELS = ['chronos-bolt-small', 'chronos-bolt-base', 'timesfm-2.0', 'moirai-2.0-small', 'lag-llama', 'kronos']

FORECAST_DIR = RESULTS_DIR / "forecasts"


def save_single_forecast(actual, forecast, model_name, ticker, horizon):
    """Save one model's forecasts to CSV."""
    out_dir = FORECAST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'date': actual.index,
        'actual': actual.values,
        'forecast': forecast.values,
    })
    df.set_index('date', inplace=True)

    safe_name = model_name.replace('-', '_').replace('.', '_').replace(' ', '_')
    filepath = out_dir / f"{safe_name}_{ticker}_h{horizon}.csv"
    df.to_csv(filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run foundation model zero-shot evaluation")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers to run (default: representative subset)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Forecast horizons (default: 1 5 22)')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'TSFM model names. Options: {AVAILABLE_MODELS}')
    parser.add_argument('--device', default=fm_cfg.device,
                        help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--context-length', type=int,
                        default=forecast_cfg.tsfm_context_length,
                        help='Context window length')
    parser.add_argument('--all-tickers', action='store_true',
                        help='Run on all 29 DJIA tickers')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip runs where output CSV already exists')
    args = parser.parse_args()

    if args.all_tickers:
        tickers = MAIN_TICKERS
    else:
        tickers = args.tickers or REPRESENTATIVE_TICKERS
    horizons = args.horizons or forecast_cfg.horizons
    model_names = args.models or ['chronos-bolt-small']
    device = args.device
    context_length = args.context_length

    logger = setup_logger("foundation")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {device}, Context length: {context_length}")

    # Load data
    data = load_data(tickers=tickers)
    logger.info(f"Data loaded: {len(data.tickers)} assets, {len(data.dates)} dates")

    all_metrics = []

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model: {model_name}")
        t_load = time.time()

        try:
            model = get_foundation_model(model_name, device=device)
            model.load_model()
            logger.info(f"Model loaded in {time.time() - t_load:.1f}s")
        except ImportError as e:
            logger.error(f"Cannot load {model_name}: {e}")
            logger.error("Install the required package and retry.")
            continue
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        for ticker in tickers:
            for horizon in horizons:
                label = f"{model_name} | {ticker} | h={horizon}"

                if args.skip_existing:
                    safe_name = model_name.replace('-', '_').replace('.', '_').replace(' ', '_')
                    out_path = FORECAST_DIR / f"{safe_name}_{ticker}_h{horizon}.csv"
                    if out_path.exists():
                        logger.info(f"  Skipping {label}: output exists")
                        continue

                logger.info(f"  Running {label}")
                t0 = time.time()

                try:
                    rv = data.rv[ticker].dropna()

                    if len(rv) < context_length + 10:
                        logger.warning(
                            f"  Skipping {label}: only {len(rv)} obs "
                            f"(need {context_length}+ for context)"
                        )
                        continue

                    actual, forecast = zero_shot_forecast(
                        rv_series=rv,
                        model=model,
                        horizon=horizon,
                        context_length=context_length,
                    )

                    # Clip negative forecasts
                    # Floor of 0.01 prevents QLIKE blowup from near-zero predictions
                    forecast = forecast.clip(lower=0.01)

                    # Save
                    fpath = save_single_forecast(
                        actual, forecast, model_name, ticker, horizon
                    )

                    # Metrics
                    metrics = compute_all_losses(actual, forecast)
                    metrics['model'] = model_name
                    metrics['ticker'] = ticker
                    metrics['horizon'] = horizon
                    metrics['n_obs'] = len(actual)
                    all_metrics.append(metrics)

                    elapsed = time.time() - t0
                    logger.info(
                        f"    Done in {elapsed:.1f}s | n={len(actual)} | "
                        f"R2={metrics['R2OOS']:.3f} | QLIKE={metrics['QLIKE']:.4f}"
                    )

                except Exception as e:
                    logger.error(f"    FAILED {label}: {e}")
                    continue

    # Summary
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary_path = RESULTS_DIR / "metrics"
        summary_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path / "foundation_metrics.csv", index=False)

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

    logger.info("Foundation model forecasting complete.")


if __name__ == "__main__":
    main()
