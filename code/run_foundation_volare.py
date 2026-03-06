"""
run_foundation_volare.py — Run zero-shot TSFMs on VOLARE dataset.

Mirrors run_foundation.py but loads VOLARE data and saves to results/volare/forecasts/.
All model and evaluation code is reused from the shared modules.

Usage:
    python run_foundation_volare.py [--tickers AAPL JPM] [--horizons 1 5 22] [--models chronos-bolt-small]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REPRESENTATIVE_TICKERS, forecast_cfg, fm_cfg, VOLARE_RESULTS_DIR,
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)
from data_loader import load_data
from models.foundation import get_foundation_model
from forecasting.rolling_forecast import zero_shot_forecast
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger


AVAILABLE_MODELS = ['chronos-bolt-small', 'chronos-bolt-base', 'timesfm-2.0', 'moirai-2.0-small', 'lag-llama', 'kronos']

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

    safe_name = model_name.replace('-', '_').replace('.', '_').replace(' ', '_')
    filepath = out_dir / f"{safe_name}_{ticker}_h{horizon}.csv"
    df.to_csv(filepath)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run TSFM zero-shot on VOLARE data")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers to run (default: AAPL AMZN CAT JPM)')
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
                        help='Run on all VOLARE tickers for the asset class')
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
    model_names = args.models or ['chronos-bolt-small']
    device = args.device
    context_length = args.context_length

    logger = setup_logger("foundation_volare")
    logger.info("=== VOLARE Dataset — Foundation Model Zero-Shot ===")
    logger.info(f"Models: {model_names}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {device}, Context length: {context_length}")

    # Load VOLARE data
    dataset_key = {"stocks": "volare", "fx": "volare_fx", "futures": "volare_futures"}[args.asset_class]
    data = load_data(dataset=dataset_key, tickers=tickers)
    logger.info(f"Data loaded: {len(data.tickers)} assets, {len(data.dates)} dates")
    logger.info(f"Date range: {data.dates[0].date()} to {data.dates[-1].date()}")

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
            continue
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue

        for ticker in tickers:
            for horizon in horizons:
                label = f"{model_name} | {ticker} | h={horizon}"
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
                        f"    Done in {elapsed:.1f}s | n={len(actual)} | "
                        f"R2={metrics['R2OOS']:.3f} | QLIKE={metrics['QLIKE']:.4f}"
                    )

                except Exception as e:
                    logger.error(f"    FAILED {label}: {e}")
                    continue

    # Summary
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary_path = VOLARE_RESULTS_DIR / "metrics"
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

    logger.info("VOLARE foundation model forecasting complete.")


if __name__ == "__main__":
    main()
