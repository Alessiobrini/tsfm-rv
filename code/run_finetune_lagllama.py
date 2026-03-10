"""
run_finetune_lagllama.py — Fine-tune Lag-Llama on VOLARE realized volatility.

Walk-forward evaluation with periodic fine-tuning:
- Fine-tune every `retrain_every` steps (default: 126, matching the walk-forward block)
- At each step, predict using the most recently fine-tuned model
- Save forecasts alongside zero-shot results for comparison

Usage:
    python run_finetune_lagllama.py --tickers AAPL --horizons 1 --device cuda
    python run_finetune_lagllama.py --all-tickers --horizons 1 5 22 --device cuda
"""

import sys
import argparse
import pandas as pd
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    forecast_cfg, fm_cfg, VOLARE_RESULTS_DIR,
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)
from data_loader import load_data
from models.foundation import LagLlamaModel
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger

logger = setup_logger("finetune_lagllama")

FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
CONTEXT_LENGTH = 512
RETRAIN_EVERY = 126  # Fine-tune every 126 steps (one walk-forward block)
MAX_EPOCHS = 10


def run_finetune_walkforward(
    series: pd.Series,
    ticker: str,
    horizon: int,
    model: LagLlamaModel,
    train_window: int = 252,
):
    """Walk-forward with periodic fine-tuning.

    Parameters
    ----------
    series : pd.Series
        Full RV time series with DatetimeIndex.
    ticker : str
        Ticker symbol.
    horizon : int
        Forecast horizon.
    model : LagLlamaModel
        Lag-Llama model instance (with max_epochs > 0).
    train_window : int
        Minimum training window before first prediction.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'actual' and 'forecast' columns, DatetimeIndex.
    """
    values = series.values.astype(np.float64)
    dates = series.index
    n = len(values)
    min_start = max(train_window, CONTEXT_LENGTH)

    if n <= min_start + horizon:
        logger.warning(f"  {ticker} h={horizon}: too short ({n} obs)")
        return None

    actuals = []
    forecasts = []
    forecast_dates = []

    current_predictor = None
    last_train_step = -RETRAIN_EVERY  # Force training at first step

    for t in range(min_start, n - horizon):
        step_num = t - min_start

        # Fine-tune periodically
        if step_num - last_train_step >= RETRAIN_EVERY or current_predictor is None:
            train_data = values[max(0, t - train_window):t]
            logger.info(f"  Fine-tuning at step {step_num} (t={t}, "
                        f"train_len={len(train_data)})")
            t0 = time.time()
            current_predictor = model.fine_tune_predictor(train_data, horizon)
            logger.info(f"  Fine-tuning took {time.time() - t0:.1f}s")
            last_train_step = step_num

        # Predict
        context = values[max(0, t - CONTEXT_LENGTH):t]
        fc = model.predict_with_predictor(current_predictor, context, horizon)

        if horizon == 1:
            actual_val = values[t + 1]
            pred_val = float(fc.point[0])
            fc_date = dates[t + 1]
        else:
            actual_val = float(np.mean(values[t + 1:t + 1 + horizon]))
            pred_val = float(np.mean(fc.point[:horizon]))
            fc_date = dates[t + horizon] if t + horizon < n else dates[-1]

        actuals.append(actual_val)
        forecasts.append(pred_val)
        forecast_dates.append(fc_date)

    if not actuals:
        return None

    return pd.DataFrame({
        'actual': actuals,
        'forecast': forecasts,
    }, index=pd.DatetimeIndex(forecast_dates, name='date'))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Lag-Llama on VOLARE RV data"
    )
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--horizons', nargs='+', type=int, default=[1])
    parser.add_argument('--device', default=fm_cfg.device)
    parser.add_argument('--all-tickers', action='store_true')
    parser.add_argument('--asset-class', default='stocks',
                        choices=['stocks', 'fx', 'futures'])
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--retrain-every', type=int, default=RETRAIN_EVERY)
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()

    # Select tickers
    if args.all_tickers:
        ticker_map = {
            'stocks': VOLARE_STOCK_TICKERS,
            'fx': VOLARE_FX_TICKERS,
            'futures': VOLARE_FUTURES_TICKERS,
        }
        tickers = ticker_map[args.asset_class]
    else:
        tickers = args.tickers or ['AAPL']

    horizons = args.horizons

    logger.info(f"=== Lag-Llama Fine-Tuning (VOLARE) ===")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Retrain every: {args.retrain_every} steps")

    # Load data
    data = load_data(dataset="volare", asset_class=args.asset_class)
    logger.info(f"Data loaded: {data.shape[1]} assets, {data.shape[0]} dates")

    # Initialize model
    model = LagLlamaModel(
        context_length=CONTEXT_LENGTH,
        num_samples=args.num_samples,
        device=args.device,
        max_epochs=args.max_epochs,
    )
    model.load_model()

    total = len(tickers) * len(horizons)
    done = 0

    for ticker in tickers:
        if ticker not in data.columns:
            logger.warning(f"Ticker {ticker} not in data, skipping")
            continue

        series = data[ticker].dropna()

        for h in horizons:
            done += 1
            safe_name = "lag_llama_ft"
            out_path = FORECAST_DIR / f"{safe_name}_{ticker}_h{h}.csv"

            if args.skip_existing and out_path.exists():
                logger.info(f"  Skipping [{done}/{total}] {ticker} h={h}: exists")
                continue

            logger.info(f"Running [{done}/{total}] Lag-Llama-FT | {ticker} | h={h}")
            t0 = time.time()

            try:
                result_df = run_finetune_walkforward(
                    series, ticker, h, model,
                    train_window=forecast_cfg.train_window,
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

            if result_df is None or len(result_df) == 0:
                logger.warning(f"  No forecasts produced for {ticker} h={h}")
                continue

            elapsed = time.time() - t0

            # Save
            FORECAST_DIR.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(out_path)

            # Compute metrics
            metrics = compute_all_losses(result_df['actual'], result_df['forecast'])
            logger.info(
                f"  Done [{done}/{total}] {ticker} h={h} in {elapsed:.1f}s | "
                f"n={len(result_df)} | R2={metrics.get('R2OOS', 0):.3f} | "
                f"QLIKE={metrics.get('QLIKE', 0):.4f}"
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
