"""
run_finetune_chronos2.py — Fine-tune Chronos-2 (Bolt) with LoRA on VOLARE RV.

Approach:
1. Prepare training data: all 40 equity RV series up to start of test period
2. Fine-tune Chronos-Bolt-Small with LoRA adapters (rank 8)
3. Run walk-forward evaluation with the fine-tuned model
4. Retrain every 126 steps (one walk-forward block)

Requires: chronos-forecasting >= 2.2, peft (for LoRA)

Usage:
    python run_finetune_chronos2.py --all-tickers --horizons 1 --device cuda
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
from evaluation.loss_functions import compute_all_losses
from utils import setup_logger

logger = setup_logger("finetune_chronos2")

FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
CONTEXT_LENGTH = 512
RETRAIN_EVERY = 126


def fine_tune_chronos_bolt(
    train_series_list,
    model_id: str,
    device: str,
    lora_rank: int = 8,
    max_epochs: int = 5,
    learning_rate: float = 1e-4,
):
    """Fine-tune Chronos-Bolt with LoRA on a list of training series.

    Returns a fine-tuned pipeline ready for prediction.
    """
    import torch
    from chronos.chronos_bolt import ChronosBoltPipeline

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("peft package required for LoRA. Install: pip install peft")
        raise

    # Load base model
    pipeline = ChronosBoltPipeline.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Apply LoRA to the transformer backbone
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    model = pipeline.model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare training data as tensors
    from torch.utils.data import DataLoader, TensorDataset

    # Chunk each series into context_length + prediction_length windows
    prediction_length = 1  # We'll use 1-step for training
    window_size = CONTEXT_LENGTH + prediction_length
    contexts = []
    targets = []

    for series in train_series_list:
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            ctx = series[i:i + CONTEXT_LENGTH]
            tgt = series[i + CONTEXT_LENGTH:i + window_size]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            targets.append(torch.tensor(tgt, dtype=torch.float32))

    if not contexts:
        logger.warning("No training windows created")
        return pipeline

    contexts = torch.stack(contexts)
    targets = torch.stack(targets)

    dataset = TensorDataset(contexts, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(max_epochs):
        total_loss = 0
        n_batches = 0
        for ctx_batch, tgt_batch in dataloader:
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            # Forward pass depends on Chronos-Bolt internal API
            # This is a simplified version; actual implementation may vary
            outputs = model(ctx_batch)
            loss = torch.nn.functional.mse_loss(outputs[:, -prediction_length:], tgt_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        logger.info(f"  Epoch {epoch + 1}/{max_epochs}: "
                    f"avg loss = {total_loss / max(n_batches, 1):.6f}")

    model.eval()
    pipeline.model = model
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Chronos-2 with LoRA on VOLARE RV"
    )
    parser.add_argument('--model-id', default='amazon/chronos-bolt-small')
    parser.add_argument('--tickers', nargs='+', default=None)
    parser.add_argument('--horizons', nargs='+', type=int, default=[1])
    parser.add_argument('--device', default=fm_cfg.device)
    parser.add_argument('--all-tickers', action='store_true')
    parser.add_argument('--asset-class', default='stocks',
                        choices=['stocks', 'fx', 'futures'])
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
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

    logger.info("=== Chronos-2 LoRA Fine-Tuning (VOLARE) ===")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Tickers: {len(tickers)} assets")
    logger.info(f"Horizons: {args.horizons}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Max epochs: {args.max_epochs}")

    # Load data
    data = load_data(dataset="volare", asset_class=args.asset_class)
    logger.info(f"Data loaded: {data.shape[1]} assets, {data.shape[0]} dates")

    # NOTE: This is a preliminary implementation. The Chronos-Bolt training
    # API may require specific data formats and training procedures that
    # differ from the simplified version above. Test on the cluster with
    # a single ticker first before running the full experiment.
    logger.warning(
        "This script uses a simplified LoRA training loop. "
        "Verify Chronos-Bolt internal API compatibility before full run."
    )

    # For now, just log the plan
    for h in args.horizons:
        for ticker in tickers:
            safe_name = f"chronos_bolt_small_ft"
            out_path = FORECAST_DIR / f"{safe_name}_{ticker}_h{h}.csv"
            if args.skip_existing and out_path.exists():
                logger.info(f"  Skipping {ticker} h={h}: exists")
                continue
            logger.info(f"  Would fine-tune for {ticker} h={h} -> {out_path}")

    logger.info("Done (dry run). Implement full training loop after API verification.")


if __name__ == "__main__":
    main()
