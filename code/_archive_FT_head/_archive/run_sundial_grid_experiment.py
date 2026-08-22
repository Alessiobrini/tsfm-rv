"""
run_sundial_grid_experiment.py — Batch-run the full Sundial FT-Head grid and export one Excel workbook.

Grid:
    - context lengths: 128, 256, 512
    - horizons: 1, 5, 22
    - selection metrics: QLIKE, MSE, MAE
    - training targets: log(RV), RV
    - full VOLARE equity sample

Outputs:
    - one Excel workbook with summary metrics, block metrics, and run metadata
    - underlying per-block forecasts/diagnostics still saved by the FT-Head pipeline
"""

import sys
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR, VOLARE_STOCK_TICKERS
from data_loader import load_data
from fine_tuning.adapters import build_head_only_adapter
from fine_tuning.pipeline import run_head_only_experiment
from fine_tuning.protocol import (
    FineTuneExperimentConfig,
    FineTuneGrid,
    FineTuneRuntimeConfig,
    RollingRefitConfig,
)


def _summarize_metrics(block_metrics_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        block_metrics_df.groupby(group_cols)[["QLIKE", "MSE", "MAE"]]
        .mean()
        .reset_index()
    )


def _prewarm_sundial_cache(
    tickers: list[str],
    context_length: int,
    horizon: int,
    runtime: FineTuneRuntimeConfig,
) -> None:
    """Populate the Sundial raw-sample cache once per (ctx, h) over the full panel."""

    data = load_data(dataset="volare", tickers=tickers)
    adapter = build_head_only_adapter(
        model_name="sundial",
        context_length=context_length,
        horizon=horizon,
        runtime_cfg=runtime,
        selection_metric="QLIKE",
        train_target_transform="log",
    )

    for ticker in tickers:
        series = data.rv[ticker].dropna()
        values = series.values.astype("float32", copy=False)
        if len(values) <= context_length + horizon:
            continue

        contexts = []
        for idx in range(context_length, len(values) - horizon + 1):
            ctx = values[idx - context_length:idx]
            if ctx.shape[0] == context_length:
                contexts.append(ctx)

        if not contexts:
            continue

        adapter._raw_samples(np.asarray(contexts, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="Run the full Sundial FT-Head experiment grid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tickers", nargs="+", default=None, help="Optional equity subset; default is full sample")
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])
    parser.add_argument("--selection-metrics", nargs="+", default=["QLIKE", "MSE", "MAE"])
    parser.add_argument("--train-target-transforms", nargs="+", default=["log", "level"])
    parser.add_argument("--min-train-size", type=int, default=756)
    parser.add_argument("--validation-size", type=int, default=126)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--step-size", type=int, default=126)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-excel", default=None)
    args = parser.parse_args()

    tickers = args.tickers or VOLARE_STOCK_TICKERS
    output_excel = Path(args.output_excel) if args.output_excel else (
        VOLARE_RESULTS_DIR / "fine_tune_head" / "metrics" / "sundial_full_grid_results.xlsx"
    )
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    all_block_metrics = []
    run_registry = []

    run_id = 0
    runtime = FineTuneRuntimeConfig(
        device=args.device,
        batch_size=args.batch_size,
    )
    for context_length, horizon in product(args.contexts, args.horizons):
        _prewarm_sundial_cache(tickers, context_length, horizon, runtime)

        for selection_metric, train_target_transform in product(
            args.selection_metrics,
            args.train_target_transforms,
        ):
            run_id += 1
            grid = FineTuneGrid(
                models=["sundial"],
                contexts=[context_length],
                horizons=[horizon],
                selection_metric=selection_metric,
                train_target_transform=train_target_transform,
            )
            rolling = RollingRefitConfig(
                min_train_size=args.min_train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                step_size=args.step_size,
            )
            cfg = FineTuneExperimentConfig(
                grid=grid,
                rolling=rolling,
                runtime=runtime,
                tickers=tickers,
                output_root=VOLARE_RESULTS_DIR / "fine_tune_head",
            )

            block_df = run_head_only_experiment(cfg, dataset_key="volare")
            if not block_df.empty:
                block_df = block_df.copy()
                block_df["run_id"] = run_id
                all_block_metrics.append(block_df)

            run_registry.append({
                "run_id": run_id,
                "model": "sundial",
                "context_length": context_length,
                "horizon": horizon,
                "selection_metric": selection_metric,
                "train_target_transform": train_target_transform,
                "n_tickers": len(tickers),
            })

    runs_df = pd.DataFrame(run_registry)
    block_metrics_df = pd.concat(all_block_metrics, ignore_index=True) if all_block_metrics else pd.DataFrame()

    if not block_metrics_df.empty:
        summary_df = _summarize_metrics(
            block_metrics_df,
            ["model", "context_length", "horizon", "selection_metric", "train_target_transform"],
        ).sort_values(["horizon", "context_length", "train_target_transform", "selection_metric"])
        asset_summary_df = _summarize_metrics(
            block_metrics_df,
            ["ticker", "model", "context_length", "horizon", "selection_metric", "train_target_transform"],
        ).sort_values(["ticker", "horizon", "context_length", "train_target_transform", "selection_metric"])
    else:
        summary_df = pd.DataFrame()
        asset_summary_df = pd.DataFrame()

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        asset_summary_df.to_excel(writer, sheet_name="by_asset", index=False)
        block_metrics_df.to_excel(writer, sheet_name="block_metrics", index=False)
        runs_df.to_excel(writer, sheet_name="runs", index=False)

    print(f"Saved workbook: {output_excel}")


if __name__ == "__main__":
    main()
