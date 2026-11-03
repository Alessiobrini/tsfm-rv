"""
run_lag_llama_grid_experiment.py — Batch-run the full Lag-Llama FT-Head grid and export one Excel workbook.

Grid:
    - context lengths: 128, 256, 512
    - horizons: 1, 5, 22
    - selection metrics: QLIKE, MSE, MAE
    - training targets: log(RV), RV
    - full VOLARE equity sample
"""

import sys
import argparse
from itertools import product
from pathlib import Path
import time

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


def _save_workbook(
    output_excel: Path,
    block_metrics_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> None:
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


def _summarize_metrics(block_metrics_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        block_metrics_df.groupby(group_cols)[["QLIKE", "MSE", "MAE"]]
        .mean()
        .reset_index()
    )


def _prewarm_lag_llama_cache(
    tickers: list[str],
    context_length: int,
    horizon: int,
    runtime: FineTuneRuntimeConfig,
) -> None:
    data = load_data(dataset="volare", tickers=tickers)
    adapter = build_head_only_adapter(
        model_name="lag-llama",
        context_length=context_length,
        horizon=horizon,
        runtime_cfg=runtime,
        selection_metric="QLIKE",
        train_target_transform="log",
    )

    for ticker in tickers:
        series = data.rv[ticker].dropna()
        values = series.values.astype(np.float32, copy=False)
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
    parser = argparse.ArgumentParser(description="Run the full Lag-Llama FT-Head experiment grid")
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tickers = args.tickers or VOLARE_STOCK_TICKERS
    output_excel = Path(args.output_excel) if args.output_excel else (
        VOLARE_RESULTS_DIR / "fine_tune_head" / "metrics" / "lag_llama_full_grid_results.xlsx"
    )
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    all_block_metrics = []
    run_registry = []
    completed_keys = set()

    if output_excel.exists() and not args.overwrite:
        try:
            existing_runs = pd.read_excel(output_excel, sheet_name="runs")
            if not existing_runs.empty:
                run_registry = existing_runs.to_dict("records")
                completed_keys = {
                    (
                        row["context_length"],
                        row["horizon"],
                        row["selection_metric"],
                        row["train_target_transform"],
                    )
                    for _, row in existing_runs.iterrows()
                }
            existing_blocks = pd.read_excel(output_excel, sheet_name="block_metrics")
            if not existing_blocks.empty:
                all_block_metrics.append(existing_blocks)
            print(f"Resuming from existing workbook: {output_excel}")
        except Exception:
            print(f"Could not resume from {output_excel}; starting fresh.")

    run_id = len(run_registry)
    runtime = FineTuneRuntimeConfig(
        device=args.device,
        batch_size=args.batch_size,
    )
    total_runs = len(args.contexts) * len(args.horizons) * len(args.selection_metrics) * len(args.train_target_transforms)
    started = time.time()

    for context_length, horizon in product(args.contexts, args.horizons):
        _prewarm_lag_llama_cache(tickers, context_length, horizon, runtime)

        for selection_metric, train_target_transform in product(
            args.selection_metrics,
            args.train_target_transforms,
        ):
            combo_key = (context_length, horizon, selection_metric, train_target_transform)
            if combo_key in completed_keys:
                print(
                    f"Skipping completed run: ctx={context_length} h={horizon} "
                    f"sel={selection_metric} train={train_target_transform}"
                )
                continue

            run_id += 1
            run_start = time.time()
            print(
                f"[{len(completed_keys) + 1}/{total_runs}] Running Lag-Llama "
                f"ctx={context_length} h={horizon} sel={selection_metric} "
                f"train={train_target_transform}"
            )
            grid = FineTuneGrid(
                models=["lag-llama"],
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
                "model": "lag-llama",
                "context_length": context_length,
                "horizon": horizon,
                "selection_metric": selection_metric,
                "train_target_transform": train_target_transform,
                "n_tickers": len(tickers),
            })
            completed_keys.add(combo_key)

            block_metrics_df = pd.concat(all_block_metrics, ignore_index=True) if all_block_metrics else pd.DataFrame()
            runs_df = pd.DataFrame(run_registry)
            _save_workbook(output_excel, block_metrics_df, runs_df)

            elapsed = time.time() - run_start
            total_elapsed = time.time() - started
            avg_per_run = total_elapsed / max(len(completed_keys), 1)
            remaining = total_runs - len(completed_keys)
            eta_min = (avg_per_run * remaining) / 60.0
            print(
                f"Finished in {elapsed/60.0:.1f} min | "
                f"{len(completed_keys)}/{total_runs} done | ETA ~ {eta_min:.1f} min"
            )

    runs_df = pd.DataFrame(run_registry)
    block_metrics_df = pd.concat(all_block_metrics, ignore_index=True) if all_block_metrics else pd.DataFrame()
    _save_workbook(output_excel, block_metrics_df, runs_df)

    print(f"Saved workbook: {output_excel}")


if __name__ == "__main__":
    main()
