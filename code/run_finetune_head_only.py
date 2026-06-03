"""
run_finetune_head_only.py — Rolling FT-Head experiment grid runner for TSFMs.

Replaces the per-model run_*_grid_experiment.py scripts. Iterates over all
combinations of --contexts, --horizons, --selection-metrics, and
--train-target-transforms, saves an Excel workbook after each combination,
and resumes from an existing workbook on restart.

Examples:
    python -m run_finetune_head_only --models lag-llama --device cpu
    python -m run_finetune_head_only --models timesfm-2.5 --device cpu --contexts 256 512 --horizons 22
    python -m run_finetune_head_only --models lag-llama --device cpu --output-excel results.xlsx --overwrite
"""

import sys
import argparse
import time
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR, VOLARE_STOCK_TICKERS
from fine_tuning.pipeline import run_head_only_experiment
from fine_tuning.protocol import (
    FineTuneExperimentConfig,
    FineTuneGrid,
    FineTuneRuntimeConfig,
    RollingRefitConfig,
)

_GRID_CONTEXTS = [128, 256, 512]
_GRID_HORIZONS = [1, 5, 22]
_GRID_SELECTION_METRICS = ["QLIKE", "MSE", "MAE"]
_GRID_TRAIN_TRANSFORMS = ["log", "level"]


def _safe_name(model_name: str) -> str:
    return model_name.replace("-", "_").replace(".", "_")


def _default_excel(models: list[str]) -> Path:
    stem = "_".join(_safe_name(m) for m in models)
    return VOLARE_RESULTS_DIR / "fine_tune_head" / "metrics" / f"{stem}_full_grid_results.xlsx"


def _summarize_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols)[["QLIKE", "MSE", "MAE"]].mean().reset_index()


def _save_workbook(output_excel: Path, block_metrics_df: pd.DataFrame, runs_df: pd.DataFrame) -> None:
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


def run_grid(args) -> None:
    tickers = args.tickers or VOLARE_STOCK_TICKERS
    output_excel = Path(args.output_excel) if args.output_excel else _default_excel(args.models)
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    all_block_metrics = []
    run_registry = []
    completed_keys: set[tuple] = set()

    if output_excel.exists() and not args.overwrite:
        try:
            existing_runs = pd.read_excel(output_excel, sheet_name="runs")
            if not existing_runs.empty:
                run_registry = existing_runs.to_dict("records")
                completed_keys = {
                    (row["model"], row["context_length"], row["horizon"],
                     row["selection_metric"], row["train_target_transform"])
                    for _, row in existing_runs.iterrows()
                }
            existing_blocks = pd.read_excel(output_excel, sheet_name="block_metrics")
            if not existing_blocks.empty:
                all_block_metrics.append(existing_blocks)
            print(f"Resuming from existing workbook: {output_excel}")
        except Exception:
            print(f"Could not resume from {output_excel}; starting fresh.")

    runtime = FineTuneRuntimeConfig(device=args.device, batch_size=args.batch_size)
    rolling = RollingRefitConfig(
        min_train_size=args.min_train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    contexts = args.contexts or _GRID_CONTEXTS
    horizons = args.horizons or _GRID_HORIZONS
    selection_metrics = args.selection_metrics or _GRID_SELECTION_METRICS
    train_transforms = args.train_target_transforms or _GRID_TRAIN_TRANSFORMS

    total_runs = len(args.models) * len(contexts) * len(horizons) * len(selection_metrics) * len(train_transforms)
    run_id = len(run_registry)
    started = time.time()

    for model_name, context_length, horizon, selection_metric, train_target_transform in product(
        args.models, contexts, horizons, selection_metrics, train_transforms
    ):
        combo_key = (model_name, context_length, horizon, selection_metric, train_target_transform)
        if combo_key in completed_keys:
            print(
                f"Skipping completed: {model_name} ctx={context_length} h={horizon} "
                f"sel={selection_metric} train={train_target_transform}"
            )
            continue

        run_id += 1
        run_start = time.time()
        print(
            f"[{len(completed_keys) + 1}/{total_runs}] {model_name} "
            f"ctx={context_length} h={horizon} sel={selection_metric} train={train_target_transform}"
        )

        grid = FineTuneGrid(
            models=[model_name],
            contexts=[context_length],
            horizons=[horizon],
            selection_metric=selection_metric,
            train_target_transform=train_target_transform,
        )
        cfg = FineTuneExperimentConfig(
            grid=grid,
            rolling=rolling,
            runtime=runtime,
            tickers=tickers,
            output_root=VOLARE_RESULTS_DIR / "fine_tune_head",
        )

        block_df = run_head_only_experiment(cfg, dataset_key="volare")
        if block_df.empty:
            print(
                f"  WARNING: no results for {model_name} ctx={context_length} h={horizon} "
                f"sel={selection_metric} train={train_target_transform} — "
                "check console for 'Adapter pending' warnings above (likely a missing dependency or import error)."
            )
            continue

        block_df = block_df.copy()
        block_df["run_id"] = run_id
        all_block_metrics.append(block_df)

        run_registry.append({
            "run_id": run_id,
            "model": model_name,
            "context_length": context_length,
            "horizon": horizon,
            "selection_metric": selection_metric,
            "train_target_transform": train_target_transform,
            "n_tickers": len(tickers),
        })
        completed_keys.add(combo_key)

        block_metrics_df = pd.concat(all_block_metrics, ignore_index=True) if all_block_metrics else pd.DataFrame()
        _save_workbook(output_excel, block_metrics_df, pd.DataFrame(run_registry))

        elapsed = time.time() - run_start
        total_elapsed = time.time() - started
        avg_per_run = total_elapsed / max(len(completed_keys), 1)
        eta_min = avg_per_run * (total_runs - len(completed_keys)) / 60.0
        print(f"  Done in {elapsed/60.0:.1f} min | {len(completed_keys)}/{total_runs} | ETA ~{eta_min:.1f} min")

    block_metrics_df = pd.concat(all_block_metrics, ignore_index=True) if all_block_metrics else pd.DataFrame()
    _save_workbook(output_excel, block_metrics_df, pd.DataFrame(run_registry))
    print(f"Saved workbook: {output_excel}")


def main():
    parser = argparse.ArgumentParser(description="Run aligned rolling FT-Head TSFM experiment grid and export Excel")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--contexts", nargs="+", type=int, default=None,
                        help="Context lengths to sweep (default: 128 256 512)")
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Horizons to sweep (default: 1 5 22)")
    parser.add_argument("--selection-metrics", nargs="+", default=None,
                        help="Selection metrics to sweep (default: QLIKE MSE MAE)")
    parser.add_argument("--train-target-transforms", nargs="+", default=None,
                        help="Train transforms to sweep (default: log level)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-train-size", type=int, default=756)
    parser.add_argument("--validation-size", type=int, default=126)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--step-size", type=int, default=126)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-excel", default=None, help="Output workbook path")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing workbook and start fresh")
    args = parser.parse_args()
    run_grid(args)


if __name__ == "__main__":
    main()
