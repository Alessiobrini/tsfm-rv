"""
pipeline.py — Rolling orchestration for the aligned head-only TSFM study.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_loader import load_data
from evaluation.loss_functions import compute_all_losses, qlike
from fine_tuning.adapters import HeadOnlyAdapterError, build_head_only_adapter
from fine_tuning.data import (
    block_test_indices,
    block_train_indices,
    block_val_indices,
    build_windowed_dataset,
    generate_refit_blocks,
)
from fine_tuning.protocol import FineTuneExperimentConfig
from utils import setup_logger


def _safe_name(model_name: str) -> str:
    return model_name.replace("-", "_").replace(".", "_")


def _flat_stem(
    model_name: str,
    ticker: str,
    horizon: int,
    context_length: int,
    selection_metric: str,
    train_target_transform: str,
) -> str:
    return (
        f"{_safe_name(model_name)}_{ticker}_h{horizon}_ctx{context_length}"
        f"_sel{selection_metric.lower()}_train{train_target_transform.lower()}"
    )


def _forecast_output_dir(root: Path) -> Path:
    return root / "forecasts"


def _metrics_output_dir(root: Path) -> Path:
    return root / "metrics"


def _diagnostics_output_dir(root: Path) -> Path:
    return root / "diagnostics"


def _build_debug_frame(
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    forecast: np.ndarray,
    split: str,
    refit_id: int,
) -> pd.DataFrame:
    ratio = actual / np.maximum(forecast, 1e-10)
    return pd.DataFrame({
        "date": dates,
        "split": split,
        "refit_id": refit_id,
        "actual": actual,
        "forecast": forecast,
        "forecast_to_actual": forecast / np.maximum(actual, 1e-10),
        "actual_to_forecast": ratio,
        "abs_error": np.abs(actual - forecast),
        "log_actual": np.log(np.maximum(actual, 1e-10)),
        "log_forecast": np.log(np.maximum(forecast, 1e-10)),
    })


def _forecast_scale_summary(actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
    ratio = forecast / np.maximum(actual, 1e-10)
    return {
        "actual_mean": float(np.mean(actual)),
        "forecast_mean": float(np.mean(forecast)),
        "actual_median": float(np.median(actual)),
        "forecast_median": float(np.median(forecast)),
        "forecast_to_actual_mean": float(np.mean(ratio)),
        "forecast_to_actual_median": float(np.median(ratio)),
    }


def _save_debug_plot(debug_df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = debug_df.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_df["date"], plot_df["actual"], label="actual", linewidth=1.5, color="0.25")
    ax.plot(plot_df["date"], plot_df["forecast"], label="forecast", linewidth=1.2, color="#c0392b")
    ax.set_title(title)
    ax.set_ylabel("Realized Variance")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summarize_metrics(metrics_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate block metrics for the active FT workflow."""

    return (
        metrics_df.groupby(group_cols)[["QLIKE", "MSE", "MAE"]]
        .mean()
        .reset_index()
    )


def run_head_only_experiment(
    cfg: FineTuneExperimentConfig,
    dataset_key: str = "volare",
) -> pd.DataFrame:
    """Run the aligned rolling head-only study and return block-level metrics."""

    logger = setup_logger("fine_tune_head")
    logger.info("=== Aligned TSFM FT-Head Study ===")
    logger.info(f"Models: {cfg.grid.models}")
    logger.info(f"Contexts: {cfg.grid.contexts}")
    logger.info(f"Horizons: {cfg.grid.horizons}")
    logger.info(f"Selection metric: {cfg.grid.selection_metric}")
    logger.info(f"Train target transform: {cfg.grid.train_target_transform}")
    logger.info(
        "Rolling schedule: "
        f"min_train={cfg.rolling.min_train_size}, "
        f"val={cfg.rolling.validation_size}, "
        f"test={cfg.rolling.test_size}, "
        f"step={cfg.rolling.step_size}"
    )

    data = load_data(dataset=dataset_key, tickers=cfg.tickers)
    tickers = cfg.tickers or data.tickers
    logger.info(f"Loaded dataset '{dataset_key}' with {len(tickers)} equity tickers.")

    all_rows: List[Dict] = []
    stitched_debug: Dict[tuple, List[pd.DataFrame]] = {}
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        series = data.rv[ticker].dropna()
        if len(series) < cfg.rolling.min_train_size + cfg.rolling.validation_size + cfg.rolling.test_size:
            logger.warning(f"Skipping {ticker}: insufficient history ({len(series)} obs).")
            continue

        blocks = generate_refit_blocks(len(series), cfg.rolling)
        logger.info(f"{ticker}: {len(blocks)} rolling refit blocks.")

        for model_name in cfg.grid.models:
            for context_length in cfg.grid.contexts:
                for horizon in cfg.grid.horizons:
                    adapter = build_head_only_adapter(
                        model_name=model_name,
                        context_length=context_length,
                        horizon=horizon,
                        runtime_cfg=cfg.runtime,
                        selection_metric=cfg.grid.selection_metric,
                        train_target_transform=cfg.grid.train_target_transform,
                    )
                    for block in blocks:
                        train_ds = build_windowed_dataset(
                            series,
                            block_train_indices(block, horizon),
                            context_length=context_length,
                            horizon=horizon,
                        )
                        val_ds = build_windowed_dataset(
                            series,
                            block_val_indices(block, horizon),
                            context_length=context_length,
                            horizon=horizon,
                        )
                        test_ds = build_windowed_dataset(
                            series,
                            block_test_indices(block, horizon),
                            context_length=context_length,
                            horizon=horizon,
                        )

                        if train_ds.is_empty or val_ds.is_empty or test_ds.is_empty:
                            logger.warning(
                                f"Skipping empty block: {ticker} | {model_name} | "
                                f"ctx={context_length} | h={horizon} | block={block.refit_id}"
                            )
                            continue

                        try:
                            selection = adapter.fit(train_ds, val_ds)
                            test_forecast = adapter.predict_levels(test_ds.contexts)
                            val_forecast = adapter.predict_levels(val_ds.contexts)
                        except HeadOnlyAdapterError as exc:
                            logger.warning(
                                f"Adapter pending: {ticker} | {model_name} | "
                                f"ctx={context_length} | h={horizon} | block={block.refit_id} | {exc}"
                            )
                            continue

                        test_metrics = compute_all_losses(test_ds.targets_level, test_forecast)
                        val_qlike = qlike(val_ds.targets_level, val_forecast)
                        val_scale = _forecast_scale_summary(val_ds.targets_level, val_forecast)
                        test_scale = _forecast_scale_summary(test_ds.targets_level, test_forecast)

                        core_metrics = {k: v for k, v in test_metrics.items() if k in {"QLIKE", "MSE", "MAE"}}

                        row = {
                            "ticker": ticker,
                            "model": model_name,
                            "context_length": context_length,
                            "horizon": horizon,
                            "selection_metric": cfg.grid.selection_metric,
                            "train_target_transform": cfg.grid.train_target_transform,
                            "refit_id": block.refit_id,
                            "train_end_date": series.index[block.train_end - 1],
                            "val_start_date": series.index[block.val_start],
                            "val_end_date": series.index[block.val_end - 1],
                            "test_start_date": series.index[block.test_start],
                            "test_end_date": series.index[block.test_end - 1],
                            "n_train": len(train_ds),
                            "n_val": len(val_ds),
                            "n_test": len(test_ds),
                            "val_qlike_selected": val_qlike,
                            "selection_best_epoch": selection.best_epoch,
                            "selection_best_val_qlike": selection.best_val_qlike,
                            "selection_notes": selection.notes,
                            "val_actual_mean": val_scale["actual_mean"],
                            "val_forecast_mean": val_scale["forecast_mean"],
                            "val_forecast_to_actual_mean": val_scale["forecast_to_actual_mean"],
                            "val_forecast_to_actual_median": val_scale["forecast_to_actual_median"],
                            "test_actual_mean": test_scale["actual_mean"],
                            "test_forecast_mean": test_scale["forecast_mean"],
                            "test_forecast_to_actual_mean": test_scale["forecast_to_actual_mean"],
                            "test_forecast_to_actual_median": test_scale["forecast_to_actual_median"],
                            **core_metrics,
                        }
                        if selection.extra:
                            for key, value in selection.extra.items():
                                row[f"selection_{key}"] = value
                        all_rows.append(row)

                        out_dir = _forecast_output_dir(cfg.output_root)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stem = _flat_stem(
                            model_name,
                            ticker,
                            horizon,
                            context_length,
                            cfg.grid.selection_metric,
                            cfg.grid.train_target_transform,
                        )
                        pd.DataFrame({
                            "date": test_ds.forecast_dates,
                            "actual": test_ds.targets_level,
                            "forecast": test_forecast,
                        }).to_csv(out_dir / f"{stem}_refit_{block.refit_id:03d}.csv", index=False)

                        diag_dir = _diagnostics_output_dir(cfg.output_root)
                        diag_dir.mkdir(parents=True, exist_ok=True)

                        val_debug = _build_debug_frame(
                            val_ds.forecast_dates,
                            val_ds.targets_level,
                            val_forecast,
                            split="validation",
                            refit_id=block.refit_id,
                        )
                        test_debug = _build_debug_frame(
                            test_ds.forecast_dates,
                            test_ds.targets_level,
                            test_forecast,
                            split="test",
                            refit_id=block.refit_id,
                        )
                        val_debug.to_csv(diag_dir / f"{stem}_validation_refit_{block.refit_id:03d}.csv", index=False)
                        test_debug.to_csv(diag_dir / f"{stem}_test_refit_{block.refit_id:03d}.csv", index=False)

                        summary_df = pd.DataFrame([
                            {"split": "validation", "refit_id": block.refit_id, **val_scale, "qlike": val_qlike},
                            {"split": "test", "refit_id": block.refit_id, **test_scale, "qlike": test_metrics["QLIKE"]},
                        ])
                        summary_df.to_csv(diag_dir / f"{stem}_scale_summary_refit_{block.refit_id:03d}.csv", index=False)

                        logger.info(
                            f"{ticker} | {model_name} | ctx={context_length} | h={horizon} | "
                            f"block={block.refit_id} | val_qlike={val_qlike:.4f} | "
                            f"val f/a mean={val_scale['forecast_to_actual_mean']:.2f} | "
                            f"test f/a mean={test_scale['forecast_to_actual_mean']:.2f}"
                        )

                        debug_key = (
                            ticker,
                            model_name,
                            horizon,
                            context_length,
                            cfg.grid.selection_metric,
                            cfg.grid.train_target_transform,
                        )
                        stitched_debug.setdefault(debug_key, []).append(test_debug)

    metrics_df = pd.DataFrame(all_rows)
    metrics_dir = _metrics_output_dir(cfg.output_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_dir / "block_metrics.csv", index=False)

        summary = _summarize_metrics(
            metrics_df,
            ["model", "context_length", "horizon", "selection_metric", "train_target_transform"],
        )
        summary.to_csv(metrics_dir / "summary_metrics.csv", index=False)

    for (ticker, model_name, horizon, context_length, selection_metric, train_target_transform), frames in stitched_debug.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True).sort_values("date")
        diag_dir = _diagnostics_output_dir(cfg.output_root)
        diag_dir.mkdir(parents=True, exist_ok=True)
        stem = _flat_stem(model_name, ticker, horizon, context_length, selection_metric, train_target_transform)
        combined.to_csv(diag_dir / f"{stem}_stitched_test_debug.csv", index=False)
        if ticker == "AAPL":
            _save_debug_plot(
                combined,
                diag_dir / f"{stem}_stitched_test_debug.png",
                title=f"{ticker} | {model_name} | h={horizon} | ctx={context_length}",
            )

    return metrics_df
