"""
compare_ft_vs_zero_shot.py — Compare FT-Head Chronos forecasts against zero-shot.

Example:
    python -m compare_ft_vs_zero_shot \
        --ticker AAPL \
        --model chronos-bolt-small \
        --horizon 1 \
        --context-length 512
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR
from evaluation.loss_functions import compute_all_losses


def safe_name(model_name: str) -> str:
    return model_name.replace("-", "_").replace(".", "_").replace(" ", "_")


def default_zero_shot_path(model_name: str, ticker: str, horizon: int, context_length: int) -> Path:
    base = VOLARE_RESULTS_DIR / "forecasts"
    name = safe_name(model_name)
    if context_length == 512:
        return base / f"{name}_{ticker}_h{horizon}.csv"
    return base / f"{name}_{ticker}_h{horizon}_ctx{context_length}.csv"


def default_ft_path(model_name: str, ticker: str, horizon: int, context_length: int) -> Path:
    return (
        VOLARE_RESULTS_DIR
        / "fine_tune_head"
        / "diagnostics"
        / f"{safe_name(model_name)}_{ticker}_h{horizon}_ctx{context_length}_stitched_test_debug.csv"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare FT-Head forecasts against zero-shot")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--model", default="chronos-bolt-small")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--zero-shot-path", default=None)
    parser.add_argument("--ft-path", default=None)
    args = parser.parse_args()

    zero_path = Path(args.zero_shot_path) if args.zero_shot_path else default_zero_shot_path(
        args.model, args.ticker, args.horizon, args.context_length
    )
    ft_path = Path(args.ft_path) if args.ft_path else default_ft_path(
        args.model, args.ticker, args.horizon, args.context_length
    )

    if not zero_path.exists():
        raise FileNotFoundError(f"Zero-shot forecast file not found: {zero_path}")
    if not ft_path.exists():
        raise FileNotFoundError(f"FT forecast file not found: {ft_path}")

    zero_df = pd.read_csv(zero_path, parse_dates=["date"])
    ft_df = pd.read_csv(ft_path, parse_dates=["date"])

    zero_df = zero_df[["date", "actual", "forecast"]].rename(columns={"forecast": "forecast_zero"})
    ft_df = ft_df[["date", "actual", "forecast"]].rename(columns={"forecast": "forecast_ft"})

    merged = zero_df.merge(ft_df, on="date", how="inner", suffixes=("_zero_actual", "_ft_actual"))
    if merged.empty:
        raise ValueError("No common dates between zero-shot and FT forecasts.")

    # Prefer the zero-shot actuals column; both should agree on common dates.
    actual = merged["actual_zero_actual"] if "actual_zero_actual" in merged else merged.iloc[:, 1]

    zero_metrics = compute_all_losses(actual, merged["forecast_zero"])
    ft_metrics = compute_all_losses(actual, merged["forecast_ft"])

    summary = pd.DataFrame([
        {"variant": "zero_shot", **zero_metrics},
        {"variant": "ft_head", **ft_metrics},
    ])[["variant", "MSE", "MAE", "QLIKE"]]

    print("\nComparison on common dates")
    print(summary.to_string(index=False))

    out_dir = VOLARE_RESULTS_DIR / "fine_tune_head" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        out_dir / f"compare_{safe_name(args.model)}_{args.ticker}_h{args.horizon}_ctx{args.context_length}.csv",
        index=False,
    )
    summary.to_csv(
        out_dir / f"compare_{safe_name(args.model)}_{args.ticker}_h{args.horizon}_ctx{args.context_length}_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
