"""
rerun_portfolio_with_floor.py — Re-evaluate GMV portfolios with a higher eigenvalue floor.

Loads existing stocks .npz forecast files, re-projects each forecast matrix
using ensure_psd with a configurable floor (default 1e-6), and runs portfolio
evaluation. Saves results to a NEW file (does not overwrite existing).

Usage:
    python rerun_portfolio_with_floor.py --floor 1e-6
    python rerun_portfolio_with_floor.py --floor 1e-7
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import COV_RESULTS_DIR, forecast_cfg
from covariance_utils import ensure_psd, build_gmv_weights
from utils import setup_logger


def compute_portfolio_with_floor(npz_path, floor, logger):
    """Load .npz, re-project with new floor, compute GMV portfolio metrics."""
    data = np.load(npz_path, allow_pickle=True)
    dates = data["dates"]
    forecasts = data["forecasts"]
    actuals = data["actuals"]
    assets = list(data["assets"])
    n = len(assets)

    daily_results = []
    prev_weights = np.ones(n) / n

    for i, (date, f_mat, a_mat) in enumerate(zip(dates, forecasts, actuals)):
        # Re-project with new floor
        f_psd = ensure_psd(f_mat, min_eigenvalue=floor)

        # Build GMV weights
        try:
            weights = build_gmv_weights(f_psd)
        except Exception as e:
            logger.warning(f"  GMV failed on {date}: {e}")
            weights = np.ones(n) / n

        # Realized portfolio variance: w' Sigma_actual w
        realized_var = float(weights @ a_mat @ weights)

        # Turnover
        turnover = float(np.sum(np.abs(weights - prev_weights)))
        max_weight = float(np.max(np.abs(weights)))

        daily_results.append({
            "date": date,
            "realized_var": realized_var,
            "turnover": turnover,
            "max_weight": max_weight,
        })

        prev_weights = weights

    return pd.DataFrame(daily_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=float, default=1e-6,
                        help="Eigenvalue floor for PSD projection")
    parser.add_argument("--asset-class", default="stocks",
                        choices=["stocks", "forex", "futures"])
    args = parser.parse_args()

    floor = args.floor
    floor_label = f"{floor:.0e}".replace("+", "").replace("0", "").rstrip("e")
    # Clean label: 1e-6 -> "1e-06" -> "1e-6"
    floor_label = f"{floor:.0e}"

    logger = setup_logger("portfolio_floor")
    logger.info(f"=== Portfolio Re-evaluation with floor={floor} ===")
    logger.info(f"Asset class: {args.asset_class}")

    forecast_dir = COV_RESULTS_DIR / args.asset_class / "forecasts"
    tables_dir = COV_RESULTS_DIR / args.asset_class / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    horizons = forecast_cfg.horizons
    all_results = []

    for h in horizons:
        pattern = f"*_h{h}.npz"
        npz_files = list(forecast_dir.glob(pattern))

        if not npz_files:
            logger.warning(f"  No files for h={h}")
            continue

        # Equal-weight benchmark from first file
        data = np.load(npz_files[0], allow_pickle=True)
        dates = data["dates"]
        actuals = data["actuals"]
        n = len(data["assets"])
        ew_weights = np.ones(n) / n
        ew_vars = [float(ew_weights @ a @ ew_weights) for a in actuals]
        ew_result = {
            "model": "1/N",
            "horizon": h,
            "avg_realized_var": np.mean(ew_vars),
            "std_realized_var": np.std(ew_vars),
            "avg_turnover": 0.0,
            "avg_max_weight": 1.0 / n,
        }
        all_results.append(ew_result)
        logger.info(f"  h={h} 1/N: avg_var={ew_result['avg_realized_var']:.6e}")

        for npz_path in sorted(npz_files):
            model_name = npz_path.stem.replace(f"_h{h}", "")
            logger.info(f"  h={h} {model_name}:")

            try:
                perf = compute_portfolio_with_floor(npz_path, floor, logger)
                result = {
                    "model": model_name,
                    "horizon": h,
                    "avg_realized_var": perf["realized_var"].mean(),
                    "std_realized_var": perf["realized_var"].std(),
                    "avg_turnover": perf["turnover"].mean(),
                    "avg_max_weight": perf["max_weight"].mean(),
                }
                all_results.append(result)
                logger.info(f"    avg_var={result['avg_realized_var']:.6e}, "
                            f"turnover={result['avg_turnover']:.4f}")
            except Exception as e:
                logger.error(f"    FAILED: {e}")

    if all_results:
        summary = pd.DataFrame(all_results)
        out_path = tables_dir / f"portfolio_metrics_floor{floor_label}.csv"
        summary.to_csv(out_path, index=False)
        logger.info(f"\nSaved to {out_path}")

        # Print comparison
        logger.info("\n=== Results Summary ===")
        for h in horizons:
            sub = summary[summary["horizon"] == h]
            logger.info(f"\nh={h}:")
            for _, row in sub.iterrows():
                logger.info(f"  {row['model']:25s}  avg_var={row['avg_realized_var']:.6e}  "
                            f"turnover={row['avg_turnover']:.4f}")


if __name__ == "__main__":
    main()
