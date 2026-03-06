"""
run_portfolio_eval.py — Evaluate GMV portfolios from covariance forecasts.

Reads forecasted covariance matrices from all models, constructs GMV portfolios,
and computes realized portfolio performance.

Usage:
    python run_portfolio_eval.py --asset-class forex
    python run_portfolio_eval.py --asset-class stocks --tc-bps 10
"""

import sys
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import COV_RESULTS_DIR, forecast_cfg
from data_loader import load_covariance_data
from evaluation.portfolio import (
    compute_portfolio_performance,
    compute_equal_weight_performance,
    summarize_portfolio_metrics,
)
from utils import setup_logger


def load_forecast_npz(filepath: Path) -> dict:
    """Load forecast matrices from npz file.

    Returns dict with 'dates', 'forecasts', 'actuals', 'assets'.
    """
    data = np.load(filepath, allow_pickle=True)
    dates = data['dates']
    forecasts = data['forecasts']
    actuals = data['actuals']
    assets = list(data['assets'])

    forecast_mats = {d: f for d, f in zip(dates, forecasts)}
    actual_mats = {d: a for d, a in zip(dates, actuals)}

    return {
        'forecast_mats': forecast_mats,
        'actual_mats': actual_mats,
        'assets': assets,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GMV portfolios from cov forecasts")
    parser.add_argument('--asset-class', default='forex',
                        choices=['stocks', 'forex', 'futures'])
    parser.add_argument('--horizons', nargs='+', type=int, default=None)
    parser.add_argument('--tc-bps', type=float, default=10.0,
                        help='Transaction costs in basis points (default: 10)')
    args = parser.parse_args()

    horizons = args.horizons or forecast_cfg.horizons

    logger = setup_logger("portfolio_eval")
    logger.info("=== GMV Portfolio Evaluation ===")
    logger.info(f"Asset class: {args.asset_class}, Horizons: {horizons}")
    logger.info(f"Transaction costs: {args.tc_bps} bps")

    forecast_dir = COV_RESULTS_DIR / args.asset_class / "forecasts"
    results_dir = COV_RESULTS_DIR / args.asset_class
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Find all forecast files
    all_results = []

    for horizon in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon h={horizon}")

        # Discover available forecast files for this horizon
        pattern = f"*_h{horizon}.npz"
        npz_files = list(forecast_dir.glob(pattern))

        if not npz_files:
            logger.warning(f"  No forecast files found for h={horizon}")
            continue

        # Load actual matrices from the first file (they're all the same)
        first_data = load_forecast_npz(npz_files[0])
        actual_mats = first_data['actual_mats']
        assets = first_data['assets']

        # Equal-weight benchmark
        ew_perf = compute_equal_weight_performance(actual_mats, assets)
        ew_summary = summarize_portfolio_metrics(ew_perf)
        ew_summary['model'] = '1/N'
        ew_summary['horizon'] = horizon
        all_results.append(ew_summary)
        logger.info(f"  1/N: avg_var={ew_summary['avg_realized_var']:.8f}")

        # Each model
        for npz_path in sorted(npz_files):
            model_name = npz_path.stem.replace(f'_h{horizon}', '')
            logger.info(f"  Model: {model_name}")

            try:
                data = load_forecast_npz(npz_path)
                perf = compute_portfolio_performance(
                    forecast_matrices=data['forecast_mats'],
                    actual_matrices=data['actual_mats'],
                    assets=data['assets'],
                    tc_bps=args.tc_bps,
                )

                if len(perf) == 0:
                    logger.warning(f"    No OOS dates for {model_name}")
                    continue

                summary = summarize_portfolio_metrics(perf)
                summary['model'] = model_name
                summary['horizon'] = horizon
                all_results.append(summary)

                logger.info(f"    avg_var={summary['avg_realized_var']:.8f}, "
                            f"avg_turnover={summary['avg_turnover']:.4f}")

                # Save daily performance
                perf.to_csv(results_dir / f"portfolio_daily_{model_name}_h{horizon}.csv",
                            index=False)

            except Exception as e:
                logger.error(f"    FAILED: {e}")
                continue

    # Summary table
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(tables_dir / "portfolio_metrics.csv", index=False)

        logger.info("\n=== PORTFOLIO SUMMARY ===")
        for h in horizons:
            sub = summary_df[summary_df['horizon'] == h]
            if len(sub) == 0:
                continue
            cols = ['model', 'avg_realized_var', 'avg_turnover']
            if 'sharpe' in sub.columns:
                cols.append('sharpe')
            logger.info(f"\nHorizon h={h}:")
            logger.info(f"\n{sub[cols].to_string(index=False)}")

    logger.info("\nPortfolio evaluation complete.")


if __name__ == "__main__":
    main()
