"""
run_evaluation_volare.py — Evaluate VOLARE forecasts: metrics, DM tests, MCS.

Mirrors run_evaluation.py but reads from results/volare/forecasts/
and saves to results/volare/metrics/ and results/volare/tables/.

Usage:
    python run_evaluation_volare.py [--horizons 1 5 22] [--alpha 0.10] [--latex]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import forecast_cfg, eval_cfg, VOLARE_RESULTS_DIR, FIGURES_DIR
from evaluation.loss_functions import compute_all_losses, compute_loss_series
from evaluation.dm_test import dm_test_matrix
from evaluation.mcs import model_confidence_set
from utils import setup_logger, save_metrics

# Import shared helpers from run_evaluation
from run_evaluation import (
    parse_forecast_filename, align_forecasts,
    compute_metrics_for_group, generate_latex_table,
)


FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
METRICS_DIR = VOLARE_RESULTS_DIR / "metrics"


def load_all_forecasts():
    """Scan VOLARE forecast directory and load all forecast CSVs."""
    if not FORECAST_DIR.exists():
        raise FileNotFoundError(f"VOLARE forecast directory not found: {FORECAST_DIR}")

    csv_files = list(FORECAST_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No forecast CSVs found in {FORECAST_DIR}")

    groups = defaultdict(dict)

    for fpath in csv_files:
        model_name, ticker, horizon = parse_forecast_filename(fpath)
        if model_name is None:
            continue

        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if 'actual' not in df.columns or 'forecast' not in df.columns:
            continue

        df = df.dropna(subset=['actual', 'forecast'])
        groups[(ticker, horizon)][model_name] = df

    return groups


def main():
    parser = argparse.ArgumentParser(description="Evaluate VOLARE model forecasts")
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Horizons to evaluate (default: all found)')
    parser.add_argument('--alpha', type=float, default=eval_cfg.mcs_alpha,
                        help='MCS significance level')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    parser.add_argument('--mcs-bootstrap', type=int, default=eval_cfg.mcs_n_bootstrap,
                        help='Number of MCS bootstrap replications')
    args = parser.parse_args()

    logger = setup_logger("evaluation_volare")
    logger.info("=== VOLARE Dataset — Forecast Evaluation ===")
    logger.info("Loading VOLARE forecast CSVs...")

    try:
        groups = load_all_forecasts()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Found {len(groups)} (ticker, horizon) groups")

    all_horizons = sorted(set(h for _, h in groups.keys()))
    horizons = args.horizons or all_horizons
    logger.info(f"Evaluating horizons: {horizons}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_metrics = []
    all_mcs_results = []

    for h in horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon h={h}")
        logger.info(f"{'='*60}")

        h_metrics = []
        h_mcs = []

        h_groups = {k: v for k, v in groups.items() if k[1] == h}

        if len(h_groups) == 0:
            logger.warning(f"No forecasts found for h={h}")
            continue

        for (ticker, horizon), model_dfs in sorted(h_groups.items()):
            n_models = len(model_dfs)
            logger.info(f"  {ticker}: {n_models} models")

            actual, forecasts = align_forecasts(model_dfs)
            if actual is None or len(actual) == 0:
                logger.warning(f"  {ticker}: no common dates, skipping")
                continue

            logger.info(f"    Common OOS dates: {len(actual)} obs")

            metrics_df, dm_pvals, mcs_result = compute_metrics_for_group(
                actual, forecasts, horizon
            )

            metrics_df['ticker'] = ticker
            metrics_df['horizon'] = h
            metrics_df['n_obs'] = len(actual)
            h_metrics.append(metrics_df)

            for model_name in metrics_df.index:
                row = metrics_df.loc[model_name]
                logger.info(
                    f"    {model_name:20s}: QLIKE={row['QLIKE']:.4f}  "
                    f"R2={row['R2OOS']:.3f}  MSE={row['MSE']:.6f}"
                )

            if mcs_result is not None:
                logger.info(f"    MCS surviving: {mcs_result.surviving_models}")
                for model_name in forecasts.keys():
                    in_mcs = 1 if model_name in mcs_result.surviving_models else 0
                    h_mcs.append({
                        'ticker': ticker,
                        'horizon': h,
                        'model': model_name,
                        'in_mcs': in_mcs,
                        'mcs_pvalue': mcs_result.p_values.get(model_name, np.nan),
                    })

            if dm_pvals is not None:
                dm_path = METRICS_DIR / f"dm_pvalues_{ticker}_h{h}.csv"
                dm_pvals.to_csv(dm_path)

        if h_metrics:
            all_h = pd.concat(h_metrics)
            all_h.to_csv(METRICS_DIR / f"metrics_by_asset_h{h}.csv")

            numeric_cols = ['MSE', 'MAE', 'QLIKE', 'R2OOS']
            avg = all_h.groupby(all_h.index)[numeric_cols].mean()
            avg['horizon'] = h

            logger.info(f"\n  Average metrics across assets (h={h}):")
            logger.info(f"\n{avg[numeric_cols].round(4).to_string()}")

            aggregate_metrics.append(avg)

            if args.latex:
                latex = generate_latex_table(
                    avg[numeric_cols].round(4),
                    caption=f"VOLARE forecast comparison, $h={h}$",
                    label=f"tab:volare_metrics_h{h}",
                )
                latex_path = VOLARE_RESULTS_DIR / "tables" / f"metrics_h{h}.tex"
                latex_path.parent.mkdir(parents=True, exist_ok=True)
                with open(latex_path, 'w') as f:
                    f.write(latex)
                logger.info(f"  LaTeX table saved: {latex_path}")

        if h_mcs:
            mcs_df = pd.DataFrame(h_mcs)
            mcs_summary = mcs_df.groupby('model').agg(
                pct_in_mcs=('in_mcs', 'mean'),
                avg_pvalue=('mcs_pvalue', 'mean'),
                n_assets=('in_mcs', 'count'),
            ).round(3)
            mcs_summary.to_csv(METRICS_DIR / f"mcs_results_h{h}.csv")
            all_mcs_results.append(mcs_df)

            logger.info(f"\n  MCS inclusion rate (h={h}):")
            logger.info(f"\n{mcs_summary.to_string()}")

    if aggregate_metrics:
        agg = pd.concat(aggregate_metrics)
        agg.to_csv(METRICS_DIR / "aggregate_metrics.csv")
        logger.info(f"\nAggregate metrics saved to {METRICS_DIR / 'aggregate_metrics.csv'}")

    if all_mcs_results:
        all_mcs = pd.concat(all_mcs_results)
        all_mcs.to_csv(METRICS_DIR / "mcs_all_results.csv", index=False)

    logger.info("\nVOLARE evaluation complete.")


if __name__ == "__main__":
    main()
