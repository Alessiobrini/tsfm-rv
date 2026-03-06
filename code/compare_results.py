"""
compare_results.py — Cross-dataset comparison of forecast results.

Reads evaluation metrics from both CAPIRe (results/metrics/) and VOLARE
(results/volare/metrics/) and produces a summary table showing whether
model rankings are stable across datasets.

Usage:
    python compare_results.py [--horizons 1 5 22] [--latex]
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR, VOLARE_RESULTS_DIR
from run_evaluation import generate_latex_table
from utils import setup_logger


CAPIRE_METRICS_DIR = RESULTS_DIR / "metrics"
VOLARE_METRICS_DIR = VOLARE_RESULTS_DIR / "metrics"


def load_aggregate_metrics(metrics_dir, dataset_name):
    """Load per-horizon metrics files and compute model averages."""
    results = []
    for h in [1, 5, 22]:
        fpath = metrics_dir / f"metrics_by_asset_h{h}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, index_col=0)
        # Average across tickers for each model
        numeric_cols = ['MSE', 'MAE', 'QLIKE', 'R2OOS']
        available_cols = [c for c in numeric_cols if c in df.columns]
        avg = df.groupby(df.index)[available_cols].mean()
        avg['horizon'] = h
        avg['dataset'] = dataset_name
        avg.index.name = 'model'
        results.append(avg.reset_index())

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Compare CAPIRe vs VOLARE results")
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 22],
                        help='Horizons to compare')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX comparison table')
    args = parser.parse_args()

    logger = setup_logger("compare_results")
    logger.info("=== Cross-Dataset Results Comparison ===")

    # Load metrics from both datasets
    capire_df = load_aggregate_metrics(CAPIRE_METRICS_DIR, 'CAPIRe')
    volare_df = load_aggregate_metrics(VOLARE_METRICS_DIR, 'VOLARE')

    if capire_df.empty:
        logger.error("No CAPIRe metrics found. Run run_evaluation.py first.")
        return
    if volare_df.empty:
        logger.error("No VOLARE metrics found. Run run_evaluation_volare.py first.")
        return

    combined = pd.concat([capire_df, volare_df], ignore_index=True)

    output_dir = RESULTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for h in args.horizons:
        logger.info(f"\n{'='*70}")
        logger.info(f"Horizon h={h}")
        logger.info(f"{'='*70}")

        h_data = combined[combined['horizon'] == h]
        if h_data.empty:
            logger.warning(f"No data for h={h}")
            continue

        # Pivot: model as rows, metrics by dataset as columns
        for metric in ['QLIKE', 'MSE', 'R2OOS']:
            if metric not in h_data.columns:
                continue

            pivot = h_data.pivot(index='model', columns='dataset', values=metric)
            if 'CAPIRe' in pivot.columns and 'VOLARE' in pivot.columns:
                pivot['diff'] = pivot['VOLARE'] - pivot['CAPIRe']
                # Rank within each dataset
                if metric == 'R2OOS':
                    pivot['rank_CAPIRe'] = pivot['CAPIRe'].rank(ascending=False).astype(int)
                    pivot['rank_VOLARE'] = pivot['VOLARE'].rank(ascending=False).astype(int)
                else:
                    pivot['rank_CAPIRe'] = pivot['CAPIRe'].rank(ascending=True).astype(int)
                    pivot['rank_VOLARE'] = pivot['VOLARE'].rank(ascending=True).astype(int)
                pivot['rank_change'] = pivot['rank_CAPIRe'] - pivot['rank_VOLARE']

            logger.info(f"\n  {metric} (h={h}):")
            logger.info(f"\n{pivot.round(4).to_string()}")

            pivot.round(6).to_csv(output_dir / f"comparison_{metric}_h{h}.csv")

        # Summary comparison table: QLIKE side by side
        qlike_data = h_data[h_data['horizon'] == h][['model', 'dataset', 'QLIKE', 'R2OOS']]
        summary = qlike_data.pivot(index='model', columns='dataset')
        summary.columns = [f'{metric}_{ds}' for metric, ds in summary.columns]
        summary = summary.sort_values(
            summary.columns[0] if len(summary.columns) > 0 else summary.index.name
        )
        summary.to_csv(output_dir / f"summary_h{h}.csv")

        if args.latex:
            latex = generate_latex_table(
                summary.round(4),
                caption=f"Cross-dataset comparison, $h={h}$",
                label=f"tab:comparison_h{h}",
            )
            latex_path = output_dir / f"comparison_h{h}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex)
            logger.info(f"  LaTeX table saved: {latex_path}")

    # Overall ranking stability
    logger.info(f"\n{'='*70}")
    logger.info("RANKING STABILITY SUMMARY")
    logger.info(f"{'='*70}")

    for h in args.horizons:
        h_data = combined[combined['horizon'] == h]
        if h_data.empty or 'QLIKE' not in h_data.columns:
            continue

        pivot = h_data.pivot(index='model', columns='dataset', values='QLIKE')
        if 'CAPIRe' not in pivot.columns or 'VOLARE' not in pivot.columns:
            continue

        rank_c = pivot['CAPIRe'].rank()
        rank_v = pivot['VOLARE'].rank()
        rank_corr = rank_c.corr(rank_v)

        top_c = pivot['CAPIRe'].idxmin()
        top_v = pivot['VOLARE'].idxmin()

        logger.info(f"\n  h={h}:")
        logger.info(f"    Best model (CAPIRe): {top_c} (QLIKE={pivot.loc[top_c, 'CAPIRe']:.4f})")
        logger.info(f"    Best model (VOLARE): {top_v} (QLIKE={pivot.loc[top_v, 'VOLARE']:.4f})")
        logger.info(f"    Rank correlation (Spearman): {rank_corr:.3f}")

        if top_c == top_v:
            logger.info("    => Same top model across datasets")
        else:
            logger.info("    => Different top models — dual presentation adds value")

    combined.to_csv(output_dir / "combined_metrics.csv", index=False)
    logger.info(f"\nAll comparison results saved to {output_dir}")


if __name__ == "__main__":
    main()
