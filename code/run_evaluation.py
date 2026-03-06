"""
run_evaluation.py — Unified evaluation: compare all models and produce paper tables.

Orchestrates:
    1. Scan results/forecasts/ for all forecast CSV files
    2. Group by ticker x horizon
    3. Align all models to common date range
    4. Compute MSE, MAE, QLIKE, R2_OOS for each model
    5. Run pairwise DM tests
    6. Run Model Confidence Set
    7. Aggregate across assets
    8. Save results and optionally generate LaTeX tables

Usage:
    python run_evaluation.py [--horizons 1 5 22] [--alpha 0.10] [--latex]
"""

import sys
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import forecast_cfg, eval_cfg, RESULTS_DIR, FIGURES_DIR
from evaluation.loss_functions import compute_all_losses, compute_loss_series
from evaluation.dm_test import dm_test_matrix
from evaluation.mcs import model_confidence_set
from utils import setup_logger, save_metrics


FORECAST_DIR = RESULTS_DIR / "forecasts"
METRICS_DIR = RESULTS_DIR / "metrics"


def parse_forecast_filename(filepath: Path):
    """Parse model name, ticker, and horizon from forecast CSV filename.

    Expected pattern: {model}_{ticker}_h{horizon}.csv
    Examples: HAR_AAPL_h1.csv, chronos_bolt_small_AAPL_h1.csv
    """
    stem = filepath.stem  # e.g., "HAR_AAPL_h1"

    # Match horizon at the end
    match = re.match(r'^(.+)_([A-Z]+)_h(\d+)$', stem)
    if match:
        model_name = match.group(1)
        ticker = match.group(2)
        horizon = int(match.group(3))
        return model_name, ticker, horizon
    return None, None, None


def load_all_forecasts():
    """Scan forecast directory and load all forecast CSVs.

    Returns
    -------
    dict
        Nested: {(ticker, horizon): {model_name: DataFrame with 'actual','forecast'}}
    """
    if not FORECAST_DIR.exists():
        raise FileNotFoundError(f"Forecast directory not found: {FORECAST_DIR}")

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

        # Drop any NaN rows
        df = df.dropna(subset=['actual', 'forecast'])

        groups[(ticker, horizon)][model_name] = df

    return groups


def align_forecasts(model_dfs):
    """Align all model forecasts to their common date range.

    Parameters
    ----------
    model_dfs : dict
        {model_name: DataFrame with 'actual','forecast' and DatetimeIndex}

    Returns
    -------
    common_actual : pd.Series
        Actual values over common dates.
    model_forecasts : dict
        {model_name: pd.Series of forecasts over common dates}
    """
    # Find common date intersection
    date_sets = [set(df.index) for df in model_dfs.values()]
    if len(date_sets) == 0:
        return None, {}

    common_dates = sorted(set.intersection(*date_sets))
    if len(common_dates) == 0:
        return None, {}

    common_idx = pd.DatetimeIndex(common_dates)

    # Use actual from first model (should be identical across all)
    first_df = next(iter(model_dfs.values()))
    common_actual = first_df.loc[common_idx, 'actual']

    model_forecasts = {}
    for model_name, df in model_dfs.items():
        model_forecasts[model_name] = df.loc[common_idx, 'forecast']

    return common_actual, model_forecasts


def compute_metrics_for_group(actual, forecasts, horizon):
    """Compute all metrics for a single ticker x horizon group.

    Returns
    -------
    metrics_df : pd.DataFrame
        Rows = models, columns = MSE, MAE, QLIKE, R2OOS.
    dm_pvals : pd.DataFrame
        Pairwise DM test p-values.
    mcs_result : MCSResult or None
    """
    metrics_rows = []
    loss_series = {}

    for model_name, fcast in forecasts.items():
        m = compute_all_losses(actual, fcast)
        m['model'] = model_name
        metrics_rows.append(m)

        # Compute element-wise QLIKE loss for DM test and MCS
        loss_series[model_name] = compute_loss_series(
            actual.values, fcast.values,
            loss_type=eval_cfg.primary_loss,
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index('model')

    # DM test matrix
    dm_pvals = None
    if len(loss_series) >= 2:
        dm_pvals = dm_test_matrix(
            loss_series,
            h=horizon,
            alternative=eval_cfg.dm_alternative,
        )

    # MCS
    mcs_result = None
    if len(loss_series) >= 2:
        try:
            mcs_result = model_confidence_set(
                loss_series,
                alpha=eval_cfg.mcs_alpha,
                n_bootstrap=eval_cfg.mcs_n_bootstrap,
                block_length=eval_cfg.mcs_block_length,
            )
        except Exception as e:
            print(f"  MCS failed: {e}")

    return metrics_df, dm_pvals, mcs_result


def generate_latex_table(df, caption, label):
    """Generate a LaTeX booktabs table from a DataFrame."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Column format
    n_cols = len(df.columns)
    col_fmt = "l" + "c" * n_cols
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join(["Model"] + [str(c) for c in df.columns]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Rows
    for idx, row in df.iterrows():
        vals = [str(idx)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare all models")
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Horizons to evaluate (default: all found)')
    parser.add_argument('--alpha', type=float, default=eval_cfg.mcs_alpha,
                        help='MCS significance level')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    parser.add_argument('--mcs-bootstrap', type=int, default=eval_cfg.mcs_n_bootstrap,
                        help='Number of MCS bootstrap replications')
    args = parser.parse_args()

    logger = setup_logger("evaluation")
    logger.info("Loading forecast CSVs...")

    # Load all forecasts
    try:
        groups = load_all_forecasts()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Found {len(groups)} (ticker, horizon) groups")

    # Determine which horizons to evaluate
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
        h_dm_pvals = []
        h_mcs = []

        # Get all groups for this horizon
        h_groups = {k: v for k, v in groups.items() if k[1] == h}

        if len(h_groups) == 0:
            logger.warning(f"No forecasts found for h={h}")
            continue

        for (ticker, horizon), model_dfs in sorted(h_groups.items()):
            n_models = len(model_dfs)
            logger.info(f"  {ticker}: {n_models} models")

            # Align to common dates
            actual, forecasts = align_forecasts(model_dfs)
            if actual is None or len(actual) == 0:
                logger.warning(f"  {ticker}: no common dates, skipping")
                continue

            logger.info(f"    Common OOS dates: {len(actual)} obs")

            # Compute metrics
            metrics_df, dm_pvals, mcs_result = compute_metrics_for_group(
                actual, forecasts, horizon
            )

            # Add ticker info
            metrics_df['ticker'] = ticker
            metrics_df['horizon'] = h
            metrics_df['n_obs'] = len(actual)
            h_metrics.append(metrics_df)

            # Log per-asset results
            for model_name in metrics_df.index:
                row = metrics_df.loc[model_name]
                logger.info(
                    f"    {model_name:20s}: QLIKE={row['QLIKE']:.4f}  "
                    f"R2={row['R2OOS']:.3f}  MSE={row['MSE']:.6f}"
                )

            # MCS results
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

            # Save DM p-values for this asset
            if dm_pvals is not None:
                dm_path = METRICS_DIR / f"dm_pvalues_{ticker}_h{h}.csv"
                dm_pvals.to_csv(dm_path)

        # Aggregate metrics across assets for this horizon
        if h_metrics:
            all_h = pd.concat(h_metrics)
            all_h.to_csv(METRICS_DIR / f"metrics_by_asset_h{h}.csv")

            # Average across assets
            numeric_cols = ['MSE', 'MAE', 'QLIKE', 'R2OOS']
            avg = all_h.groupby(all_h.index)[numeric_cols].mean()
            avg['horizon'] = h

            logger.info(f"\n  Average metrics across assets (h={h}):")
            logger.info(f"\n{avg[numeric_cols].round(4).to_string()}")

            aggregate_metrics.append(avg)

            # LaTeX table
            if args.latex:
                latex = generate_latex_table(
                    avg[numeric_cols].round(4),
                    caption=f"Forecast comparison, $h={h}$",
                    label=f"tab:metrics_h{h}",
                )
                latex_path = RESULTS_DIR / "tables" / f"metrics_h{h}.tex"
                latex_path.parent.mkdir(parents=True, exist_ok=True)
                with open(latex_path, 'w') as f:
                    f.write(latex)
                logger.info(f"  LaTeX table saved: {latex_path}")

        # MCS aggregate: % times in MCS across assets
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

    # Save aggregate metrics
    if aggregate_metrics:
        agg = pd.concat(aggregate_metrics)
        agg.to_csv(METRICS_DIR / "aggregate_metrics.csv")
        logger.info(f"\nAggregate metrics saved to {METRICS_DIR / 'aggregate_metrics.csv'}")

    if all_mcs_results:
        all_mcs = pd.concat(all_mcs_results)
        all_mcs.to_csv(METRICS_DIR / "mcs_all_results.csv", index=False)

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
