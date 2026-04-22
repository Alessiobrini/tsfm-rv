"""
run_evaluation.py — Helper functions for forecast evaluation.

Imported by run_evaluation_volare.py and run_advanced_evaluation.py.
Not a standalone entry point; run the pipeline through run_evaluation_volare.py.
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import eval_cfg
from evaluation.loss_functions import compute_all_losses, compute_loss_series
from evaluation.dm_test import dm_test_matrix
from evaluation.mcs import model_confidence_set


def parse_forecast_filename(filepath: Path):
    """Parse model name, ticker, and horizon from forecast CSV filename.

    Expected pattern: {model}_{ticker}_h{horizon}.csv
    Examples: HAR_AAPL_h1.csv, chronos_bolt_small_AAPL_h1.csv
    """
    stem = filepath.stem
    match = re.match(r'^(.+)_([A-Z]+)_h(\d+)$', stem)
    if match:
        model_name = match.group(1)
        ticker = match.group(2)
        horizon = int(match.group(3))
        return model_name, ticker, horizon
    return None, None, None


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
    date_sets = [set(df.index) for df in model_dfs.values()]
    if len(date_sets) == 0:
        return None, {}

    common_dates = sorted(set.intersection(*date_sets))
    if len(common_dates) == 0:
        return None, {}

    common_idx = pd.DatetimeIndex(common_dates)
    first_df = next(iter(model_dfs.values()))
    common_actual = first_df.loc[common_idx, 'actual']

    model_forecasts = {}
    for model_name, df in model_dfs.items():
        model_forecasts[model_name] = df.loc[common_idx, 'forecast']

    return common_actual, model_forecasts


def compute_metrics_for_group(actual, forecasts, horizon):
    """Compute MSE/MAE/QLIKE/R2OOS, DM test matrix, and MCS for one group."""
    metrics_rows = []
    loss_series = {}

    for model_name, fcast in forecasts.items():
        m = compute_all_losses(actual, fcast)
        m['model'] = model_name
        metrics_rows.append(m)

        loss_series[model_name] = compute_loss_series(
            actual.values, fcast.values,
            loss_type=eval_cfg.primary_loss,
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index('model')

    dm_pvals = None
    if len(loss_series) >= 2:
        dm_pvals = dm_test_matrix(
            loss_series,
            h=horizon,
            alternative=eval_cfg.dm_alternative,
        )

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

    n_cols = len(df.columns)
    col_fmt = "l" + "c" * n_cols
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    header = " & ".join(["Model"] + [str(c) for c in df.columns]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for idx, row in df.iterrows():
        vals = [str(idx)] + [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)
