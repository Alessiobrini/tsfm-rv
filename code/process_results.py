"""
process_results.py — Generate publication-quality LaTeX tables from forecast results.

Reads CSV results from results/volare/metrics/ and results/metrics/ (CAPIRe),
produces LaTeX table files in paper/tables/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOLARE_METRICS = PROJECT_ROOT / "results" / "volare" / "metrics"
CAPIRE_METRICS = PROJECT_ROOT / "results" / "metrics"
COV_RESULTS = PROJECT_ROOT / "results" / "covariance"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Add code dir
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS

# Display names for models (order matters for table presentation)
MODEL_ORDER = [
    "HAR", "HAR_J", "HAR_RS", "HARQ", "Log_HAR", "ARFIMA",
    "chronos_bolt_small", "chronos_bolt_base", "moirai_2_0_small",
    "lag_llama", "timesfm_2_5", "toto", "sundial", "moirai_moe_small",
]
MODEL_DISPLAY = {
    "HAR": "HAR",
    "HAR_J": "HAR-J",
    "HAR_RS": "HAR-RS",
    "HARQ": "HARQ",
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S",
    "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto",
    "sundial": "Sundial",
    "moirai_moe_small": "Moirai-MoE-S",
}

HORIZONS = [1, 5, 22]


def bold_best(series, higher_better=False):
    """Return list of formatted strings with best value bolded."""
    if higher_better:
        best_idx = series.idxmax()
    else:
        best_idx = series.idxmin()
    out = []
    for idx, val in series.items():
        s = f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}"
        if idx == best_idx:
            s = f"\\textbf{{{s}}}"
        out.append(s)
    return out


def format_metric(values, metric, higher_better=False):
    """Format a series of metric values with appropriate precision and bolding."""
    best_idx = values.idxmax() if higher_better else values.idxmin()
    out = []
    for idx, val in values.items():
        if metric == "MSE":
            s = f"{val:.3f}"
        elif metric == "MAE":
            s = f"{val:.2f}"
        elif metric == "QLIKE":
            if val > 1.0:
                s = f"{val:.2f}$^{{\\dagger}}$"
            else:
                s = f"{val:.3f}"
        elif metric == "R2OOS":
            s = f"{val:.3f}"
        else:
            s = f"{val:.3f}"
        if idx == best_idx and not (metric == "QLIKE" and val > 1.0):
            s = f"\\textbf{{{s}}}"
        out.append(s)
    return out


def compute_asset_class_metrics(per_asset_dfs, tickers, agg="mean"):
    """Compute cross-sectional aggregate metrics for a given set of tickers.

    Parameters
    ----------
    agg : str
        "mean" or "median" for the cross-sectional aggregation.
    """
    rows = []
    for h in HORIZONS:
        df = per_asset_dfs[h]
        sub = df[df["ticker"].isin(tickers)].copy()
        if agg == "median":
            avg = sub.groupby("model")[["MSE", "MAE", "QLIKE", "R2OOS"]].median()
        else:
            avg = sub.groupby("model")[["MSE", "MAE", "QLIKE", "R2OOS"]].mean()
        avg["horizon"] = h
        avg = avg.reset_index()
        rows.append(avg)
    return pd.concat(rows, ignore_index=True)


def make_forecast_table(avg_df, caption, label, n_assets, mse_scale="1e6",
                        mae_scale="1e4", note=""):
    """Generate a LaTeX table with panels for each horizon."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")

    mse_header = "MSE"
    mae_header = "MAE"
    if mse_scale == "1e6":
        mse_header = "MSE ($\\times 10^{-6}$)"
    elif mse_scale == "1e8":
        mse_header = "MSE ($\\times 10^{-8}$)"
    if mae_scale == "1e4":
        mae_header = "MAE ($\\times 10^{-4}$)"
    elif mae_scale == "1e3":
        mae_header = "MAE ($\\times 10^{-3}$)"

    # For "none" scale, use raw headers
    if mse_scale == "none":
        mse_header = "MSE"
    if mae_scale == "none":
        mae_header = "MAE"

    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(f"Model & {mse_header} & {mae_header} & QLIKE & $R^2_{{\\text{{OOS}}}}$ \\\\")
    lines.append("\\midrule")

    for h in HORIZONS:
        sub = avg_df[avg_df["horizon"] == h].copy()
        # Reindex to MODEL_ORDER
        sub = sub.set_index("model")
        available_models = [m for m in MODEL_ORDER if m in sub.index]
        sub = sub.reindex(available_models)

        # Scale MSE and MAE
        mse_vals = sub["MSE"].copy()
        mae_vals = sub["MAE"].copy()
        if mse_scale == "1e6":
            mse_vals = mse_vals * 1e6
        elif mse_scale == "1e8":
            mse_vals = mse_vals * 1e8
        # "none" means no scaling
        if mae_scale == "1e4":
            mae_vals = mae_vals * 1e4
        elif mae_scale == "1e3":
            mae_vals = mae_vals * 1e3

        horizon_label = {1: "1 day", 5: "5 days", 22: "22 days"}[h]
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{Panel: $h = {h}$ ({horizon_label})}}}} \\\\[3pt]")

        # Determine best
        mse_best = mse_vals.idxmin()
        mae_best = mae_vals.idxmin()
        # For QLIKE best, only consider non-inflated values
        qlike_vals = sub["QLIKE"]
        qlike_valid = qlike_vals[qlike_vals < 1.0]
        qlike_best = qlike_valid.idxmin() if len(qlike_valid) > 0 else None
        r2_best = sub["R2OOS"].idxmax()

        for model in sub.index:
            name = MODEL_DISPLAY.get(model, model)
            mse_s = f"{mse_vals[model]:.3f}"
            mae_s = f"{mae_vals[model]:.2f}"
            qlike_v = sub.loc[model, "QLIKE"]
            r2_v = sub.loc[model, "R2OOS"]

            if qlike_v > 1.0:
                qlike_s = f"{qlike_v:.2f}$^{{\\dagger}}$"
            else:
                qlike_s = f"{qlike_v:.3f}"

            r2_s = f"{r2_v:.3f}"

            # Bold best
            if model == mse_best:
                mse_s = f"\\textbf{{{mse_s}}}"
            if model == mae_best:
                mae_s = f"\\textbf{{{mae_s}}}"
            if model == qlike_best:
                qlike_s = f"\\textbf{{{qlike_s}}}"
            if model == r2_best:
                r2_s = f"\\textbf{{{r2_s}}}"

            lines.append(f"{name} & {mse_s} & {mae_s} & {qlike_s} & {r2_s} \\\\")

        if h != 22:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append(f"\\\\[6pt]")
        lines.append(f"\\parbox{{\\textwidth}}{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_mcs_table(mcs_df, tickers, caption, label):
    """Generate MCS inclusion rate table."""
    sub = mcs_df[mcs_df["ticker"].isin(tickers)].copy()
    n_tickers = len(tickers)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Model & $h=1$ (\\%) & $h=5$ (\\%) & $h=22$ (\\%) \\\\")
    lines.append("\\midrule")

    for model in MODEL_ORDER:
        # Skip models not present in MCS data (avoids phantom 0% rows)
        if model not in sub["model"].values:
            continue
        name = MODEL_DISPLAY.get(model, model)
        rates = []
        for h in HORIZONS:
            mask = (sub["model"] == model) & (sub["horizon"] == h)
            matched = sub[mask]
            if len(matched) > 0:
                rate = matched["in_mcs"].mean() * 100
            else:
                rate = 0.0
            rates.append(rate)

        # Bold highest rate per column is not standard; just format
        lines.append(f"{name} & {rates[0]:.1f} & {rates[1]:.1f} & {rates[2]:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_dm_summary_table(dm_dir, tickers, caption, label):
    """Generate DM test summary: for each model pair, count wins at 5%."""
    models_in_order = [m for m in MODEL_ORDER]
    # Collect pairwise wins across all tickers and horizons
    # Format: for each (row_model, col_model), count how many tickers
    # row_model significantly beats col_model (p < 0.05)
    results = {}
    for h in HORIZONS:
        wins = {}
        for m in models_in_order:
            wins[m] = {}
            for m2 in models_in_order:
                wins[m][m2] = 0

        # Load per-asset QLIKE to determine direction of DM significance
        metrics_path = dm_dir / f"metrics_by_asset_h{h}.csv"
        metrics_df = pd.read_csv(metrics_path)
        # Pre-index as dict for fast lookup: {(ticker, model): qlike}
        qlike_lookup = {}
        for _, row in metrics_df.iterrows():
            qlike_lookup[(row["ticker"], row["model"])] = row["QLIKE"]

        for ticker in tickers:
            fpath = dm_dir / f"dm_pvalues_{ticker}_h{h}.csv"
            if not fpath.exists():
                continue
            dm = pd.read_csv(fpath, index_col=0)
            for m_row in dm.index:
                for m_col in dm.columns:
                    if m_row == m_col:
                        continue
                    pval = dm.loc[m_row, m_col]
                    # Two-sided DM test: p < 0.05 means losses differ
                    # significantly. Credit the win to whichever model
                    # has lower QLIKE for this ticker.
                    if pval < 0.05 and m_row in wins and m_col in wins[m_row]:
                        row_qlike = qlike_lookup.get((ticker, m_row))
                        col_qlike = qlike_lookup.get((ticker, m_col))
                        if row_qlike is not None and col_qlike is not None and row_qlike < col_qlike:
                            wins[m_row][m_col] += 1

        results[h] = wins

    # Create condensed table: for each model, count how many pairwise
    # comparisons it wins (summed across tickers)
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")

    n = len(tickers)
    # Table: Model | h=1 wins | h=5 wins | h=22 wins
    # "wins" = number of (ticker, opponent) pairs where this model has
    # significantly lower QLIKE
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append(f"Model & $h=1$ & $h=5$ & $h=22$ \\\\")
    lines.append("\\midrule")

    for model in models_in_order:
        name = MODEL_DISPLAY.get(model, model)
        vals = []
        for h in HORIZONS:
            # Total pairwise wins for this model
            total_wins = sum(results[h].get(model, {}).get(m2, 0)
                            for m2 in models_in_order if m2 != model)
            max_possible = (len(models_in_order) - 1) * n
            pct = total_wins / max_possible * 100 if max_possible > 0 else 0
            vals.append(f"{pct:.1f}\\%")
        lines.append(f"{name} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    n_opponents = len(models_in_order) - 1
    note = (f"Each cell reports the percentage of pairwise DM tests (across "
            f"{n} assets $\\times$ {n_opponents} opponents = {n_opponents*n} tests) "
            f"in which the row model has significantly lower QLIKE at the 5\\% level.")
    lines.append(f"\\\\[6pt]")
    lines.append(f"\\parbox{{\\textwidth}}{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_portfolio_table(forex_path, futures_path, caption, label):
    """Generate portfolio performance table from covariance results."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Model & $h$ & Avg RV ($\\times 10^{-6}$) & Std RV ($\\times 10^{-6}$) & Turnover & Max Wt \\\\")
    lines.append("\\midrule")

    portfolio_model_display = {
        "1/N": "1/$N$",
        "chronos_bolt_small": "Chronos-Bolt-S",
        "moirai_2_0_small": "Moirai-2.0-S",
        "element_har": "Element-HAR",
        "har_drd": "HAR-DRD",
        "timesfm_2_5": "TimesFM-2.5",
        "toto": "Toto",
        "sundial": "Sundial",
        "moirai_moe_small": "Moirai-MoE-S",
    }
    portfolio_model_order = ["1/N", "element_har", "har_drd", "chronos_bolt_small", "moirai_2_0_small", "timesfm_2_5", "toto", "sundial", "moirai_moe_small"]

    for panel_label, fpath in [("Forex (5 assets)", forex_path),
                               ("Futures (5 contracts)", futures_path)]:
        if not fpath.exists():
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{panel_label}: results pending}}}} \\\\")
            lines.append("\\addlinespace")
            continue

        df = pd.read_csv(fpath)
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{panel_label}}}}} \\\\[3pt]")

        for h in HORIZONS:
            sub = df[df["horizon"] == h].copy()
            # Reorder models
            available = [m for m in portfolio_model_order if m in sub["model"].values]
            sub = sub.set_index("model").reindex(available)

            # Find best (lowest) avg realized var among non-1/N models
            non_naive = sub.drop("1/N", errors="ignore")
            if len(non_naive) > 0:
                best_model = non_naive["avg_realized_var"].idxmin()
            else:
                best_model = None

            for model in available:
                row = sub.loc[model]
                name = portfolio_model_display.get(model, model)
                rv = row["avg_realized_var"] * 1e6
                rv_std = row["std_realized_var"] * 1e6
                turnover = row["avg_turnover"]
                max_wt = row["avg_max_weight"]

                rv_s = f"{rv:.2f}"
                if model == best_model:
                    rv_s = f"\\textbf{{{rv_s}}}"

                if pd.isna(turnover) or turnover == 0:
                    turn_s = "---"
                else:
                    turn_s = f"{turnover:.3f}"

                if pd.isna(max_wt):
                    wt_s = "---"
                else:
                    wt_s = f"{max_wt:.3f}"

                h_s = str(h) if model == available[0] else ""
                lines.append(f"{name} & {h_s} & {rv_s} & {rv_std:.2f} & {turn_s} & {wt_s} \\\\")

            if h != 22:
                lines.append("\\addlinespace")

        lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def make_cov_metrics_table(metrics_dir, caption, label):
    """Generate covariance forecast accuracy table."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Model & $h$ & Avg Frobenius ($\\times 10^{-5}$) & Diag QLIKE \\\\")
    lines.append("\\midrule")

    model_names = {"element-har": "Element-HAR", "har-drd": "HAR-DRD"}

    for h in HORIZONS:
        first = True
        for model_key in ["element_har", "har_drd"]:
            fpath = metrics_dir / f"cov_metrics_{model_key}_h{h}.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            row = df.iloc[0]
            name = model_names.get(row["model"], row["model"])
            frob = row["avg_frobenius"] * 1e5
            qlike = row["avg_diag_qlike"]
            h_s = str(h) if first else ""
            lines.append(f"{name} & {h_s} & {frob:.3f} & {qlike:.3f} \\\\")
            first = False
        if h != 22:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    print("Loading data...")

    # Load per-asset metrics for VOLARE
    per_asset = {}
    for h in HORIZONS:
        per_asset[h] = pd.read_csv(VOLARE_METRICS / f"metrics_by_asset_h{h}.csv")

    # Load MCS results
    mcs_volare = pd.read_csv(VOLARE_METRICS / "mcs_all_results.csv")
    # mcs_capire = pd.read_csv(CAPIRE_METRICS / "mcs_all_results.csv")

    # Load aggregate CAPIRe (commented out — CAPIRe removed from paper)
    # agg_capire = pd.read_csv(CAPIRE_METRICS / "aggregate_metrics.csv")

    # ================================================================
    # Table 2: Equity metrics (40 stocks)
    # ================================================================
    print("Generating Table 2: Equity forecast accuracy...")
    eq_metrics = compute_asset_class_metrics(per_asset, VOLARE_STOCK_TICKERS)
    eq_metrics = eq_metrics.reset_index()
    table2 = make_forecast_table(
        eq_metrics,
        caption=(
            "Forecast accuracy for 40 U.S.\\ equities (VOLARE). "
            "Values are cross-sectional averages of per-asset loss functions. "
            "Bold indicates the best value in each column within each panel. "
            "$\\dagger$ denotes inflated QLIKE due to near-zero forecasts or systematic forecast bias."
        ),
        label="tab:main_results",
        n_assets=40,
        mse_scale="1e6",
        mae_scale="1e4",
        note="$\\dagger$ Marks QLIKE $> 1$. For HAR-RS and HARQ, this reflects near-zero forecasts "
             "from levels-based OLS.",
    )
    (TABLE_DIR / "table_equity_metrics.tex").write_text(table2)

    # ================================================================
    # Table 2b: Equity metrics (MEDIAN)
    # ================================================================
    print("Generating Table 2b: Equity forecast accuracy (median)...")
    eq_metrics_med = compute_asset_class_metrics(per_asset, VOLARE_STOCK_TICKERS, agg="median")
    eq_metrics_med = eq_metrics_med.reset_index()
    table2b = make_forecast_table(
        eq_metrics_med,
        caption=(
            "Forecast accuracy for 40 U.S.\\ equities (VOLARE), cross-sectional medians. "
            "Median aggregation is robust to outlier assets with degenerate forecasts. "
            "Bold indicates the best value in each column within each panel."
        ),
        label="tab:main_results_median",
        n_assets=40,
        mse_scale="1e6",
        mae_scale="1e4",
    )
    (TABLE_DIR / "table_equity_metrics_median.tex").write_text(table2b)

    # ================================================================
    # Table 3: Equity MCS inclusion rates
    # ================================================================
    print("Generating Table 3: Equity MCS inclusion rates...")
    table3 = make_mcs_table(
        mcs_volare,
        VOLARE_STOCK_TICKERS,
        caption=(
            "Model Confidence Set inclusion rates for 40 U.S.\\ equities (VOLARE). "
            "Each cell reports the percentage of stocks for which the model is "
            "included in the MCS at the 10\\% significance level (QLIKE loss, "
            "$T_{\\max}$ statistic, block bootstrap with $B = 10{,}000$)."
        ),
        label="tab:mcs",
    )
    (TABLE_DIR / "table_equity_mcs.tex").write_text(table3)

    # ================================================================
    # Table 4: FX results
    # ================================================================
    print("Generating Table 4: FX forecast accuracy...")
    fx_metrics = compute_asset_class_metrics(per_asset, VOLARE_FX_TICKERS)
    fx_metrics = fx_metrics.reset_index()
    table4 = make_forecast_table(
        fx_metrics,
        caption=(
            "Forecast accuracy for 5 FX pairs (VOLARE). "
            "Values are averages across the five currency pairs. "
            "Format as Table~\\ref{tab:main_results}."
        ),
        label="tab:fx_results",
        n_assets=5,
        mse_scale="1e8",
        mae_scale="1e4",
    )
    (TABLE_DIR / "table_fx_metrics.tex").write_text(table4)

    # ================================================================
    # Table 5: Futures results
    # ================================================================
    print("Generating Table 5: Futures forecast accuracy...")
    fut_metrics = compute_asset_class_metrics(per_asset, VOLARE_FUTURES_TICKERS)
    fut_metrics = fut_metrics.reset_index()
    table5 = make_forecast_table(
        fut_metrics,
        caption=(
            "Forecast accuracy for 5 futures contracts (VOLARE). "
            "Values are averages across the five contracts. "
            "Format as Table~\\ref{tab:main_results}."
        ),
        label="tab:futures_results",
        n_assets=5,
        mse_scale="1e6",
        mae_scale="1e4",
    )
    (TABLE_DIR / "table_futures_metrics.tex").write_text(table5)

    # ================================================================
    # Table 6: DM test summary
    # ================================================================
    print("Generating Table 6: DM test summary...")
    all_tickers = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
    table6 = make_dm_summary_table(
        VOLARE_METRICS,
        all_tickers,
        caption=(
            "Diebold--Mariano test: pairwise win rates. "
            "Each cell reports the percentage of pairwise comparisons "
            f"(across {len(all_tickers)} assets $\\times$ {len(MODEL_ORDER)-1} opponents) "
            "in which the row model "
            "achieves significantly lower QLIKE at the 5\\% level."
        ),
        label="tab:dm_summary",
    )
    (TABLE_DIR / "table_dm_summary.tex").write_text(table6)

    # ================================================================
    # Table A1: CAPIRe aggregate metrics (commented out — CAPIRe removed from paper)
    # ================================================================
    # print("Generating Table A1: CAPIRe aggregate metrics...")
    # agg_capire_fmt = agg_capire.copy()
    # agg_capire_fmt = agg_capire_fmt.rename(columns={"model": "model"})
    # table_a1 = make_forecast_table(
    #     agg_capire_fmt,
    #     caption=(
    #         "Forecast accuracy for 29 DJIA stocks (CAPIRe dataset). "
    #         "Values are cross-sectional averages. RV is in annualized \\% units. "
    #         "Bold indicates best value per column per panel. "
    #         "$\\dagger$ denotes inflated QLIKE."
    #     ),
    #     label="tab:capire_results",
    #     n_assets=29,
    #     mse_scale="none",
    #     mae_scale="none",
    # )
    # (TABLE_DIR / "table_capire_metrics.tex").write_text(table_a1)

    # ================================================================
    # Table A2: CAPIRe MCS inclusion rates (commented out — CAPIRe removed from paper)
    # ================================================================
    # print("Generating Table A2: CAPIRe MCS inclusion rates...")
    # capire_tickers = [t for t in
    #     ['AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    #      'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD',
    #      'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT']
    # ]
    # table_a2 = make_mcs_table(
    #     mcs_capire,
    #     capire_tickers,
    #     caption=(
    #         "Model Confidence Set inclusion rates for 29 DJIA stocks (CAPIRe dataset). "
    #         "Format as Table~\\ref{tab:mcs}."
    #     ),
    #     label="tab:capire_mcs",
    # )
    # (TABLE_DIR / "table_capire_mcs.tex").write_text(table_a2)

    # ================================================================
    # Table 7 & 8: Portfolio and covariance accuracy
    # These tables are manually maintained in paper/tables/ because they
    # include multi-asset-class data (stocks panel) not in the CSVs.
    # ================================================================
    print("Skipping Table 7 (portfolio) and Table 8 (cov accuracy) — manually maintained.")

    # ================================================================
    # Print summary statistics for paper text
    # ================================================================
    print("\n" + "=" * 60)
    print("KEY NUMBERS FOR PAPER TEXT")
    print("=" * 60)

    print("\n--- VOLARE Equity averages (40 stocks) ---")
    for h in HORIZONS:
        sub = eq_metrics[eq_metrics["horizon"] == h].set_index("model")
        print(f"\nh={h}:")
        for m in MODEL_ORDER:
            if m in sub.index:
                row = sub.loc[m]
                print(f"  {MODEL_DISPLAY[m]:20s}  MSE={row['MSE']:.2e}  "
                      f"MAE={row['MAE']:.2e}  QLIKE={row['QLIKE']:.3f}  "
                      f"R2OOS={row['R2OOS']:.3f}")

    print("\n--- VOLARE MCS inclusion rates (40 equities) ---")
    for h in HORIZONS:
        print(f"\nh={h}:")
        for m in MODEL_ORDER:
            mask = (mcs_volare["model"] == m) & (mcs_volare["horizon"] == h) & \
                   (mcs_volare["ticker"].isin(VOLARE_STOCK_TICKERS))
            matched = mcs_volare[mask]
            if len(matched) > 0:
                rate = matched["in_mcs"].mean() * 100
                print(f"  {MODEL_DISPLAY[m]:20s}  {rate:.1f}%")

    print("\n--- Forex portfolio: variance reduction vs 1/N ---")
    forex_port = COV_RESULTS / "forex" / "portfolio_metrics.csv"
    if forex_port.exists():
        fp = pd.read_csv(forex_port)
        for h in HORIZONS:
            naive = fp[(fp["model"] == "1/N") & (fp["horizon"] == h)]["avg_realized_var"].values[0]
            for m in ["chronos_bolt_small", "har_drd", "element_har", "moirai_2_0_small"]:
                row = fp[(fp["model"] == m) & (fp["horizon"] == h)]
                if len(row) > 0:
                    rv = row["avg_realized_var"].values[0]
                    reduction = (1 - rv / naive) * 100
                    print(f"  h={h} {m:25s}  reduction={reduction:.1f}%")

    print("\nDone! All tables written to", TABLE_DIR)


if __name__ == "__main__":
    main()
