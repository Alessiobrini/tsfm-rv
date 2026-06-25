"""
process_results.py — Generate publication-quality LaTeX tables from forecast results.

Reads CSV results from results/volare/metrics/ and writes LaTeX table files
to paper/tables/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOLARE_METRICS = PROJECT_ROOT / "results" / "volare" / "metrics"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Add code dir
sys.path.insert(0, str(PROJECT_ROOT / "code"))
from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS

# Display names for models (order matters for table presentation)
MODEL_ORDER = [
    "Log_HAR", "HAR", "HAR_J", "HAR_RS", "HARQ", "ARFIMA", "ARMA", "MEM",
    "chronos_bolt_small", "chronos_bolt_base", "moirai_2_0_small",
    "moirai_moe_small", "lag_llama", "timesfm_2_5", "toto", "sundial", "ttm",
]
MODEL_DISPLAY = {
    "Log_HAR": "Log-HAR",
    "HAR": "HAR",
    "HAR_J": "HAR-J",
    "HAR_RS": "HAR-RS",
    "HARQ": "HARQ",
    "ARFIMA": "ARFIMA",
    "ARMA": "ARMA",
    "MEM": "MEM",
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S",
    "moirai_moe_small": "Moirai-MoE-S",
    "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto",
    "sundial": "Sundial",
    "ttm": "TTM",
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


def _longtable(colspec, header, ncol, body_lines, caption, label, note="",
               font="\\small"):
    """Wrap a multi-row table body in a longtable so it breaks across pages.

    These metric tables stack three horizon blocks (and, for the combined
    FX+futures table, two asset-class panels) of 16 model rows each, which
    overflows a single ``table[H]`` page (the futures panel was clipped off the
    bottom). A longtable page-breaks automatically, repeats the column header on
    each continuation page, and keeps one caption/number/label so cross-references
    are unchanged. ``ncol`` is the number of columns spanned by header rows."""
    head = [
        f"{{\\singlespacing{font}",
        f"\\begin{{longtable}}{{{colspec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\toprule",
        f"{header} \\\\",
        "\\midrule",
        "\\endfirsthead",
        f"\\multicolumn{{{ncol}}}{{c}}{{\\tablename\\ \\thetable\\ -- continued from previous page}} \\\\",
        "\\toprule",
        f"{header} \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{ncol}}}{{r}}{{\\footnotesize\\textit{{continued on next page}}}} \\\\",
        "\\endfoot",
        "\\bottomrule",
    ]
    if note:
        head.append(
            f"\\multicolumn{{{ncol}}}{{l}}{{\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}}} \\\\")
    head.append("\\endlastfoot")
    return "\n".join(head + body_lines + ["\\end{longtable}", "}"])


def mcs_majority_by_h(tickers, horizons=None):
    """Per horizon, the set of models included in the Model Confidence Set (10%)
    for a strict majority of the given assets. Used to add a statistical-test
    marker to the cross-sectional-average metric tables (Referee 2 minor 1: the
    merged tables should carry test indicators, not just the QLIKE>1 dagger)."""
    horizons = horizons or HORIZONS
    try:
        mcs = pd.read_csv(VOLARE_METRICS / "mcs_all_results.csv")
    except FileNotFoundError:
        return {h: set() for h in horizons}
    mcs = mcs[mcs["ticker"].isin(list(tickers))]
    out = {}
    for h in horizons:
        sub = mcs[mcs["horizon"] == h]
        if sub.empty:
            out[h] = set(); continue
        frac = sub.groupby("model")["in_mcs"].mean()
        out[h] = set(frac[frac > 0.5].index)
    return out


def make_forecast_table(avg_df, caption, label, n_assets, mse_scale="1e6",
                        mae_scale="1e4", note="", mcs_by_h=None):
    """Generate a LaTeX table with panels for each horizon."""
    body = []
    mcs_by_h = mcs_by_h or {}

    mse_header = "MSE"
    if mse_scale == "1e6":
        mse_header = "MSE ($\\times 10^{-6}$)"
    elif mse_scale == "1e8":
        mse_header = "MSE ($\\times 10^{-8}$)"

    # For "none" scale, use raw headers
    if mse_scale == "none":
        mse_header = "MSE"

    header = f"Model & {mse_header} & QLIKE"

    for h in HORIZONS:
        sub = avg_df[avg_df["horizon"] == h].copy()
        # Reindex to MODEL_ORDER
        sub = sub.set_index("model")
        available_models = [m for m in MODEL_ORDER if m in sub.index]
        sub = sub.reindex(available_models)

        # Scale MSE
        mse_vals = sub["MSE"].copy()
        if mse_scale == "1e6":
            mse_vals = mse_vals * 1e6
        elif mse_scale == "1e8":
            mse_vals = mse_vals * 1e8
        # "none" means no scaling

        horizon_label = {1: "1 day", 5: "5 days", 22: "22 days"}[h]
        body.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Panel: $h = {h}$ ({horizon_label})}}}} \\\\[3pt]")

        # Determine best
        mse_best = mse_vals.idxmin()
        # For QLIKE best, only consider non-inflated values
        qlike_vals = sub["QLIKE"]
        qlike_valid = qlike_vals[qlike_vals < 1.0]
        qlike_best = qlike_valid.idxmin() if len(qlike_valid) > 0 else None

        for model in sub.index:
            name = MODEL_DISPLAY.get(model, model)
            mse_s = f"{mse_vals[model]:.3f}"
            qlike_v = sub.loc[model, "QLIKE"]

            if qlike_v > 1.0:
                qlike_s = f"{qlike_v:.2f}$^{{\\dagger}}$"
            else:
                qlike_s = f"{qlike_v:.3f}"

            # Bold best
            if model == mse_best:
                mse_s = f"\\textbf{{{mse_s}}}"
            if model == qlike_best:
                qlike_s = f"\\textbf{{{qlike_s}}}"
            if model in mcs_by_h.get(h, set()):
                qlike_s = f"{qlike_s}$^{{\\ast}}$"

            body.append(f"{name} & {mse_s} & {qlike_s} \\\\")

        if h != 22:
            body.append("\\addlinespace")

    return _longtable("lrr", header, 3, body, caption, label, note=note)


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


def make_loss_ratio_table(per_asset_dfs, tickers, caption, label, benchmark="Log_HAR"):
    """Average QLIKE loss ratio vs a benchmark across assets (Referee 2 sec. 3).

    For each asset i: ratio_i = QLIKE_{model,i} / QLIKE_{benchmark,i}; we report the
    mean ratio across assets. Averaging *ratios* is robust to outlier assets (a few
    high-loss assets cannot dominate, unlike a pooled/average loss). A value < 1 means
    the model has lower QLIKE than the benchmark on average; the benchmark row is 1.000.
    """
    table = {}  # model -> {h: mean ratio}
    for h in HORIZONS:
        sub = per_asset_dfs[h]
        sub = sub[sub["ticker"].isin(tickers)]
        piv = sub.pivot_table(index="ticker", columns="model", values="QLIKE")
        if benchmark not in piv.columns:
            continue
        ratios = piv.div(piv[benchmark], axis=0).mean(axis=0)
        for m in ratios.index:
            table.setdefault(m, {})[h] = float(ratios[m])

    best = {}
    for h in HORIZONS:
        col = {m: table[m][h] for m in table if h in table[m]}
        if col:
            best[h] = min(col, key=col.get)

    bname = MODEL_DISPLAY.get(benchmark, benchmark)
    lines = ["\\begin{table}[H]", "\\centering", "\\singlespacing",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Model & $h=1$ & $h=5$ & $h=22$ \\\\", "\\midrule"]
    for m in MODEL_ORDER:
        if m not in table:
            continue
        cells = []
        for h in HORIZONS:
            v = table[m].get(h, np.nan)
            s = f"{v:.3f}" if np.isfinite(v) else "--"
            if m == best.get(h):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"{MODEL_DISPLAY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\\\[6pt]",
              f"\\parbox{{\\textwidth}}{{\\footnotesize Mean across assets of the per-asset "
              f"QLIKE ratio to {bname} (which is 1.000 by construction). Values below 1 indicate "
              f"lower QLIKE than {bname} on average. Averaging loss ratios is robust to outlier "
              f"assets, unlike the pooled averages in Table~\\ref{{tab:pooled50}}.}}",
              "\\end{table}"]
    return "\n".join(lines)


def make_combined_metrics_table(specs, caption, label):
    """One table with several asset-class panels (Referee 2 minor 1: combine the
    separate FX and futures tables). `specs` is a list of
    (panel_label, avg_df, mse_scale, mae_scale, mcs_by_h), where mcs_by_h is the
    per-horizon set of models in the MCS for a majority of the panel's assets
    (may be omitted/empty)."""
    header = "Model & MSE & QLIKE"
    lines = []
    for pi, spec in enumerate(specs):
        panel_label, avg_df, mse_scale, mae_scale = spec[:4]
        mcs_by_h = spec[4] if len(spec) > 4 else {}
        if pi > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{panel_label}}}}} \\\\[2pt]")
        for h in HORIZONS:
            sub = avg_df[avg_df["horizon"] == h].set_index("model")
            sub = sub.reindex([m for m in MODEL_ORDER if m in sub.index])
            mse = sub["MSE"] * (1e6 if mse_scale == "1e6" else 1e8 if mse_scale == "1e8" else 1)
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{$h = {h}$}}}} \\\\[2pt]")
            mse_b = mse.idxmin()
            qv = sub["QLIKE"]; qval = qv[qv < 1.0]
            # Bold the lowest QLIKE and any model tied with it at displayed precision
            # (3 decimals), so e.g. a Sundial/TTM tie at 0.184 bolds both.
            qmin = qval.min() if len(qval) else None
            for m in sub.index:
                ms = f"{mse[m]:.3f}"
                qq = sub.loc[m, "QLIKE"]; qs = f"{qq:.2f}$^{{\\dagger}}$" if qq > 1.0 else f"{qq:.3f}"
                if m == mse_b: ms = f"\\textbf{{{ms}}}"
                if qmin is not None and qq < 1.0 and round(qq, 3) == round(qmin, 3):
                    qs = f"\\textbf{{{qs}}}"
                if m in mcs_by_h.get(h, set()):
                    qs = f"{qs}$^{{\\ast}}$"
                lines.append(f"{MODEL_DISPLAY.get(m, m)} & {ms} & {qs} \\\\")
            lines.append("\\addlinespace")
    return _longtable("lrr", header, 3, lines, caption, label)


def main():
    print("Loading data...")

    # Load per-asset metrics for VOLARE
    per_asset = {}
    for h in HORIZONS:
        per_asset[h] = pd.read_csv(VOLARE_METRICS / f"metrics_by_asset_h{h}.csv")

    # Load MCS results
    mcs_volare = pd.read_csv(VOLARE_METRICS / "mcs_all_results.csv")

    # MCS-majority membership per horizon, by asset class (statistical-test marker
    # for the metric tables; Referee 2 minor 1).
    mcs_stocks = mcs_majority_by_h(VOLARE_STOCK_TICKERS)
    mcs_fx = mcs_majority_by_h(VOLARE_FX_TICKERS)
    mcs_fut = mcs_majority_by_h(VOLARE_FUTURES_TICKERS)

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
            "MSE is on the volatility scale; QLIKE is on the variance scale. "
            "$\\dagger$ marks QLIKE $> 1$. $^{\\ast}$ marks models in the Model Confidence Set "
            "(10\\%) for a majority of the 40 equities at that horizon. Under non-negativity-constrained "
            "estimation of the level HAR variants and winsorization to the in-sample volatility support, "
            "no model produces near-zero forecasts, so no flooring mechanism inflates QLIKE."
        ),
        label="tab:main_results",
        n_assets=40,
        mse_scale="1e6",
        mae_scale="1e4",
        mcs_by_h=mcs_stocks,
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
    # Table 3: MCS inclusion rates (all 50 assets)
    # ================================================================
    all_tickers = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
    print("Generating Table 3: MCS inclusion rates (50 assets)...")
    table3 = make_mcs_table(
        mcs_volare,
        all_tickers,
        caption=(
            "Model Confidence Set inclusion rates for 50 assets "
            "(40 equities, 5 FX, 5 futures). "
            "Each cell reports the percentage of assets for which the model is "
            "included in the MCS at the 10\\% significance level (QLIKE loss, "
            "$T_{\\max}$ statistic, block bootstrap with $B = 10{,}000$)."
        ),
        label="tab:mcs",
    )
    (TABLE_DIR / "table_equity_mcs.tex").write_text(table3)

    # ================================================================
    # FX and futures per-class metrics (used by the combined table below and the
    # MCS markers). The standalone single-class tables (table_fx_metrics /
    # table_futures_metrics) were superseded by the combined FX+futures table
    # (Referee 2 minor 1) and are no longer written.
    # ================================================================
    fx_metrics = compute_asset_class_metrics(per_asset, VOLARE_FX_TICKERS).reset_index()
    fut_metrics = compute_asset_class_metrics(per_asset, VOLARE_FUTURES_TICKERS).reset_index()

    # ================================================================
    # Combined FX + Futures table (Referee 2 minor 1)
    # ================================================================
    print("Generating combined FX + futures table...")
    combined = make_combined_metrics_table(
        [("Panel A: FX (5 pairs; MSE $\\times 10^{-8}$)",
          fx_metrics, "1e8", "1e4", mcs_fx),
         ("Panel B: Futures (5 contracts; MSE $\\times 10^{-6}$)",
          fut_metrics, "1e6", "1e4", mcs_fut)],
        caption=("Forecast accuracy for FX and futures (VOLARE), cross-sectional averages. "
                 "Bold marks the best value per column within each horizon block. "
                 "$\\dagger$ marks QLIKE $>1$; $^{\\ast}$ marks models in the Model Confidence Set "
                 "(10\\%) for a majority of the panel's assets at that horizon."),
        label="tab:fx_futures_results",
    )
    (TABLE_DIR / "table_fx_futures_metrics.tex").write_text(combined)

    # ================================================================
    # Average QLIKE loss-ratio table vs Log-HAR (Referee 2 sec. 3), 50 assets
    # ================================================================
    print("Generating loss-ratio table (vs Log-HAR)...")
    lr = make_loss_ratio_table(
        per_asset,
        VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS,
        caption=("Average QLIKE loss ratios relative to Log-HAR across all 50 assets. "
                 "Robust to outlier assets; values below 1 beat Log-HAR on average."),
        label="tab:loss_ratios", benchmark="Log_HAR",
    )
    (TABLE_DIR / "table_loss_ratios.tex").write_text(lr)

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

    all_tickers_mcs = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
    print("\n--- VOLARE MCS inclusion rates (50 assets) ---")
    for h in HORIZONS:
        print(f"\nh={h}:")
        for m in MODEL_ORDER:
            mask = (mcs_volare["model"] == m) & (mcs_volare["horizon"] == h) & \
                   (mcs_volare["ticker"].isin(all_tickers_mcs))
            matched = mcs_volare[mask]
            if len(matched) > 0:
                rate = matched["in_mcs"].mean() * 100
                print(f"  {MODEL_DISPLAY[m]:20s}  {rate:.1f}%")

    print("\nDone! All tables written to", TABLE_DIR)


if __name__ == "__main__":
    main()
