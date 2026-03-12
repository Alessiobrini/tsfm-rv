"""Generate per-asset QLIKE breakdown tables for the paper (h=1, h=5, h=22)."""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "volare", "forecasts")
TABLES_DIR = os.path.join(BASE_DIR, "paper", "tables")

# Models to include and their display names
MODELS = {
    "HAR": "HAR",
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chr-Bolt-S",
    "chronos_bolt_base": "Chr-Bolt-B",
    "moirai_2_0_small": "Moirai-S",
    "lag_llama": "Lag-Llama",
    "kronos": "Kronos",
}

# Asset grouping
STOCKS = [
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT",
    "CRM", "CSCO", "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON",
    "IBM", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MRK", "MSFT",
    "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "SHW", "TRV", "TSLA",
    "UNH", "V", "VZ", "WMT", "XOM",
]
FX = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY"]
FUTURES = ["C", "CL", "ES", "GC", "NG"]

ALL_ASSETS = STOCKS + FX + FUTURES


def qlike(actual, forecast):
    """QLIKE loss: mean(actual/forecast - log(actual/forecast) - 1).
    Matches evaluation/loss_functions.py: filters non-positive values."""
    mask = (actual > 0) & (forecast > 0)
    a, f = actual[mask], forecast[mask]
    ratio = a / f
    return np.mean(ratio - np.log(ratio) - 1)


def format_row(asset_name, values, best_idx):
    """Format a single row with bold on the best value."""
    parts = [asset_name]
    for i, v in enumerate(values):
        if np.isnan(v):
            parts.append("--")
        elif i == best_idx:
            parts.append(r"\textbf{" + f"{v:.4f}" + "}")
        else:
            parts.append(f"{v:.4f}")
    return " & ".join(parts) + r" \\"


def format_avg_row(label, means):
    """Format an average row with bold on the best value."""
    best_mean = np.nanargmin(means)
    parts = [label]
    for i, v in enumerate(means):
        if np.isnan(v):
            parts.append("--")
        elif i == best_mean:
            parts.append(r"\textbf{" + f"{v:.4f}" + "}")
        else:
            parts.append(f"{v:.4f}")
    return " & ".join(parts) + r" \\"


def generate_table(horizon):
    """Generate per-asset QLIKE table for a given forecast horizon."""
    h_str = f"h{horizon}"
    model_cols = list(MODELS.values())
    ncols = len(model_cols)

    # Output path
    if horizon == 1:
        out_path = os.path.join(TABLES_DIR, "table_per_asset_qlike.tex")
        label = "tab:per_asset_qlike"
    else:
        out_path = os.path.join(TABLES_DIR, f"table_per_asset_qlike_h{horizon}.tex")
        label = f"tab:per_asset_qlike_h{horizon}"

    # Compute QLIKE for each (asset, model)
    rows = []
    for ticker in ALL_ASSETS:
        row = {"Asset": ticker}
        for model_key, model_name in MODELS.items():
            fpath = os.path.join(RESULTS_DIR, f"{model_key}_{ticker}_{h_str}.csv")
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                row[model_name] = qlike(df["actual"].values, df["forecast"].values)
            else:
                row[model_name] = np.nan
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # --- Build LaTeX ---
    lines = []
    lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(r"\tiny")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + "l" + "r" * ncols + "}")
    lines.append(r"\toprule")

    # Header
    header = "Asset & " + " & ".join(model_cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    def write_group(group_label, tickers):
        lines.append(r"\multicolumn{" + str(ncols + 1) + r"}{l}{\textit{" + group_label + r"}} \\")
        for ticker in tickers:
            row_data = result_df[result_df["Asset"] == ticker]
            if row_data.empty:
                continue
            vals = row_data[model_cols].values[0]
            valid = ~np.isnan(vals)
            best = np.nanargmin(vals) if valid.any() else -1
            lines.append(format_row(ticker, vals, best))
        # Group average
        group_df = result_df[result_df["Asset"].isin(tickers)]
        means = group_df[model_cols].mean().values
        lines.append(format_avg_row(r"\textit{Average}", means))

    write_group("Stocks", STOCKS)
    lines.append(r"\midrule")
    write_group("FX", FX)
    lines.append(r"\midrule")
    write_group("Futures", FUTURES)
    lines.append(r"\midrule")

    # Overall average
    overall_means = result_df[model_cols].mean().values
    lines.append(format_avg_row(r"\textbf{Overall Average}", overall_means))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\\[6pt]")
    lines.append(
        r"\parbox{\linewidth}{\footnotesize Per-asset QLIKE loss at $h="
        + str(horizon) + r"$. "
        r"QLIKE is computed as $\mathrm{mean}(RV_t / \hat{RV}_t - \ln(RV_t / \hat{RV}_t) - 1)$ "
        r"over observations with positive actual and forecast values. Bold indicates the lowest (best) QLIKE in each row. "
        r"Chr-Bolt-S = Chronos-Bolt-Small, Chr-Bolt-B = Chronos-Bolt-Base, Moirai-S = Moirai-2.0-Small. "
        r"HAR-J, HAR-RS, and HARQ are omitted as their QLIKE values are dominated by "
        r"near-zero forecast artifacts (see Section~5.1).}"
    )
    lines.append(r"\end{table}")
    lines.append(r"\end{landscape}")

    tex = "\n".join(lines)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(tex)

    print(f"\n=== h={horizon} ===")
    print(f"Table written to {out_path}")
    print(f"Assets: {len(ALL_ASSETS)}, Models: {len(MODELS)}")

    # Count missing
    missing = result_df[model_cols].isna().sum().sum()
    if missing > 0:
        print(f"WARNING: {missing} missing values")

    print("\nOverall average QLIKE:")
    for col in model_cols:
        val = result_df[col].mean()
        print(f"  {col:15s}: {val:.4f}" if not np.isnan(val) else f"  {col:15s}: --")


if __name__ == "__main__":
    # Generate tables for all three horizons
    for h in [1, 5, 22]:
        generate_table(h)
