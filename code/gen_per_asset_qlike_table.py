"""Generate per-asset QLIKE breakdown table (h=1) for the paper."""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance\results\volare\forecasts"
OUT_PATH = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance\paper\tables\table_per_asset_qlike.tex"

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
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "C", "CAT",
    "CRM", "CSCO", "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON",
    "IBM", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MRK", "MSFT",
    "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "SHW", "TRV", "TSLA",
    "UNH", "V", "VZ", "WMT", "XOM",
]
FX = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY"]
FUTURES = ["CL", "ES", "GC", "NG"]

ALL_ASSETS = STOCKS + FX + FUTURES


def qlike(actual, forecast):
    """QLIKE loss: mean(actual/forecast - log(actual/forecast) - 1)."""
    f = np.maximum(forecast, 1e-6)
    ratio = actual / f
    return np.mean(ratio - np.log(ratio) - 1)


# Compute QLIKE for each (asset, model)
rows = []
for ticker in ALL_ASSETS:
    row = {"Asset": ticker}
    for model_key, model_name in MODELS.items():
        fpath = os.path.join(RESULTS_DIR, f"{model_key}_{ticker}_h1.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            row[model_name] = qlike(df["actual"].values, df["forecast"].values)
        else:
            row[model_name] = np.nan
    rows.append(row)

result_df = pd.DataFrame(rows)
model_cols = list(MODELS.values())

# --- Build LaTeX ---
lines = []
lines.append(r"\begin{landscape}")
lines.append(r"\begin{table}[p]")
lines.append(r"\centering")
lines.append(r"\tiny")
lines.append(r"\caption{Per-Asset QLIKE Loss at $h=1$}")
lines.append(r"\label{tab:per_asset_qlike}")

ncols = len(model_cols)
col_fmt = "l" + "r" * ncols
lines.append(r"\begin{tabular}{" + col_fmt + "}")
lines.append(r"\toprule")

# Header
header = "Asset & " + " & ".join(model_cols) + r" \\"
lines.append(header)
lines.append(r"\midrule")


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


def write_group(label, tickers):
    lines.append(r"\multicolumn{" + str(ncols + 1) + r"}{l}{\textit{" + label + r"}} \\")
    for ticker in tickers:
        row_data = result_df[result_df["Asset"] == ticker]
        if row_data.empty:
            continue
        vals = row_data[model_cols].values[0]
        valid = ~np.isnan(vals)
        if valid.any():
            best = np.nanargmin(vals)
        else:
            best = -1
        lines.append(format_row(ticker, vals, best))
    # Group average
    group_df = result_df[result_df["Asset"].isin(tickers)]
    means = group_df[model_cols].mean()
    best_mean = np.nanargmin(means.values)
    parts = [r"\textit{Average}"]
    for i, v in enumerate(means.values):
        if np.isnan(v):
            parts.append("--")
        elif i == best_mean:
            parts.append(r"\textbf{" + f"{v:.4f}" + "}")
        else:
            parts.append(f"{v:.4f}")
    lines.append(" & ".join(parts) + r" \\")


write_group("Stocks", STOCKS)
lines.append(r"\midrule")
write_group("FX", FX)
lines.append(r"\midrule")
write_group("Futures", FUTURES)
lines.append(r"\midrule")

# Overall average
overall_means = result_df[model_cols].mean()
best_overall = np.nanargmin(overall_means.values)
parts = [r"\textbf{Overall Average}"]
for i, v in enumerate(overall_means.values):
    if np.isnan(v):
        parts.append("--")
    elif i == best_overall:
        parts.append(r"\textbf{" + f"{v:.4f}" + "}")
    else:
        parts.append(f"{v:.4f}")
lines.append(" & ".join(parts) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(
    r"\par\smallskip\noindent\footnotesize "
    r"Notes: QLIKE is computed as $\mathrm{mean}(RV_t / \hat{RV}_t - \ln(RV_t / \hat{RV}_t) - 1)$ "
    r"with forecasts floored at $10^{-6}$. Bold indicates the lowest (best) QLIKE in each row. "
    r"Chr-Bolt-S = Chronos-Bolt-Small, Chr-Bolt-B = Chronos-Bolt-Base, Moirai-S = Moirai-2.0-Small."
)
lines.append(r"\end{table}")
lines.append(r"\end{landscape}")

tex = "\n".join(lines)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write(tex)

print(f"Table written to {OUT_PATH}")
print(f"Assets: {len(ALL_ASSETS)}, Models: {len(MODELS)}")
print("\nOverall average QLIKE:")
for col in model_cols:
    print(f"  {col:15s}: {result_df[col].mean():.4f}")
