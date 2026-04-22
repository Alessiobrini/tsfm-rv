"""
Generate publication-quality figures for the RV forecasting paper.
Saves all figures as PDF to paper/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = r"G:\Other computers\Dell Duke\Workfiles\Postdoc_file\human_x_AI_finance"
FORECAST_DIR = os.path.join(BASE, "results", "volare", "forecasts")
METRICS_DIR = os.path.join(BASE, "results", "volare", "metrics")
FIG_DIR = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 17,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

VOLARE_TICKERS = [
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM",
    "CSCO", "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "META", "MMM", "MRK", "MSFT", "NFLX", "NKE",
    "NVDA", "ORCL", "PG", "PM", "SHW", "TRV", "TSLA", "UNH", "V", "VZ",
    "WMT", "XOM",
]

# ===========================================================================
# Figure 1 — Forecast vs Actual time series (2x2 grid: AAPL, JPM, TSLA, EURUSD)
# ===========================================================================
def figure1():
    assets = ["AAPL", "JPM", "TSLA", "EURUSD"]
    panel_labels = ["(a) AAPL", "(b) JPM", "(c) TSLA", "(d) EUR/USD"]

    model_specs = {
        "Log-HAR": ("Log_HAR", "#1f77b4"),
        "Sundial": ("sundial", "#d62728"),
        "Moirai-MoE-S": ("moirai_moe_small", "#2ca02c"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for i, (ticker, panel_title) in enumerate(zip(assets, panel_labels)):
        ax = axes[i]

        # Load all model forecasts for this ticker
        dfs = {}
        for label, (prefix, _) in model_specs.items():
            fname = f"{prefix}_{ticker}_h1.csv"
            df = pd.read_csv(os.path.join(FORECAST_DIR, fname), parse_dates=["date"])
            df = df.set_index("date").sort_index()
            dfs[label] = df

        # Intersect dates across models
        common_idx = dfs["Log-HAR"].index
        for df in dfs.values():
            common_idx = common_idx.intersection(df.index)
        common_idx = common_idx.sort_values()

        # Last 500 observations
        common_idx = common_idx[-500:]

        # Plot actual RV
        actual = dfs["Log-HAR"].loc[common_idx, "actual"]
        ax.plot(common_idx, actual, color="0.6", linewidth=0.7, label="Actual RV",
                zorder=1)

        # Plot forecasts
        for label, (_, color) in model_specs.items():
            fc = dfs[label].loc[common_idx, "forecast"]
            ax.plot(common_idx, fc, color=color, linewidth=0.8, label=label, zorder=2)

        # Title removed — caption provides panel labels
        ax.text(0.02, 0.95, panel_title, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
        ax.set_ylabel("Realized Variance", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

        # Only show legend on first panel
        if i == 0:
            ax.legend(loc="upper right", fontsize=11)

    fig.tight_layout(h_pad=3.0, w_pad=2.5)

    outpath = os.path.join(FIG_DIR, "fig1_forecast_vs_actual.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Created: {outpath}")


# ===========================================================================
# Figure 2 — MCS Inclusion Heatmap (40 equities)
# ===========================================================================
def figure2():
    mcs = pd.read_csv(os.path.join(METRICS_DIR, "mcs_all_results.csv"))

    # Model name mapping (internal -> display)
    model_map = {
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
        "timesfm_2_5": "TimesFM 2.5",
        "toto": "Toto",
        "sundial": "Sundial",
        "moirai_moe_small": "Moirai-MoE-S",
        "ttm": "TTM",
    }

    # Use all 50 tickers
    from config import VOLARE_ALL_TICKERS
    all_tickers = VOLARE_ALL_TICKERS

    mcs = mcs[mcs["ticker"].isin(all_tickers) & mcs["model"].isin(model_map.keys())].copy()
    mcs["model_display"] = mcs["model"].map(model_map)

    # Sort tickers by asset class (equities, FX, futures) then alphabetically within class
    from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS
    tickers_sorted = (sorted([t for t in all_tickers if t in VOLARE_STOCK_TICKERS])
                      + sorted([t for t in all_tickers if t in VOLARE_FX_TICKERS])
                      + sorted([t for t in all_tickers if t in VOLARE_FUTURES_TICKERS]))

    # Model display order
    model_order = [
        "HAR", "HAR-J", "HAR-RS", "HARQ", "Log-HAR", "ARFIMA",
        "Chronos-Bolt-S", "Chronos-Bolt-B", "Moirai-2.0-S", "Moirai-MoE-S",
        "Lag-Llama", "TimesFM 2.5", "Toto", "Sundial", "TTM",
    ]

    horizons = [1, 5, 22]
    fig, axes = plt.subplots(1, 3, figsize=(18, 12), sharey=True)

    for ax, h in zip(axes, horizons):
        sub = mcs[mcs["horizon"] == h].copy()
        pivot = sub.pivot_table(
            index="ticker", columns="model_display", values="in_mcs", aggfunc="max"
        )
        # Reindex to ensure consistent ordering
        pivot = pivot.reindex(index=tickers_sorted, columns=model_order).fillna(0)

        # Draw heatmap manually
        data = pivot.values.astype(float)
        ax.imshow(
            data,
            aspect="auto",
            cmap=plt.cm.colors.ListedColormap(["white", "#90EE90"]),
            vmin=0, vmax=1,
            interpolation="nearest",
        )

        # Grid lines
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(data.shape[0]))
        if ax == axes[0]:
            ax.set_yticklabels(tickers_sorted, fontsize=7)
        ax.set_title(f"$h = {h}$", fontsize=15)

        # Minor gridlines for cell borders
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="0.8", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, "fig2_mcs_heatmap.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Created: {outpath}")


# ===========================================================================
if __name__ == "__main__":
    print("Generating Figure 1...")
    figure1()
    print("Generating Figure 2...")
    figure2()
    print("Done.")
