"""
Generate Fig. 1 (forecasts against realized volatility) and Appendix Fig. 5 (MCS inclusion
heatmap) of the paper. Saves both as PDF to paper/figures/, drawn at their printed size.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import fig_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent          # project root (portable)
FORECAST_DIR = str(BASE / "results" / "volare" / "forecasts")
METRICS_DIR = str(BASE / "results" / "volare" / "metrics")
FIG_DIR = str(BASE / "paper" / "figures")
os.makedirs(FIG_DIR, exist_ok=True)

fig_style.apply_style()

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

    # Headline benchmark plus the two representative foundation models named in
    # the caption (TTM, the only FM to beat Log-HAR, and Sundial).
    model_specs = {
        fig_style.MODEL_DISPLAY[m]: (m, fig_style.MODEL_COLORS[m])
        for m in ("Log_HAR", "ttm", "sundial")
    }

    fig, axes = plt.subplots(2, 2, figsize=fig_style.figsize(1.0, (2.4, 2)),
                             layout="constrained")
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

        # Last 125 observations, about six months, so that the forecasts can be told apart
        common_idx = common_idx[-125:]

        # Plot actual RV
        actual = dfs["Log-HAR"].loc[common_idx, "actual"]
        ax.plot(common_idx, actual, color=fig_style.ACTUAL_COLOR, linewidth=0.5,
                label="Realized volatility", zorder=1)

        # Plot forecasts
        for label, (_, color) in model_specs.items():
            fc = dfs[label].loc[common_idx, "forecast"]
            ax.plot(common_idx, fc, color=color, linewidth=0.8, label=label, zorder=2)

        # Panel label above the axes, so that it never covers the data
        ax.set_title(panel_title, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    # One legend above the four panels, and one label per axis (the panels keep their own
    # vertical scale, since the volatility level differs across assets)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    fig.supylabel("Realized volatility (decimal, daily)")
    fig.supxlabel("Forecast date")

    outpath = os.path.join(FIG_DIR, "fig1_forecast_vs_actual.pdf")
    fig_style.savefig(fig, outpath)
    plt.close(fig)
    print(f"Created: {outpath}")


# ===========================================================================
# Appendix Figure 5 — MCS inclusion heatmap (50 assets)
# ===========================================================================
def figure5_mcs_heatmap():
    mcs = pd.read_csv(os.path.join(METRICS_DIR, "mcs_all_results.csv"))

    # Models shown (internal -> display, the names of the tables). ARMA and MEM are omitted
    # for space, as the caption states.
    shown = ["HAR", "HAR_J", "HAR_RS", "HARQ", "Log_HAR", "ARFIMA"] + fig_style.TSFM_ORDER
    model_map = {m: fig_style.MODEL_DISPLAY[m] for m in shown}

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

    # Model display order: econometric models, then the foundation models in table order
    model_order = [model_map[m] for m in shown]

    horizons = [1, 5, 22]
    fig, axes = plt.subplots(1, 3, figsize=fig_style.figsize(1.0, (1.95, 1)), sharey=True,
                             layout="constrained")

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
        ax.set_xticklabels(model_order, rotation=90, fontsize=fig_style.DENSE_TICK_SIZE)
        ax.set_yticks(np.arange(data.shape[0]))
        if ax == axes[0]:
            ax.set_yticklabels(tickers_sorted)
        ax.set_title(f"$h = {h}$")
        ax.tick_params(which="major", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)

        # Minor gridlines for cell borders
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="0.8", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

    # The file keeps its original name, fig2_mcs_heatmap.pdf, which main.tex includes.
    outpath = os.path.join(FIG_DIR, "fig2_mcs_heatmap.pdf")
    fig_style.savefig(fig, outpath)
    plt.close(fig)
    print(f"Created: {outpath}")


# ===========================================================================
if __name__ == "__main__":
    print("Generating Figure 1...")
    figure1()
    print("Generating Appendix Figure 5...")
    figure5_mcs_heatmap()
    print("Done.")
