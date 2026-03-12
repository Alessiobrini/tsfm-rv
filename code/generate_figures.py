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
    "font.size": 11,
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
        "Moirai-2.0-S": ("moirai_2_0_small", "#d62728"),
        "Chronos-Bolt-S": ("chronos_bolt_small", "#2ca02c"),
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

        ax.set_title(panel_title, fontsize=14, fontweight="bold")
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
    }

    # Filter to 40 tickers and the 9 models
    mcs = mcs[mcs["ticker"].isin(VOLARE_TICKERS) & mcs["model"].isin(model_map.keys())].copy()
    mcs["model_display"] = mcs["model"].map(model_map)

    # Sort tickers alphabetically
    tickers_sorted = sorted(VOLARE_TICKERS)

    # Model display order
    model_order = [
        "HAR", "HAR-J", "HAR-RS", "HARQ", "Log-HAR", "ARFIMA",
        "Chronos-Bolt-S", "Chronos-Bolt-B", "Moirai-2.0-S",
    ]

    horizons = [1, 5, 22]
    fig, axes = plt.subplots(1, 3, figsize=(14, 12), sharey=True)

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
        ax.set_title(f"h = {h}", fontsize=11, fontweight="bold")

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
# Figure 3 — Cumulative QLIKE loss differential (Moirai vs Log-HAR)
# ===========================================================================
def figure3():
    all_cum_diffs = []

    for ticker in VOLARE_TICKERS:
        moirai_file = os.path.join(FORECAST_DIR, f"moirai_2_0_small_{ticker}_h1.csv")
        loghar_file = os.path.join(FORECAST_DIR, f"Log_HAR_{ticker}_h1.csv")

        if not os.path.exists(moirai_file) or not os.path.exists(loghar_file):
            print(f"  Skipping {ticker}: missing file")
            continue

        df_m = pd.read_csv(moirai_file, parse_dates=["date"]).set_index("date").sort_index()
        df_l = pd.read_csv(loghar_file, parse_dates=["date"]).set_index("date").sort_index()

        # Intersect dates
        common = df_m.index.intersection(df_l.index).sort_values()
        if len(common) == 0:
            continue

        actual = df_l.loc[common, "actual"].values
        fc_m = df_m.loc[common, "forecast"].values
        fc_l = df_l.loc[common, "forecast"].values

        # QLIKE: L = actual/forecast - log(actual/forecast) - 1
        # Clip to avoid division by zero / log of non-positive
        eps = 1e-12
        fc_m = np.clip(fc_m, eps, None)
        fc_l = np.clip(fc_l, eps, None)
        actual_c = np.clip(actual, eps, None)

        ql_m = actual_c / fc_m - np.log(actual_c / fc_m) - 1
        ql_l = actual_c / fc_l - np.log(actual_c / fc_l) - 1

        # d_t = L_LogHAR - L_Moirai (positive = Moirai better)
        d_t = ql_l - ql_m
        cum_d = np.cumsum(d_t)

        s = pd.Series(cum_d, index=common, name=ticker)
        all_cum_diffs.append(s)

    # Align all series on a common date index
    combined = pd.concat(all_cum_diffs, axis=1)
    mean_cum = combined.mean(axis=1)
    std_cum = combined.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_cum.index, mean_cum.values, color="#1f77b4", linewidth=1.2)
    ax.fill_between(
        mean_cum.index,
        (mean_cum - std_cum).values,
        (mean_cum + std_cum).values,
        alpha=0.2, color="#1f77b4",
    )
    ax.axhline(0, color="0.4", linewidth=0.6, linestyle="-")

    # COVID vertical line
    covid_date = pd.Timestamp("2020-03-01")
    if mean_cum.index.min() < covid_date < mean_cum.index.max():
        ax.axvline(covid_date, color="0.3", linewidth=0.8, linestyle="--")
        # Place label slightly to the right
        ypos = ax.get_ylim()[1] * 0.92
        ax.text(covid_date + pd.Timedelta(days=10), ypos, "COVID-19",
                fontsize=9, color="0.3", va="top")

    ax.set_ylabel("Cumulative QLIKE differential (Log-HAR minus Moirai)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()

    outpath = os.path.join(FIG_DIR, "fig3_cumulative_qlike_diff.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Created: {outpath}")


# ===========================================================================
if __name__ == "__main__":
    print("Generating Figure 1...")
    figure1()
    print("Generating Figure 2...")
    figure2()
    print("Generating Figure 3...")
    figure3()
    print("Done.")
