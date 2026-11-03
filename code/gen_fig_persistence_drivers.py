"""Analyze relationship between asset persistence (rho_1) and TSFM relative performance.

Inspired by Carriero et al. (2024) Section 6.2 which shows TSLMs struggle
with highly persistent series.

Produces: paper/figures/fig_persistence_drivers.pdf
Prints: correlation statistics for the paper text.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import fig_style
from config import VOLARE_RESULTS_DIR
from data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "results" / "volare" / "metrics"
FIG_DIR = BASE_DIR / "paper" / "figures"

STOCKS = [
    "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AXP", "BA", "CAT",
    "CRM", "CSCO", "CVX", "DIS", "GE", "GOOGL", "GS", "HD", "HON",
    "IBM", "JNJ", "JPM", "KO", "MCD", "META", "MMM", "MRK", "MSFT",
    "NFLX", "NKE", "NVDA", "ORCL", "PG", "PM", "SHW", "TRV", "TSLA",
    "UNH", "V", "VZ", "WMT", "XOM",
]
FX = ["AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDJPY"]
FUTURES = ["C", "CL", "ES", "GC", "NG"]
ALL_TICKERS = STOCKS + FX + FUTURES

# Every foundation model except Toto, in table order
TSFM_MODELS = {m: fig_style.MODEL_DISPLAY[m] for m in fig_style.TSFM_ORDER if m != "toto"}

# Asset-class colors and markers
CLASS_STYLE = {
    "Equities": ("#1f77b4", "o"),
    "FX": ("#d62728", "s"),
    "Futures": ("#2ca02c", "^"),
}


def format_p(p):
    """p-value for a panel title, as the text reports it: p < 0.001 when it rounds to zero."""
    return "$p < 0.001$" if round(p, 3) == 0 else f"$p = {p:.3f}$"


def compute_persistence():
    """Compute first-order autocorrelation for each asset from VOLARE CSVs."""
    from config import VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE
    rho1 = {}

    for fpath in [VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE]:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        for ticker in df["symbol"].unique():
            if ticker not in ALL_TICKERS:
                continue
            sub = df[df["symbol"] == ticker].sort_values("date")
            rv = sub["rv5"].dropna()
            if len(rv) > 100:
                rho1[ticker] = rv.autocorr(lag=1)

    return rho1


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Compute persistence
    print("Computing persistence (rho_1) for all assets...")
    rho1 = compute_persistence()
    print(f"Computed rho_1 for {len(rho1)} assets")

    # Load QLIKE ratios at h=1
    df_h1 = pd.read_csv(METRICS_DIR / "metrics_by_asset_h1.csv")
    har_h1 = df_h1[df_h1["model"] == "HAR"].set_index("ticker")["QLIKE"]

    n_models = len(TSFM_MODELS)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    remainder = n_models % ncols  # number of plots in last row

    fig_style.apply_style()
    fig = plt.figure(figsize=fig_style.figsize(1.0, (4.5, 3)), layout="constrained")

    # Build axes: full rows of 3, then centered last row
    axes_list = []
    first_in_row = set()
    full_rows = n_models // ncols
    gs = fig.add_gridspec(nrows, 2 * ncols)

    for r in range(full_rows):
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c * 2:(c + 1) * 2])
            if c == 0:
                first_in_row.add(len(axes_list))
            axes_list.append(ax)

    if remainder > 0:
        # Center the last row: for 2 plots in 6-unit grid, offset by 1
        offset = (2 * ncols - 2 * remainder) // 2
        for c in range(remainder):
            ax = fig.add_subplot(gs[full_rows, offset + c * 2:offset + (c + 1) * 2])
            if c == 0:
                first_in_row.add(len(axes_list))
            axes_list.append(ax)

    for idx, (model_key, model_name) in enumerate(TSFM_MODELS.items()):
        ax = axes_list[idx]
        sub = df_h1[df_h1["model"] == model_key].set_index("ticker")["QLIKE"]
        common = list(set(sub.index) & set(har_h1.index) & set(rho1.keys()))
        common.sort()

        x = np.array([rho1[t] for t in common])
        y = np.array([sub[t] / har_h1[t] for t in common])

        # Color by asset class
        colors = []
        markers = []
        for t in common:
            cls = "Equities" if t in STOCKS else "FX" if t in FX else "Futures"
            colors.append(CLASS_STYLE[cls][0])
            markers.append(CLASS_STYLE[cls][1])

        for t, xi, yi, c, m in zip(common, x, y, colors, markers):
            ax.scatter(xi, yi, c=c, marker=m, s=10, alpha=0.7, edgecolors="none")

        # Trend line
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", alpha=0.5, linewidth=0.8)

        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\rho_1$ (autocorrelation)")
        if idx in first_in_row:
            ax.set_ylabel("QLIKE ratio (model / HAR)")
        ax.set_title(f"{model_name}\n$r = {r_value:.2f}$, {format_p(p_value)}")

        print(f"\n{model_name} at h=1:")
        print(f"  Correlation(rho_1, QLIKE ratio): r={r_value:.3f}, p={p_value:.4f}")
        print(f"  Slope: {slope:.3f}")

        # Split by persistence
        high_mask = x > np.median(x)
        low_mask = ~high_mask
        print(f"  High persistence (rho_1 > {np.median(x):.3f}): "
              f"median ratio={np.median(y[high_mask]):.3f}, frac<1={np.mean(y[high_mask]<1):.1%}")
        print(f"  Low persistence  (rho_1 <= {np.median(x):.3f}): "
              f"median ratio={np.median(y[low_mask]):.3f}, frac<1={np.mean(y[low_mask]<1):.1%}")

    # No unused subplots to hide (gridspec handles this)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor=c, markersize=5, label=cls)
        for cls, (c, m) in CLASS_STYLE.items()
    ]
    # Directly above the first row of panels
    fig.legend(handles=legend_elements, loc="outside upper center", ncol=3)

    out_path = FIG_DIR / "fig_persistence_drivers.pdf"
    fig_style.savefig(fig, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
