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

TSFM_MODELS = {
    "chronos_bolt_small": "Chronos-Bolt-S",
    "moirai_2_0_small": "Moirai-2.0",
    "lag_llama": "Lag-Llama",
}


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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for idx, (model_key, model_name) in enumerate(TSFM_MODELS.items()):
        ax = axes[idx]
        sub = df_h1[df_h1["model"] == model_key].set_index("ticker")["QLIKE"]
        common = list(set(sub.index) & set(har_h1.index) & set(rho1.keys()))
        common.sort()

        x = np.array([rho1[t] for t in common])
        y = np.array([sub[t] / har_h1[t] for t in common])

        # Color by asset class
        colors = []
        markers = []
        for t in common:
            if t in STOCKS:
                colors.append("#1f77b4")
                markers.append("o")
            elif t in FX:
                colors.append("#d62728")
                markers.append("s")
            else:
                colors.append("#2ca02c")
                markers.append("^")

        for t, xi, yi, c, m in zip(common, x, y, colors, markers):
            ax.scatter(xi, yi, c=c, marker=m, s=30, alpha=0.7, edgecolors="none")

        # Trend line
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", alpha=0.5, linewidth=1)

        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\rho_1$ (first-order autocorrelation)")
        ax.set_ylabel("QLIKE ratio (model / HAR)" if idx == 0 else "")
        ax.set_title(f"{model_name}\n$r = {r_value:.2f}$, $p = {p_value:.3f}$",
                     fontsize=11)
        ax.tick_params(labelsize=9)

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

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Equities"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728", markersize=8, label="FX"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c", markersize=8, label="Futures"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout()
    out_path = FIG_DIR / "fig_persistence_drivers.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
