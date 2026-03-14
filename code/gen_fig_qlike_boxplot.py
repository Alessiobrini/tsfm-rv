"""Generate box plot figure showing distribution of QLIKE ratios (model/HAR) across assets.

Inspired by Carriero et al. (2024) Figures 2-8 which use box plots of relative
RMSFE to show the full distribution of cross-asset performance.

Produces: paper/figures/fig_qlike_boxplot.pdf
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "results" / "volare" / "metrics"
FIG_DIR = BASE_DIR / "paper" / "figures"

# Models to include (exclude levels-HAR variants with extreme QLIKE)
MODELS = {
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chr-Bolt-S",
    "chronos_bolt_base": "Chr-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0",
    "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "sundial": "Sundial",
    "moirai_moe_small": "Moirai-MoE-S",
}

COLORS = {
    "Log-HAR": "#1f77b4",
    "ARFIMA": "#2ca02c",
    "Chr-Bolt-S": "#d62728",
    "Chr-Bolt-B": "#e377c2",
    "Moirai-2.0": "#ff7f0e",
    "Lag-Llama": "#9467bd",
    "TimesFM-2.5": "#8c564b",
    "Sundial": "#17becf",
    "Moirai-MoE-S": "#bcbd22",
}

HORIZONS = [1, 5, 22]


def load_qlike_ratios(horizon):
    """Load per-asset QLIKE and compute ratios relative to HAR."""
    df = pd.read_csv(METRICS_DIR / f"metrics_by_asset_h{horizon}.csv")

    # Get HAR QLIKE per asset
    har = df[df["model"] == "HAR"][["ticker", "QLIKE"]].rename(columns={"QLIKE": "QLIKE_HAR"})

    ratios = {}
    for model_key, model_name in MODELS.items():
        sub = df[df["model"] == model_key][["ticker", "QLIKE"]]
        merged = sub.merge(har, on="ticker")
        # Ratio: values < 1 mean the model beats HAR
        ratio = merged["QLIKE"] / merged["QLIKE_HAR"]
        ratios[model_name] = ratio.values

    return ratios


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for i, h in enumerate(HORIZONS):
        ax = axes[i]
        ratios = load_qlike_ratios(h)

        model_names = list(MODELS.values())
        data = [ratios[m] for m in model_names]
        colors = [COLORS[m] for m in model_names]

        bp = ax.boxplot(
            data,
            labels=model_names,
            patch_artist=True,
            widths=0.6,
            showfliers=False,  # hide outliers for cleaner look
            medianprops=dict(color="black", linewidth=1.5),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"$h = {h}$", fontsize=13)
        ax.set_ylabel("QLIKE ratio (model / HAR)" if i == 0 else "")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(labelsize=9)

        # Set reasonable y limits
        all_vals = np.concatenate(data)
        q01, q99 = np.percentile(all_vals, [1, 99])
        ymin = max(0, q01 - 0.3)
        ymax = q99 + 0.3
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Distribution of QLIKE Ratios Relative to HAR Across 50 Assets",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = FIG_DIR / "fig_qlike_boxplot.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {out_path}")

    # Also print summary statistics for the paper
    print("\n=== Summary Statistics (Median QLIKE Ratio) ===")
    for h in HORIZONS:
        ratios = load_qlike_ratios(h)
        print(f"\nh = {h}:")
        for m in MODELS.values():
            vals = ratios[m]
            print(f"  {m:15s}: median={np.median(vals):.3f}, "
                  f"IQR=[{np.percentile(vals, 25):.3f}, {np.percentile(vals, 75):.3f}], "
                  f"frac<1={np.mean(vals < 1):.1%}")


if __name__ == "__main__":
    main()
