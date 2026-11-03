"""Generate box plot figure showing distribution of QLIKE ratios (model/Log-HAR) across assets.

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
import fig_style

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "results" / "volare" / "metrics"
FIG_DIR = BASE_DIR / "paper" / "figures"

# Models to include. The denominator Log-HAR is the reference line, not a box.
# Every econometric benchmark and every foundation model except Toto is shown. The
# augmented HARs are back on scale since their direct engine was corrected (their old
# per-asset ratios, HARQ median ~1.9 and max >70, came from a variance floor). Toto stays
# out because its per-asset ratios would compress the scale for the other models.
_SHOWN = (["HAR", "HAR_J", "HAR_RS", "HARQ", "ARFIMA", "ARMA", "MEM"]
          + [m for m in fig_style.TSFM_ORDER if m != "toto"])
MODELS = {m: fig_style.MODEL_DISPLAY[m] for m in _SHOWN}
COLORS = {fig_style.MODEL_DISPLAY[m]: fig_style.MODEL_COLORS[m] for m in _SHOWN}

HORIZONS = [1, 5, 22]


def load_qlike_ratios(horizon):
    """Load per-asset QLIKE and compute ratios relative to Log-HAR (the headline
    benchmark, matching Table loss_ratios and the figure caption/text)."""
    df = pd.read_csv(METRICS_DIR / f"metrics_by_asset_h{horizon}.csv")

    # Get Log-HAR QLIKE per asset (the denominator)
    base = df[df["model"] == "Log_HAR"][["ticker", "QLIKE"]].rename(columns={"QLIKE": "QLIKE_BASE"})

    ratios = {}
    for model_key, model_name in MODELS.items():
        sub = df[df["model"] == model_key][["ticker", "QLIKE"]]
        merged = sub.merge(base, on="ticker")
        # Ratio: values < 1 mean the model beats Log-HAR
        ratio = merged["QLIKE"] / merged["QLIKE_BASE"]
        ratios[model_name] = ratio.values

    return ratios


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=fig_style.figsize(1.0, (1, 1)), sharey=True,
                             layout="constrained")

    # One vertical scale for the three horizons: the union of the per-horizon ranges, each
    # the 1st to 99th percentile of the plotted ratios widened by 0.3
    ylims = []
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
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
            capprops=dict(linewidth=0.5),
            medianprops=dict(color="black", linewidth=1.0),
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, zorder=0)
        ax.set_title(f"$h = {h}$")
        if i == 0:
            ax.set_ylabel("QLIKE ratio (model / Log-HAR)")
        ax.tick_params(axis="x", rotation=90, labelsize=fig_style.DENSE_TICK_SIZE)

        all_vals = np.concatenate(data)
        q01, q99 = np.percentile(all_vals, [1, 99])
        ylims.append((max(0, q01 - 0.3), q99 + 0.3))

    axes[0].set_ylim(min(lo for lo, _ in ylims), max(hi for _, hi in ylims))

    out_path = FIG_DIR / "fig_qlike_boxplot.pdf"
    fig_style.savefig(fig, out_path)
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
