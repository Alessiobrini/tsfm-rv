"""
compute_bootstrap_ci.py — Compute bootstrap confidence intervals on aggregate metrics.

Reads per-asset metrics files, bootstraps across tickers (B=10,000), and reports
95% confidence intervals for each (model, horizon, metric) combination.

Usage:
    python compute_bootstrap_ci.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS

METRICS_DIR = PROJECT_ROOT / "results" / "volare" / "metrics"
HORIZONS = [1, 5, 22]
B = 10_000
SEED = 42

MODEL_DISPLAY = {
    "HAR": "HAR", "HAR_J": "HAR-J", "HAR_RS": "HAR-RS", "HARQ": "HARQ",
    "Log_HAR": "Log-HAR", "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S", "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto", "sundial": "Sundial", "moirai_moe_small": "Moirai-MoE-S",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())
METRICS = ["MSE", "MAE", "QLIKE", "R2OOS"]

ASSET_CLASSES = {
    "equity": VOLARE_STOCK_TICKERS,
    "fx": VOLARE_FX_TICKERS,
    "futures": VOLARE_FUTURES_TICKERS,
}


def bootstrap_ci(values, B=10_000, alpha=0.05, rng=None):
    """Compute bootstrap percentile CI for the mean."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(values)
    means = np.array([
        np.mean(rng.choice(values, size=n, replace=True)) for _ in range(B)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    rng = np.random.default_rng(SEED)

    rows = []
    for h in HORIZONS:
        fpath = METRICS_DIR / f"metrics_by_asset_h{h}.csv"
        df = pd.read_csv(fpath)
        print(f"h={h}: {len(df)} rows, {df['model'].nunique()} models")

        for ac_name, tickers in ASSET_CLASSES.items():
            sub_ac = df[df["ticker"].isin(tickers)]
            n_tickers = len(tickers)

            for model in MODEL_ORDER:
                sub = sub_ac[sub_ac["model"] == model]
                if len(sub) < 2:
                    continue

                for metric in METRICS:
                    vals = sub[metric].values
                    mean_val = np.mean(vals)
                    lo, hi = bootstrap_ci(vals, B=B, rng=rng)

                    rows.append({
                        "model": model,
                        "horizon": h,
                        "asset_class": ac_name,
                        "metric": metric,
                        "mean": mean_val,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_tickers": len(vals),
                    })

    result = pd.DataFrame(rows)
    out_path = METRICS_DIR / "aggregate_metrics_with_ci.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(result)} rows)")

    # Print summary for equity QLIKE
    print("\n=== Equity QLIKE with 95% CI ===")
    for h in HORIZONS:
        print(f"\nh={h}:")
        sub = result[(result["horizon"] == h) & (result["asset_class"] == "equity")
                     & (result["metric"] == "QLIKE")]
        sub = sub.set_index("model").reindex(MODEL_ORDER).dropna(subset=["mean"])
        for model in sub.index:
            r = sub.loc[model]
            name = MODEL_DISPLAY.get(model, model)
            print(f"  {name:20s}  {r['mean']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
