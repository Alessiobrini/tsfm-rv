"""
compute_subsample_metrics.py — Compute pre/post-COVID forecast metrics from existing CSVs.

Reads all forecast CSVs from results/volare/forecasts/, splits at 2020-03-01,
computes MSE/MAE/QLIKE/R2OOS per (model, horizon, period), aggregates across
40 equity tickers. Saves updated subsample_metrics.csv and regenerates the
LaTeX subsample table.

Usage:
    python compute_subsample_metrics.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import VOLARE_STOCK_TICKERS, VOLARE_ALL_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS
from evaluation.loss_functions import mse, mae, qlike, r2_oos

FORECAST_DIR = PROJECT_ROOT / "results" / "volare" / "forecasts"
METRICS_DIR = PROJECT_ROOT / "results" / "volare" / "metrics"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
SPLIT_DATE = "2020-03-01"
HORIZONS = [1, 5, 22]

MODEL_DISPLAY = {
    "Log_HAR": "Log-HAR", "HAR": "HAR", "HAR_J": "HAR-J", "HAR_RS": "HAR-RS", "HARQ": "HARQ",
    "ARFIMA": "ARFIMA", "ARMA": "ARMA", "MEM": "MEM",
    "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S", "moirai_moe_small": "Moirai-MoE-S",
    "lag_llama": "Lag-Llama", "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto", "sundial": "Sundial", "ttm": "TTM",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())


def compute_metrics(actual, forecast):
    """Compute all four loss functions. Forecasts are on the volatility scale
    (already winsorized at generation); QLIKE squares them back to variance to
    match the main-results QLIKE (Patton 2011 proxy-robustness is a variance
    property). MSE/MAE/R2 stay on the volatility scale."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return {
        "MSE": mse(actual, forecast),
        "MAE": mae(actual, forecast),
        "QLIKE": qlike(actual, forecast, scale="vol"),
        "R2OOS": r2_oos(actual, forecast),
    }


def main():
    # Discover all models from filenames
    all_csvs = list(FORECAST_DIR.glob("*.csv"))
    model_ticker_horizon = {}
    for f in all_csvs:
        name = f.stem
        # Parse: {model}_{ticker}_h{horizon}
        parts = name.rsplit("_h", 1)
        if len(parts) != 2:
            continue
        # Skip context-sensitivity files (e.g., _h1_ctx128)
        try:
            h = int(parts[1])
        except ValueError:
            continue
        # model_ticker part — ticker is last token after model name
        mt = parts[0]
        # Find ticker: try matching known tickers from the end
        ticker = None
        for t in VOLARE_ALL_TICKERS:
            if mt.endswith(f"_{t}"):
                ticker = t
                model = mt[: -(len(t) + 1)]
                break
        if ticker is None:
            continue  # skip unknown tickers
        model_ticker_horizon[(model, ticker, h)] = f

    print(f"Found {len(model_ticker_horizon)} equity forecast files")

    # Compute per-asset subsample metrics
    rows = []
    for (model, ticker, h), fpath in sorted(model_ticker_horizon.items()):
        df = pd.read_csv(fpath, parse_dates=["date"])
        pre = df[df["date"] < SPLIT_DATE]
        post = df[df["date"] >= SPLIT_DATE]

        for period, sub in [("pre-COVID", pre), ("post-COVID", post)]:
            if len(sub) < 10:
                continue
            metrics = compute_metrics(sub["actual"].values, sub["forecast"].values)
            metrics["model"] = model
            metrics["ticker"] = ticker
            metrics["horizon"] = h
            metrics["period"] = period
            metrics["n_obs"] = len(sub)
            rows.append(metrics)

    per_asset = pd.DataFrame(rows)
    print(f"Computed {len(per_asset)} per-asset subsample entries")

    # Aggregate across 40 equities (mean)
    agg_rows = []
    for model in MODEL_ORDER:
        for h in HORIZONS:
            for period in ["pre-COVID", "post-COVID"]:
                mask = (
                    (per_asset["model"] == model)
                    & (per_asset["horizon"] == h)
                    & (per_asset["period"] == period)
                )
                sub = per_asset[mask]
                if len(sub) == 0:
                    continue
                agg = {
                    "model": model,
                    "horizon": h,
                    "period": period,
                    "MSE": sub["MSE"].mean(),
                    "MAE": sub["MAE"].mean(),
                    "QLIKE": sub["QLIKE"].mean(),
                    "R2OOS": sub["R2OOS"].mean(),
                    "n_tickers": len(sub),
                }
                agg_rows.append(agg)

    agg_df = pd.DataFrame(agg_rows)
    out_path = METRICS_DIR / "subsample_metrics.csv"
    agg_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(agg_df)} rows, {agg_df['model'].nunique()} models)")

    # Generate LaTeX table
    generate_table(agg_df)


def generate_table(agg_df):
    """Generate table_subsample.tex: two panels (Pre-COVID, Post-COVID), each
    with the three horizons side by side (Model + MSE h=1,5,22 + QLIKE
    h=1,5,22). Bold marks the lowest MSE/QLIKE per column within each panel;
    $\\dagger$ marks QLIKE>1. Plain (non-longtable) \\small table; fits a page."""
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
        r"\caption{Sub-sample forecast accuracy: pre-COVID (2015--2020) and "
        r"post-COVID (2020--2026) periods across all 50 assets (VOLARE). MSE "
        r"($\times 10^{-6}$) on the volatility scale; QLIKE on the variance "
        r"scale. Bold marks the lowest MSE and lowest QLIKE in each horizon "
        r"column within each panel. $\dagger$ marks QLIKE $>1$.}",
        r"\label{tab:subsample}", r"\small",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"& \multicolumn{3}{c}{MSE ($\times 10^{-6}$)} & \multicolumn{3}{c}{QLIKE} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\",
    ]

    period_label = {"pre-COVID": "Panel A: Pre-COVID (2015--2020)",
                    "post-COVID": "Panel B: Post-COVID (2020--2026)"}

    for period in ["pre-COVID", "post-COVID"]:
        mse = {}; qlike = {}; models = None
        for h in HORIZONS:
            sub = agg_df[(agg_df["horizon"] == h) & (agg_df["period"] == period)]
            sub = sub.set_index("model")
            avail = [m for m in MODEL_ORDER if m in sub.index]
            sub = sub.reindex(avail)
            mse[h] = sub["MSE"] * 1e6
            qlike[h] = sub["QLIKE"]
            if models is None:
                models = avail

        mse_best = {h: mse[h].idxmin() for h in HORIZONS}
        qlike_best = {}
        for h in HORIZONS:
            valid = qlike[h][qlike[h] < 1.0]
            qlike_best[h] = valid.idxmin() if len(valid) > 0 else None

        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{period_label[period]}}}}} \\[2pt]")
        lines.append(r"\midrule")

        for model in models:
            cells = [MODEL_DISPLAY.get(model, model)]
            for h in HORIZONS:
                s = f"{mse[h][model]:.3f}"
                if model == mse_best[h]:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            for h in HORIZONS:
                qv = qlike[h][model]
                if qv > 1.0:
                    s = f"{qv:.3f}$^{{\\dagger}}$"
                else:
                    s = f"{qv:.3f}"
                if model == qlike_best[h]:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    out = TABLE_DIR / "table_subsample.tex"
    out.write_text(tex)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
