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

from config import VOLARE_STOCK_TICKERS
from evaluation.loss_functions import mse, mae, qlike, r2_oos

FORECAST_DIR = PROJECT_ROOT / "results" / "volare" / "forecasts"
METRICS_DIR = PROJECT_ROOT / "results" / "volare" / "metrics"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
SPLIT_DATE = "2020-03-01"
HORIZONS = [1, 5, 22]

MODEL_DISPLAY = {
    "HAR": "HAR", "HAR_J": "HAR-J", "HAR_RS": "HAR-RS", "HARQ": "HARQ",
    "Log_HAR": "Log-HAR", "ARFIMA": "ARFIMA",
    "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S", "lag_llama": "Lag-Llama",
    "toto": "Toto", "sundial": "Sundial", "moirai_moe_small": "Moirai-MoE-S",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())


def compute_metrics(actual, forecast):
    """Compute all four loss functions."""
    actual = np.asarray(actual)
    forecast = np.maximum(np.asarray(forecast), 1e-6)
    return {
        "MSE": mse(actual, forecast),
        "MAE": mae(actual, forecast),
        "QLIKE": qlike(actual, forecast),
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
        h = int(parts[1])
        # model_ticker part — ticker is last token after model name
        mt = parts[0]
        # Find ticker: try matching known tickers from the end
        ticker = None
        for t in VOLARE_STOCK_TICKERS:
            if mt.endswith(f"_{t}"):
                ticker = t
                model = mt[: -(len(t) + 1)]
                break
        if ticker is None:
            continue  # skip non-equity
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
    """Generate table_subsample.tex with all 11 models."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sub-sample forecast accuracy: pre-COVID (2015--2020) and post-COVID (2020--2026) periods for 40 U.S.\ equities (VOLARE).}")
    lines.append(r"\label{tab:subsample}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l cccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & MSE ($\times 10^{-6}$) & MAE ($\times 10^{-4}$) & QLIKE & $R^2_{\mathrm{OOS}}$ \\")
    lines.append(r"\midrule")

    for h in HORIZONS:
        for period in ["pre-COVID", "post-COVID"]:
            label = f"$h={h}$, {'Pre' if 'pre' in period else 'Post'}-COVID"
            lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Panel: {label}}}}} \\")
            lines.append(r"\addlinespace")

            sub = agg_df[(agg_df["horizon"] == h) & (agg_df["period"] == period)]
            sub = sub.set_index("model")
            available = [m for m in MODEL_ORDER if m in sub.index]
            sub = sub.reindex(available)

            if len(sub) == 0:
                lines.append(r"--- & --- & --- & --- & --- \\")
            else:
                mse_vals = sub["MSE"] * 1e6
                mae_vals = sub["MAE"] * 1e4
                qlike_vals = sub["QLIKE"]
                r2_vals = sub["R2OOS"]

                mse_best = mse_vals.idxmin()
                mae_best = mae_vals.idxmin()
                qlike_valid = qlike_vals[qlike_vals < 1.0]
                qlike_best = qlike_valid.idxmin() if len(qlike_valid) > 0 else None
                r2_best = r2_vals.idxmax()

                for model in available:
                    name = MODEL_DISPLAY.get(model, model)
                    ms = f"{mse_vals[model]:.3f}"
                    ma = f"{mae_vals[model]:.3f}"
                    qv = qlike_vals[model]
                    rv = r2_vals[model]

                    if qv > 1.0:
                        qs = f"{qv:.3f}$^{{\\dagger}}$"
                    else:
                        qs = f"{qv:.3f}"
                    rs = f"{rv:.3f}"

                    if model == mse_best:
                        ms = rf"\textbf{{{ms}}}"
                    if model == mae_best:
                        ma = rf"\textbf{{{ma}}}"
                    if model == qlike_best:
                        qs = rf"\textbf{{{qs}}}"
                    if model == r2_best:
                        rs = rf"\textbf{{{rs}}}"

                    lines.append(f"{name} & {ms} & {ma} & {qs} & {rs} \\\\")

            lines.append(r"\addlinespace")
            lines.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out = TABLE_DIR / "table_subsample.tex"
    out.write_text(tex)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
