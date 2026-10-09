"""
compute_subsample_metrics.py — Compute pre/post-COVID forecast metrics from existing CSVs.

Reads the forecast CSVs of the paper's 17 models from results/volare/forecasts/, keeps the dates
common to the 17 models of each asset and horizon (the sample of the main tables), splits them at
2020-03-01, computes MSE/MAE/QLIKE/R2OOS per (model, asset, horizon, period), and averages across
the 50 assets. Saves subsample_metrics.csv and regenerates the LaTeX subsample table.

Usage:
    python compute_subsample_metrics.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import VOLARE_ALL_TICKERS, PAPER_MODELS
from evaluation.loss_functions import mse, mae, qlike, r2_oos
from evaluation.targets import common_dates

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
MODEL_ORDER = list(PAPER_MODELS)


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


def per_asset_subsample(fc_dir, tickers=VOLARE_ALL_TICKERS, horizons=HORIZONS, models=MODEL_ORDER):
    """Per-asset metrics before and after SPLIT_DATE, on the dates common to all models."""
    rows = []
    for h in horizons:
        for ticker in tickers:
            paths = {m: fc_dir / f"{m}_{ticker}_h{h}.csv" for m in models}
            missing = [m for m, p in paths.items() if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{ticker} h={h}: no forecast file for {missing}")
            frames = {m: pd.read_csv(p, index_col=0, parse_dates=True) for m, p in paths.items()}
            idx = common_dates(frames)
            actual = frames[models[0]].loc[idx, "actual"]
            for m, d in frames.items():
                if not np.allclose(d.loc[idx, "actual"].to_numpy(), actual.to_numpy(), rtol=1e-10, atol=0):
                    raise ValueError(f"{ticker} h={h}: {m} has other actuals on the common dates")
            for period, mask in [("pre-COVID", idx < SPLIT_DATE), ("post-COVID", idx >= SPLIT_DATE)]:
                if mask.sum() < 10:
                    continue
                for m, d in frames.items():
                    metrics = compute_metrics(actual[mask].to_numpy(), d.loc[idx[mask], "forecast"].to_numpy())
                    metrics.update(model=m, ticker=ticker, horizon=h, period=period, n_obs=int(mask.sum()))
                    rows.append(metrics)
    return pd.DataFrame(rows)


def main():
    per_asset = per_asset_subsample(FORECAST_DIR)
    print(f"Computed {len(per_asset)} per-asset subsample entries")

    # Aggregate across the 50 assets (mean)
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
        r"\caption{Sub-sample forecast accuracy before and from 1 March 2020, averaged "
        r"across the 50 assets, on the dates common to the 17 models of each asset. MSE "
        r"($\times 10^{-6}$) is on the volatility scale and QLIKE on the variance scale. "
        r"Bold marks the lowest MSE and lowest QLIKE in each horizon column within each "
        r"panel, and $\dagger$ marks QLIKE $>1$.}",
        r"\label{tab:subsample}", r"\small",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"& \multicolumn{3}{c}{MSE ($\times 10^{-6}$)} & \multicolumn{3}{c}{QLIKE} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\",
    ]

    period_label = {"pre-COVID": "Panel A: before March 2020",
                    "post-COVID": "Panel B: March 2020 to January 2026"}

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
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "table_subsample.tex"
    out.write_text(tex)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
