"""
compute_context_sensitivity.py — Aggregate TSFM forecasts across context lengths
and produce a LaTeX table for the robustness section.

Usage:
    python compute_context_sensitivity.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VOLARE_RESULTS_DIR, VOLARE_STOCK_TICKERS,
    VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)
from evaluation.loss_functions import compute_all_losses

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
METRICS_DIR = VOLARE_RESULTS_DIR / "metrics"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"

CONTEXT_LENGTHS = [128, 256, 512]
HORIZONS = [1, 5, 22]
TSFM_MODELS = [
    "chronos_bolt_small",
    "chronos_bolt_base",
    "moirai_2_0_small",
    "moirai_moe_small",
    "lag_llama",
    "timesfm_2_5",
    "toto",
    "sundial",
    "ttm",
]
MODEL_DISPLAY = {
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S",
    "moirai_moe_small": "Moirai-MoE-S",
    "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto",
    "sundial": "Sundial",
    "ttm": "TTM",
}

ALL_TICKERS = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS


def load_forecast(model, ticker, horizon, ctx):
    """Load a forecast CSV for a given context length."""
    if ctx == 512:
        fname = f"{model}_{ticker}_h{horizon}.csv"
    else:
        fname = f"{model}_{ticker}_h{horizon}_ctx{ctx}.csv"
    fpath = FORECAST_DIR / fname
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath, index_col=0)
    return df


def main():
    print("Computing context-length sensitivity metrics...")

    rows = []
    for model in TSFM_MODELS:
        for ctx in CONTEXT_LENGTHS:
            for h in HORIZONS:
                asset_metrics = []
                for ticker in ALL_TICKERS:
                    df = load_forecast(model, ticker, h, ctx)
                    if df is None:
                        continue
                    actual = df["actual"]
                    forecast = df["forecast"].clip(lower=1e-6)
                    m = compute_all_losses(actual, forecast)
                    m["ticker"] = ticker
                    asset_metrics.append(m)

                if not asset_metrics:
                    continue

                adf = pd.DataFrame(asset_metrics)
                row = {
                    "model": model,
                    "context_length": ctx,
                    "horizon": h,
                    "n_assets": len(adf),
                    "MSE": adf["MSE"].mean(),
                    "MAE": adf["MAE"].mean(),
                    "QLIKE": adf["QLIKE"].mean(),
                    "R2OOS": adf["R2OOS"].mean(),
                }
                rows.append(row)
                print(f"  {MODEL_DISPLAY[model]:18s} ctx={ctx:3d} h={h:2d}  "
                      f"QLIKE={row['QLIKE']:.3f}  R2OOS={row['R2OOS']:.3f}  "
                      f"({row['n_assets']} assets)")

    if not rows:
        print("No forecast files found for any non-default context lengths.")
        print("Run: python run_foundation_volare.py --context-length 128 --all-tickers")
        print("     python run_foundation_volare.py --context-length 256 --all-tickers")
        return

    results = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "context_sensitivity_metrics.csv", index=False)
    print(f"\nMetrics saved to {METRICS_DIR / 'context_sensitivity_metrics.csv'}")

    # Generate LaTeX table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex = generate_latex_table(results)
    out_path = TABLE_DIR / "table_context_sensitivity.tex"
    out_path.write_text(tex)
    print(f"LaTeX table saved to {out_path}")


def generate_latex_table(df):
    """Generate a LaTeX table showing QLIKE and R2OOS by model, horizon, context length."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\singlespacing")
    lines.append("\\caption{Context-length sensitivity of TSFM forecasts. "
                 "QLIKE and $R^2_{\\text{OOS}}$ averaged across all assets, by horizon and context window length. "
                 "Bold indicates the best context length for each model--horizon pair.}")
    lines.append("\\label{tab:context_sensitivity}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append(" & & \\multicolumn{3}{c}{QLIKE} & \\multicolumn{3}{c}{$R^2_{\\text{OOS}}$} \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}")
    lines.append("Model & $h$ & ctx=128 & ctx=256 & ctx=512 & ctx=128 & ctx=256 & ctx=512 \\\\")
    lines.append("\\midrule")

    models_present = [m for m in TSFM_MODELS if m in df["model"].values]

    for model in models_present:
        name = MODEL_DISPLAY[model]
        for h in HORIZONS:
            sub = df[(df["model"] == model) & (df["horizon"] == h)]
            if len(sub) == 0:
                continue

            qlike_vals = {}
            r2_vals = {}
            for _, row in sub.iterrows():
                ctx = int(row["context_length"])
                qlike_vals[ctx] = row["QLIKE"]
                r2_vals[ctx] = row["R2OOS"]

            # Find best (lowest QLIKE, highest R2OOS)
            best_qlike_ctx = min(qlike_vals, key=qlike_vals.get) if qlike_vals else None
            best_r2_ctx = max(r2_vals, key=r2_vals.get) if r2_vals else None

            cells = []
            for ctx in CONTEXT_LENGTHS:
                if ctx in qlike_vals:
                    s = f"{qlike_vals[ctx]:.3f}"
                    if ctx == best_qlike_ctx and len(qlike_vals) > 1:
                        s = f"\\textbf{{{s}}}"
                else:
                    s = "---"
                cells.append(s)
            for ctx in CONTEXT_LENGTHS:
                if ctx in r2_vals:
                    s = f"{r2_vals[ctx]:.3f}"
                    if ctx == best_r2_ctx and len(r2_vals) > 1:
                        s = f"\\textbf{{{s}}}"
                else:
                    s = "---"
                cells.append(s)

            h_label = str(h) if h == HORIZONS[0] else str(h)
            model_label = name if h == HORIZONS[0] else ""
            lines.append(f"{model_label} & {h_label} & {' & '.join(cells)} \\\\")

        if model != models_present[-1]:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
