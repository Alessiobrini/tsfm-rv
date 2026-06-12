"""gen_avg_target_table.py — Averaged-target robustness appendix table (audit B9).

Builds the appendix table for the h-day-AVERAGE realized-volatility target
(Patton & Sheppard 2015) from results/volare_avg/forecasts/, so the robustness
section can show the headline ranking is reproduced under the alternative target.

It mirrors the main loss-ratio table (Table loss_ratios): per asset it computes
QLIKE on the variance scale, forms each model's QLIKE ratio to Log-HAR, and
averages those ratios across the 50 assets (an aggregation robust to outliers).
Values below 1 beat Log-HAR on average. Writes paper/tables/table_avg_target.tex
and prints the ranking for the text.

Usage:  python code/gen_avg_target_table.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS
from evaluation.loss_functions import qlike

FC = ROOT / "results" / "volare_avg" / "forecasts"
TAB = ROOT / "paper" / "tables"
ALL = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
HORIZONS = [1, 5, 22]

ORDER = [("Log_HAR", "Log-HAR"), ("HAR", "HAR"), ("HAR_J", "HAR-J"),
         ("HAR_RS", "HAR-RS"), ("HARQ", "HARQ"), ("ARFIMA", "ARFIMA"),
         ("ARMA", "ARMA"), ("MEM", "MEM"),
         ("chronos_bolt_small", "Chronos-Bolt-S"), ("chronos_bolt_base", "Chronos-Bolt-B"),
         ("moirai_2_0_small", "Moirai-2.0-S"), ("moirai_moe_small", "Moirai-MoE-S"),
         ("lag_llama", "Lag-Llama"), ("timesfm_2_5", "TimesFM-2.5"),
         ("toto", "Toto"), ("sundial", "Sundial"), ("ttm", "TTM")]


def per_asset_qlike(model, h):
    """{ticker: QLIKE} for this model/horizon from the averaged-target forecasts."""
    out = {}
    for t in ALL:
        f = FC / f"{model}_{t}_h{h}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if {"actual", "forecast"} <= set(d.columns) and len(d):
            out[t] = qlike(d["actual"].values, d["forecast"].values, scale="vol")
    return out


def main():
    # loss ratio vs Log-HAR, averaged across assets, per (model, horizon)
    ratios = {}      # (model, h) -> mean ratio
    missing = []
    base = {h: per_asset_qlike("Log_HAR", h) for h in HORIZONS}
    for h in HORIZONS:
        if len(base[h]) < len(ALL):
            missing.append(f"Log_HAR h={h}: {len(base[h])}/{len(ALL)}")
    for key, _ in ORDER:
        for h in HORIZONS:
            q = per_asset_qlike(key, h)
            if len(q) < len(ALL):
                missing.append(f"{key} h={h}: {len(q)}/{len(ALL)}")
            common = [t for t in ALL if t in q and t in base[h] and base[h][t] > 0]
            rs = [q[t] / base[h][t] for t in common]
            ratios[(key, h)] = float(np.mean(rs)) if rs else np.nan
    if missing:
        print("INCOMPLETE averaged-target results -- table NOT written. Missing/partial:")
        for s in missing[:30]:
            print("  " + s)
        return

    best = {h: min(ratios[(k, h)] for k, _ in ORDER) for h in HORIZONS}
    L = [r"\begin{table}[H]", r"\centering", r"\singlespacing",
         r"\caption{Averaged-target robustness: average QLIKE loss ratios relative to "
         r"Log-HAR across all 50 assets, under the $h$-day-\emph{average} realized-volatility "
         r"target (Patton \& Sheppard 2015) rather than the point-in-time target of the main "
         r"results. Per asset, each model's QLIKE (variance scale) is divided by Log-HAR's and "
         r"the ratios are averaged across assets; values below 1 beat Log-HAR. Compare with "
         r"Table~\ref{tab:loss_ratios} (point-in-time target).}",
         r"\label{tab:avg_target}", r"\small",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule"]
    for key, disp in ORDER:
        cells = []
        for h in HORIZONS:
            v = ratios[(key, h)]
            s = f"{v:.3f}"
            if key != "Log_HAR" and abs(v - best[h]) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        L.append(f"{disp} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_avg_target.tex").write_text("\n".join(L))
    print("wrote table_avg_target.tex")
    # ranking echo for the text
    for h in HORIZONS:
        beat = sorted((ratios[(k, h)], d) for k, d in ORDER if k != "Log_HAR" and ratios[(k, h)] < 1.0)
        print(f"h={h}: models beating Log-HAR (ratio<1): " + (", ".join(f"{d} {v:.3f}" for v, d in beat) or "none"))


if __name__ == "__main__":
    main()
