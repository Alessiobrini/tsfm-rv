"""gen_pooled50.py — All-50-asset pooled QLIKE / R2OOS table.

Section 5.1 contrasts the *pooled* cross-sectional average (which a few
high-volatility assets dominate) with the equal-weighted loss ratio
(Tab. loss_ratios). The pooled discussion therefore needs an explicit
all-50-asset table on the same 50-asset basis as the loss ratios, rather than
citing the 40-equity table (the source of audit item B1). This builds it from
results/volare/metrics/metrics_by_asset_h{h}.csv by averaging each model's
per-asset loss over all 50 assets (40 equities, 5 FX, 5 futures).

Columns: pooled mean QLIKE at h = 1, 5, 22 and pooled mean R2OOS at the same
horizons. The lowest QLIKE and highest R2OOS per horizon are bolded.

Usage:  python code/gen_pooled50.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS

MET = ROOT / "results" / "volare" / "metrics"
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


def main():
    # pooled mean per (model, horizon) across all 50 assets
    qlike = {}   # (model, h) -> mean QLIKE
    r2 = {}      # (model, h) -> mean R2OOS
    for h in HORIZONS:
        df = pd.read_csv(MET / f"metrics_by_asset_h{h}.csv")
        df = df[df["ticker"].isin(ALL)]
        g = df.groupby("model")[["QLIKE", "R2OOS"]].mean()
        for key, _ in ORDER:
            if key in g.index:
                qlike[(key, h)] = float(g.loc[key, "QLIKE"])
                r2[(key, h)] = float(g.loc[key, "R2OOS"])

    # best per column (min QLIKE, max R2) for bolding
    qmin = {h: min(qlike[(k, h)] for k, _ in ORDER if (k, h) in qlike) for h in HORIZONS}
    rmax = {h: max(r2[(k, h)] for k, _ in ORDER if (k, h) in r2) for h in HORIZONS}

    L = [r"\begin{table}[H]", r"\centering", r"\singlespacing",
         r"\caption{Pooled forecast accuracy across all 50 assets (40 equities, 5 FX, "
         r"5 futures). Each cell is the simple average of the per-asset loss over the 50 "
         r"assets; QLIKE is on the variance scale. The pooled mean is dominated by a few "
         r"high-volatility assets and is reported here only as the naive aggregate that "
         r"Table~\ref{tab:loss_ratios} corrects. Bold marks the lowest QLIKE and highest "
         r"$R^2_{\mathrm{OOS}}$ in each horizon column.}",
         r"\label{tab:pooled50}", r"\small",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{QLIKE} & \multicolumn{3}{c}{$R^2_{\mathrm{OOS}}$} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule"]
    for key, disp in ORDER:
        cells = [disp]
        for h in HORIZONS:
            v = qlike.get((key, h))
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and abs(v - qmin[h]) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        for h in HORIZONS:
            v = r2.get((key, h))
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and abs(v - rmax[h]) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_pooled50.tex").write_text("\n".join(L))
    print("wrote table_pooled50.tex")
    # echo the QLIKE orderings for text verification
    for h in HORIZONS:
        s = sorted(((qlike[(k, h)], d) for k, d in ORDER if (k, h) in qlike))
        print(f"h={h}: " + ", ".join(f"{d} {v:.3f}" for v, d in s[:8]))
    for h in HORIZONS:
        print(f"Toto R2 mean h={h}: {r2[('toto', h)]:.3f}")


if __name__ == "__main__":
    main()
