"""gen_pooled50.py — All-50-asset pooled MSE / QLIKE table.

Section 5.1 contrasts the *pooled* cross-sectional average (which a few
high-volatility assets dominate) with the equal-weighted loss ratio
(Tab. loss_ratios). The pooled discussion therefore needs an explicit
all-50-asset table on the same 50-asset basis as the loss ratios, rather than
citing the 40-equity table (the source of audit item B1). This builds it from
results/volare/metrics/metrics_by_asset_h{h}.csv by averaging each model's
per-asset loss over all 50 assets (40 equities, 5 FX, 5 futures).

Columns: pooled mean MSE at h = 1, 5, 22 and pooled mean QLIKE at the same
horizons. The lowest MSE and lowest QLIKE per horizon are bolded.

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


MSE_SCALE = 1e6   # pooled MSE is ~3e-5 to 9e-5; display as MSE (x 10^-6)


def main():
    # pooled mean per (model, horizon) across all 50 assets
    mse = {}     # (model, h) -> mean MSE (scaled)
    qlike = {}   # (model, h) -> mean QLIKE
    for h in HORIZONS:
        df = pd.read_csv(MET / f"metrics_by_asset_h{h}.csv")
        df = df[df["ticker"].isin(ALL)]
        g = df.groupby("model")[["MSE", "QLIKE"]].mean()
        for key, _ in ORDER:
            if key in g.index:
                mse[(key, h)] = float(g.loc[key, "MSE"]) * MSE_SCALE
                qlike[(key, h)] = float(g.loc[key, "QLIKE"])

    # best per column (min MSE, min QLIKE) for bolding
    mmin = {h: min(mse[(k, h)] for k, _ in ORDER if (k, h) in mse) for h in HORIZONS}
    qmin = {h: min(qlike[(k, h)] for k, _ in ORDER if (k, h) in qlike) for h in HORIZONS}

    L = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
         r"\caption{Pooled forecast accuracy across all 50 assets (40 equities, 5 FX, "
         r"5 futures). Each cell is the simple average of the per-asset loss over the 50 "
         r"assets; MSE is on the volatility scale and QLIKE is on the variance scale. The "
         r"pooled mean is dominated by a few high-volatility assets and is reported here "
         r"only as the naive aggregate that Table~\ref{tab:loss_ratios} corrects. Bold "
         r"marks the lowest MSE and lowest QLIKE in each horizon column.}",
         r"\label{tab:pooled50}", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{MSE ($\times 10^{-6}$)} & \multicolumn{3}{c}{QLIKE} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule"]
    for key, disp in ORDER:
        cells = [disp]
        for h in HORIZONS:
            v = mse.get((key, h))
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and abs(v - mmin[h]) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        for h in HORIZONS:
            v = qlike.get((key, h))
            s = "--" if v is None else f"{v:.3f}"
            if v is not None and abs(v - qmin[h]) < 1e-9:
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


if __name__ == "__main__":
    main()
