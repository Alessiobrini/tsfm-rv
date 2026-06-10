"""gen_per_asset_qlike.py — Per-asset QLIKE appendix tables (h = 1, 5, 22).

Reads results/volare/metrics/metrics_by_asset_h{h}.csv (per-asset QLIKE, already on
the variance scale via run_evaluation_volare.py --scale vol) and writes
paper/tables/table_per_asset_qlike{,_h5,_h22}.tex: a 12-model x 50-asset grid grouped
by asset class, with the lowest QLIKE per asset bolded.

Usage:  python code/gen_per_asset_qlike.py
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

COLS = [("HAR", "HAR"), ("Log_HAR", "Log-HAR"), ("ARFIMA", "ARFIMA"),
        ("chronos_bolt_small", "Chr-Bolt-S"), ("chronos_bolt_base", "Chr-Bolt-B"),
        ("moirai_2_0_small", "Moirai-S"), ("moirai_moe_small", "MoE-S"),
        ("lag_llama", "Lag-Llama"), ("timesfm_2_5", "TFM-2.5"),
        ("toto", "Toto"), ("sundial", "Sundial"), ("ttm", "TTM")]
KEYS = [k for k, _ in COLS]
DISP = [d for _, d in COLS]
FUT = {"C": "Corn", "CL": "Crude Oil", "ES": "E-mini S\\&P", "GC": "Gold", "NG": "Nat. Gas"}
CLASSES = [("Stocks", VOLARE_STOCK_TICKERS), ("FX", VOLARE_FX_TICKERS),
           ("Futures", VOLARE_FUTURES_TICKERS)]


def build(h, suffix):
    df = pd.read_csv(MET / f"metrics_by_asset_h{h}.csv")
    mcol = "model" if "model" in df.columns else df.columns[0]
    piv = df.pivot_table(index="ticker", columns=mcol, values="QLIKE")

    L = [r"\begin{table}[p]", r"\centering", r"\singlespacing",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "r" * len(KEYS) + "}",
         r"\toprule", "Asset & " + " & ".join(DISP) + r" \\", r"\midrule"]
    for cname, tickers in CLASSES:
        L.append(rf"\multicolumn{{{len(KEYS)+1}}}{{l}}{{\textit{{{cname}}}}} \\")
        for t in tickers:
            if t not in piv.index:
                continue
            vals = [piv.loc[t, k] if k in piv.columns else np.nan for k in KEYS]
            fin = [v for v in vals if np.isfinite(v)]
            bestv = min(fin) if fin else None
            cells = []
            for v in vals:
                if not np.isfinite(v):
                    cells.append("--"); continue
                s = f"{v:.4f}" if v < 10 else f"{v:.1f}"
                if bestv is not None and abs(v - bestv) < 1e-12:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            L.append(f"{FUT.get(t, t)} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}}",
          rf"\caption{{Per-asset QLIKE at $h={h}$ across 50 assets (volatility forecasts; "
          rf"QLIKE on the variance scale). Bold marks the lowest QLIKE per asset among the "
          rf"displayed models.}}",
          rf"\label{{tab:per_asset_qlike{suffix}}}", r"\end{table}"]
    (TAB / f"table_per_asset_qlike{suffix}.tex").write_text("\n".join(L))
    print(f"  wrote table_per_asset_qlike{suffix}.tex ({piv.shape[0]} assets)")


if __name__ == "__main__":
    for h, suf in [(1, ""), (5, "_h5"), (22, "_h22")]:
        build(h, suf)
