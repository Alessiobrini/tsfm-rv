"""gen_context_sensitivity.py — Context-length sensitivity table (audit B6).

Aggregates the foundation-model context-length sweep into
paper/tables/table_context_sensitivity.tex under the *revised* pipeline (point
target, mean forecast, volatility scale, QLIKE on the variance scale).

Inputs (all in results/volare/forecasts/):
  - context lengths 128/256/512: files named  {model}_{ticker}_h{h}_ctx{N}.csv
    (produced by cluster/run_rev_context_sensitivity.slurm).
  - context length 1000 (the default): the main files {model}_{ticker}_h{h}.csv.

For each (model, horizon, context) it computes per-asset QLIKE on the variance
scale, then averages across the 50 assets, and bolds the best context per
model-horizon. Writes the table only if every context column is populated;
otherwise it prints what is missing and exits without overwriting, so the table
is never silently built from a partial sweep.

Usage:  python code/gen_context_sensitivity.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS
from evaluation.loss_functions import qlike

FC = ROOT / "results" / "volare" / "forecasts"
TAB = ROOT / "paper" / "tables"
ALL = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
HORIZONS = [1, 5, 22]
CONTEXTS = [128, 256, 512, 1000]   # 1000 = default (main run, no suffix)

MODELS = [("chronos_bolt_small", "Chronos-Bolt-S"), ("chronos_bolt_base", "Chronos-Bolt-B"),
          ("timesfm_2_5", "TimesFM-2.5"), ("moirai_2_0_small", "Moirai-2.0-S"),
          ("moirai_moe_small", "Moirai-MoE-S"), ("lag_llama", "Lag-Llama"),
          ("toto", "Toto"), ("sundial", "Sundial"), ("ttm", "TTM")]

# Models architecturally capped at a 512-token context (cannot run ctx=1000):
# TTM's r2.1 branch maxes at 512; Moirai-MoE's positional encoding is fixed at 512.
# For these, ctx=1000 is not applicable and their 512 column is the paper default.
CAPPED_512 = {"ttm", "moirai_moe_small"}


def mean_qlike(model, h, ctx):
    """Mean across assets of per-asset QLIKE (variance scale) at this context."""
    vals = []
    for t in ALL:
        suffix = "" if ctx == 1000 else f"_ctx{ctx}"
        f = FC / f"{model}_{t}_h{h}{suffix}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if {"actual", "forecast"} <= set(d.columns) and len(d) > 0:
            vals.append(qlike(d["actual"].values, d["forecast"].values, scale="vol"))
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def main():
    grid = {}   # (model, h, ctx) -> mean qlike
    missing = []
    for key, _ in MODELS:
        for h in HORIZONS:
            for ctx in CONTEXTS:
                if ctx == 1000 and key in CAPPED_512:
                    grid[(key, h, ctx)] = np.nan  # not applicable: capped at 512
                    continue
                m, n = mean_qlike(key, h, ctx)
                grid[(key, h, ctx)] = m
                if n < len(ALL):
                    missing.append(f"{key} h={h} ctx={ctx}: {n}/{len(ALL)} assets")
    if missing:
        print("INCOMPLETE context sweep -- table NOT written. Missing/partial:")
        for s in missing[:30]:
            print("  " + s)
        print(f"  ... {len(missing)} (model,h,ctx) cells incomplete in total.")
        print("Run cluster/run_rev_context_sensitivity.slurm and pull results, then re-run.")
        return

    L = [r"\begin{table}[H]", r"\centering", r"\singlespacing", r"\small",
         r"\begin{tabular}{ll" + "r" * len(CONTEXTS) + "}", r"\toprule",
         "Model & $h$ & " + " & ".join(f"ctx={c}" for c in CONTEXTS) + r" \\", r"\midrule"]
    for key, disp in MODELS:
        for i, h in enumerate(HORIZONS):
            row = [grid[(key, h, c)] for c in CONTEXTS]
            best = np.nanmin(row)
            cells = []
            for v in row:
                if np.isnan(v):
                    cells.append("--")
                    continue
                s = f"{v:.3f}"
                if abs(v - best) < 1e-9:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            name = disp if i == 0 else ""
            L.append(f"{name} & {h} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Context-length sensitivity of TSFM forecasts across 50 assets "
          r"(40 equities, 5 FX, 5 futures), revised pipeline (point-in-time target, mean "
          r"forecast, volatility scale). QLIKE on the variance scale, averaged across assets, "
          r"by horizon and context length; ctx$=$1{,}000 is the default for all models except "
          r"TTM and Moirai-MoE, which are architecturally capped at a 512-token context "
          r"(ctx$=$1{,}000 not available, marked --) and use 512 as their default. Bold marks "
          r"the best available context length for each model--horizon pair.}",
          r"\label{tab:context_sensitivity}", r"\end{table}"]
    (TAB / "table_context_sensitivity.tex").write_text("\n".join(L))
    print("wrote table_context_sensitivity.tex")


if __name__ == "__main__":
    main()
