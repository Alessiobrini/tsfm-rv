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
from functools import lru_cache

from config import VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS, PAPER_MODELS
from evaluation.loss_functions import qlike
from evaluation.targets import common_dates

FC = ROOT / "results" / "volare" / "forecasts"
TAB = ROOT / "paper" / "tables"
ALL = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
HORIZONS = [1, 5, 22]
CONTEXTS = [128, 256, 512, 1000]   # 1000 = default (main run, no suffix)

MODELS = [("chronos_bolt_small", "Chronos-Bolt-S"), ("chronos_bolt_base", "Chronos-Bolt-B"),
          ("timesfm_2_5", "TimesFM-2.5"), ("moirai_2_0_small", "Moirai-2.0-S"),
          ("moirai_moe_small", "Moirai-MoE-S"), ("lag_llama", "Lag-Llama"),
          ("toto", "Toto"), ("sundial", "Sundial"), ("ttm", "TTM")]

# TTM's r2.1 release supports at most a 512-day context, so ctx=1000 is not applicable to it and
# its 512 column is the paper default. Every other model, Moirai-MoE included, runs at 1,000 days.
CAPPED_512 = {"ttm"}


@lru_cache(maxsize=None)
def ref_dates(h, ticker):
    """Dates common to the 17 models' main files for this asset and horizon, the sample of the
    main tables.

    Every context length is scored on this one per-asset date set, which isolates the effect of
    context length from the sample period (shorter contexts otherwise start earlier and would be
    scored on a longer window), and the default-context column then uses the headline sample.
    """
    frames = {}
    for m in PAPER_MODELS:
        f = FC / f"{m}_{ticker}_h{h}.csv"
        if not f.exists():
            return None
        frames[m] = pd.read_csv(f, index_col=0, parse_dates=True)
    return common_dates(frames)


def mean_qlike(model, h, ctx):
    """Mean across assets of per-asset QLIKE (variance scale) at this context, evaluated on the
    dates common to the 17 models' main files."""
    vals = []
    for t in ALL:
        suffix = "" if ctx == 1000 else f"_ctx{ctx}"
        f = FC / f"{model}_{t}_h{h}{suffix}.csv"
        if not f.exists():
            continue
        rd = ref_dates(h, t)
        if rd is None:
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        absent = rd.difference(d["forecast"].dropna().index)
        if len(absent):
            raise ValueError(f"{f.name}: no forecast on {len(absent)} headline dates")
        d = d.loc[rd]
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

    L = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\small",
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
          r"(40 equities, 5 FX, 5 futures), evaluated under the point-in-time target, mean "
          r"forecast, and volatility scale. QLIKE on the variance scale, averaged across assets, "
          r"by horizon and context length; ctx$=$1{,}000 is the default for all models except "
          r"TTM, whose r2.1 release supports at most 512 days (ctx$=$1{,}000 not available, "
          r"marked --) and which uses 512 as its default. All context "
          r"lengths are scored on the common out-of-sample window of the default run, so "
          r"differences reflect context length rather than sample period. Bold marks "
          r"the best available context length for each model--horizon pair.}",
          r"\label{tab:context_sensitivity}", r"\end{table}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table_context_sensitivity.tex").write_text("\n".join(L))
    print("wrote table_context_sensitivity.tex")


if __name__ == "__main__":
    main()
