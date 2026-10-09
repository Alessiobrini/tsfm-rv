"""gen_avg_target_table.py — Averaged-target appendix table.

Scores the averaged-target forecasts in results/volare_avg/forecasts/ the way the main loss-ratio
table scores the point forecasts. Every model is scored against one quantity, the mean daily
realized variance over the h days of the target (evaluation/targets.py), with its volatility
forecast squared, on the dates common to the 17 models of each asset and horizon. The stored
`actual` column is not used as the target, because it holds the square of the mean volatility for
most models and the mean variance for the augmented HARs. Per asset, each model's QLIKE is
divided by Log-HAR's, and the ratios are averaged across the 50 assets. Values below 1 beat
Log-HAR on average.

Before scoring, the script checks the target against the files it can be checked against: the
augmented HARs store the mean variance itself, so their `actual` squared must equal the target.

Usage:  python code/gen_avg_target_table.py [--dry-run] [--fc DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import VOLARE_ALL_TICKERS, PAPER_MODELS  # noqa: E402
from evaluation.loss_functions import qlike  # noqa: E402
from evaluation.targets import common_dates, holding_rv, rv_series_all  # noqa: E402

FC = ROOT / "results" / "volare_avg" / "forecasts"
TAB = ROOT / "paper" / "tables"
HORIZONS = [1, 5, 22]
STORES_MEAN_RV = {"HAR_J", "HAR_RS", "HARQ"}   # their `actual` is sqrt of the mean variance

DISPLAY = {"Log_HAR": "Log-HAR", "HAR": "HAR", "HAR_J": "HAR-J", "HAR_RS": "HAR-RS", "HARQ": "HARQ",
           "ARFIMA": "ARFIMA", "ARMA": "ARMA", "MEM": "MEM",
           "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
           "moirai_2_0_small": "Moirai-2.0-S", "moirai_moe_small": "Moirai-MoE-S",
           "lag_llama": "Lag-Llama", "timesfm_2_5": "TimesFM-2.5", "toto": "Toto",
           "sundial": "Sundial", "ttm": "TTM"}


def read(p):
    return pd.read_csv(p, index_col=0, parse_dates=True, float_precision="round_trip")


def score_asset(frames, rv, h):
    """{model: QLIKE} for one asset and horizon, against the mean daily RV on common dates."""
    idx = common_dates(frames)
    if len(idx) == 0:
        raise ValueError("no common dates")
    target = holding_rv(rv, idx, h)
    for m in STORES_MEAN_RV & set(frames):
        a2 = frames[m].loc[idx, "actual"].to_numpy() ** 2
        if not np.allclose(a2, target, rtol=1e-9, atol=0):
            raise ValueError(f"{m}: stored mean variance differs from the target "
                             f"(max abs gap {np.max(np.abs(a2 - target)):.3g})")
    return {m: qlike(target, d.loc[idx, "forecast"].to_numpy() ** 2, scale="var")
            for m, d in frames.items()}, len(idx)


def loss_ratios(fc_dir, rv_all, tickers=VOLARE_ALL_TICKERS, horizons=HORIZONS, models=PAPER_MODELS):
    """Per-asset QLIKE and ratio to Log-HAR for every model, asset and horizon."""
    rows = []
    for h in horizons:
        for t in tickers:
            paths = {m: fc_dir / f"{m}_{t}_h{h}.csv" for m in models}
            missing = [m for m, p in paths.items() if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{t} h={h}: no averaged-target file for {missing}")
            q, n = score_asset({m: read(p) for m, p in paths.items()}, rv_all[t], h)
            for m, v in q.items():
                rows.append(dict(ticker=t, h=h, model=m, qlike=v, ratio=v / q["Log_HAR"], n=n))
    return pd.DataFrame(rows)


def write_table(ratios):
    best = {h: min(ratios[(k, h)] for k in PAPER_MODELS if k != "Log_HAR") for h in HORIZONS}
    L = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
         r"\caption{Averaged target. Mean across the 50 assets of the per-asset QLIKE loss ratio to "
         r"Log-HAR when the target is the average of daily realized variance over the next $h$ days "
         r"\citep{patton2015}, with each volatility forecast squared. Every model is scored on the "
         r"dates common to the 17 models of each asset. Values below 1 beat Log-HAR on average, and "
         r"the lowest ratio at each horizon is in bold. Compare with Tab.~\ref{tab:loss_ratios}, "
         r"which uses the point-in-time target.}",
         r"\label{tab:avg_target}", r"\small",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule"]
    for key in PAPER_MODELS:
        cells = []
        for h in HORIZONS:
            v = ratios[(key, h)]
            s = f"{v:.3f}"
            if key != "Log_HAR" and abs(v - best[h]) < 1e-12:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        L.append(f"{DISPLAY[key]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table_avg_target.tex").write_text("\n".join(L) + "\n")
    print("wrote table_avg_target.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc", type=Path, default=FC, help="folder of averaged-target forecasts")
    ap.add_argument("--dry-run", action="store_true", help="print the ratios, write nothing")
    a = ap.parse_args()
    res = loss_ratios(a.fc, rv_series_all())
    mean = res.groupby(["model", "h"])["ratio"].mean()
    ratios = {(m, h): float(mean.loc[(m, h)]) for m in PAPER_MODELS for h in HORIZONS}
    for h in HORIZONS:
        beat = sorted((ratios[(k, h)], DISPLAY[k]) for k in PAPER_MODELS
                      if k != "Log_HAR" and ratios[(k, h)] < 1.0)
        print(f"h={h}: below Log-HAR: " + (", ".join(f"{d} {v:.3f}" for v, d in beat) or "none"))
    if not a.dry_run:
        out = ROOT / "results" / "volare_avg" / "metrics"
        out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "avg_target_per_asset.csv", index=False)
        write_table(ratios)


if __name__ == "__main__":
    main()
