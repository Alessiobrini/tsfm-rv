"""table_equal_context.tex, from the output of compute_equal_context.py.

Panel A: each foundation model at its main context against Log-HAR on 1,000 days, and at 512 days
against Log-HAR on 512 days. Panel B: TTM by context within the r2 release, next to the r2.1 model.

Usage:  python code/gen_equal_context_table.py [--metrics DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, NAME, TABLES, TSFM_ROWS  # noqa: E402

H = ["1", "5", "22"]


def table(p1, p2):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
             (r"\caption{Foundation models at an equal context. Mean across the 50 assets of the "
              r"per-asset QLIKE loss ratio, every forecast clipped to the bounds of "
              r"Sec.~\ref{subsec:evaluation}. Panel~A: each model at its main context against Log-HAR "
              r"estimated on 1{,}000 days, and at 512 days against Log-HAR estimated on 512 days, on the "
              r"dates of the main comparison. TTM's r2.1 release supports at most 512 days, so its two "
              r"columns coincide in the context. Panel~B: TTM by context against Log-HAR on 1{,}000 days, "
              r"on the dates for which the 1{,}536-day model has forecasts. The r2.1 release is the one "
              r"used in the main comparison, pretrained with daily series and a frequency token. The r2 "
              r"release is the earlier one, pretrained on a smaller corpus without daily series. Its "
              r"512-day checkpoint has a 192-day forecast head and the others a 96-day head.}"),
             r"\label{tab:equal_context}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"& & \multicolumn{3}{c}{Main context} & \multicolumn{3}{c}{512 days} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Model & Context & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule",
             r"\multicolumn{8}{l}{\textbf{Panel A: every foundation model at 512 days}} \\[2pt]"]
    for m in TSFM_ROWS:
        ctx = "512" if m == "ttm" else "1{,}000"
        a = [f"{p1.loc[f'{m}@main', ('Log_HAR@1000', h)]:.3f}" for h in H]
        b = [f"{p1.loc[f'{m}@512', ('Log_HAR@512', h)]:.3f}" for h in H]
        lines.append(" & ".join([NAME[m], ctx] + a + b) + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{\textbf{Panel B: TTM by context}} \\[2pt]"]
    for k, lab, ctx in [("ttm_r2.1@512", "TTM r2.1", "512"), ("ttm_r2@512", "TTM r2", "512"),
                        ("ttm_r2@1024", "TTM r2", "1{,}024"), ("ttm_r2@1536", "TTM r2", "1{,}536")]:
        lines.append(" & ".join([lab, ctx] + [f"{p2.loc[k, h]:.3f}" for h in H] + ["", "", ""]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", type=Path, default=METRICS, help="output folder of compute_equal_context.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    p1 = pd.read_csv(a.metrics / "equal_context_panel1.csv", header=[0, 1], index_col=0)
    p2 = pd.read_csv(a.metrics / "equal_context_panel2.csv", index_col=0)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_equal_context.tex").write_text(table(p1, p2))
    print("written to", a.out / "table_equal_context.tex")


if __name__ == "__main__":
    main()
