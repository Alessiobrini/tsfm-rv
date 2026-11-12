"""The two contamination tables, from the output of compute_contamination.py.

  table_contamination_split.tex  loss ratio to Log-HAR before and after the split at 1 January 2024
                                 (every model) and at the date of each foundation model's weight file,
                                 with the panel t-statistic of the excess over the four controls
  table_contamination_years.tex  loss ratio to Log-HAR by calendar year of the target

Usage:  python code/gen_contamination_tables.py [--metrics DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, NAME, TABLES, TSFM_ROWS as TSFM  # noqa: E402

ECON = ["HAR", "ARFIMA", "ARMA", "MEM", "HAR_J", "HAR_RS", "HARQ"]
H = [1, 5, 22]


def split_table(summ, ctrl):
    def cells(split, model):
        out = []
        for h in H:
            s = summ[(summ.split == split) & (summ.h == h) & (summ.model == model)].iloc[0]
            c = ctrl[(ctrl.split == split) & (ctrl.h == h) & (ctrl.model == model)]
            t = f"{c.t_panel_excess.iloc[0]:.2f}" if len(c) else "--"
            out += [f"{s.ratio_before:.3f}", f"{s.ratio_after:.3f}", t]
        return out

    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
             (r"\caption{Contamination check by sample split. Mean across the 50 assets of the per-asset "
              r"QLIKE loss ratio to Log-HAR before and after the split, by the date of the forecast "
              r"target. Panel~A: every model split at 1 January 2024. Panel~B: each foundation model "
              r"split at the date its pinned weight file was last changed on Hugging Face, which cannot "
              r"be earlier than the release of the weights and is therefore an upper bound on the end of "
              r"its pretraining data, with that date and the number of trading days after it (median "
              r"across assets, at $h = 1$). $t$ is the Newey--West $t$-statistic of a dummy for the later "
              r"period in a regression of the cross-asset daily mean of the model's QLIKE minus the "
              r"average QLIKE of HAR, ARFIMA, ARMA and MEM. A positive $t$ means the model loses ground "
              r"after the split relative to these four models, which are estimated only on the 1{,}000 "
              r"days before each origin. A dash (--) marks these four control models, for which we do not "
              r"run the test.}"),
             r"\label{tab:contamination_split}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llr" + "rrr" * 3 + "}", r"\toprule",
             r"& & & \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
             r"Model & Weights & Days & Before & After & $t$ & Before & After & $t$ & Before & After & $t$ \\",
             r"\midrule",
             r"\multicolumn{12}{l}{\textbf{Panel A: split at 1 January 2024}} \\[2pt]"]
    for m in ECON + TSFM:
        lines.append(" & ".join([NAME[m], "", ""] + cells("A_2024", m)) + r" \\")
    lines += [r"\midrule", r"\multicolumn{12}{l}{\textbf{Panel B: split at the date of each model's weight file}} \\[2pt]"]
    for m in TSFM:
        c = ctrl[(ctrl.split == "B_weights") & (ctrl.model == m)].iloc[0]
        wdate = (pd.Timestamp(c.cut) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        lines.append(" & ".join([NAME[m], wdate, str(int(c.days_after))] + cells("B_weights", m)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def years_table(years):
    cols = list(years.columns)
    ylab = [("Jan.~" + c) if c == "2026" else c for c in cols]
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing",
             r"\caption{QLIKE loss ratio to Log-HAR by calendar year of the forecast target. Mean across the "
             r"50 assets of the per-asset ratio of mean QLIKE to Log-HAR's on the same days. The last column "
             r"covers January 2026, the end of the sample.}",
             r"\label{tab:contamination_years}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(ylab) + r" \\", r"\midrule"]
    for i, h in enumerate(H):
        if i:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textbf{Panel %s: $h=%d$}} \\[2pt]" % (len(cols) + 1, "ABC"[i], h))
        for m in ECON + TSFM:
            v = years.loc[(h, m)]
            lines.append(NAME[m] + " & " + " & ".join(f"{x:.3f}" for x in v.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", type=Path, default=METRICS, help="output folder of compute_contamination.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    summ = pd.read_csv(a.metrics / "contamination_summary.csv")
    ctrl = pd.read_csv(a.metrics / "contamination_control.csv")
    years = pd.read_csv(a.metrics / "contamination_years.csv", index_col=[0, 1])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_contamination_split.tex").write_text(split_table(summ, ctrl))
    (a.out / "table_contamination_years.tex").write_text(years_table(years))
    print("written to", a.out)


if __name__ == "__main__":
    main()
