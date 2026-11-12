"""The two tables of the foundation models applied to log volatility.

  table_log_input.tex         loss ratio to Log-HAR of each foundation model with volatility as input
                              and of its log version, with the percentage of assets where the log
                              version beats Log-HAR (from compute_log_input.py)
  table_log_input_mcs_dm.tex  MCS inclusion and DM win rates with the log version of every foundation
                              model, built with the functions behind the main MCS and DM table
                              (process_results.py), from the metrics that run_evaluation_volare.py
                              writes for the set of build_log_input_set.py

Usage:  python code/gen_log_input_tables.py [--summary CSV] [--login-metrics DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VOLARE_ALL_TICKERS  # noqa: E402
from evaluation.forecast_sets import METRICS, NAME, ROOT, TABLES, TSFM_ROWS  # noqa: E402

H = [1, 5, 22]
LOGIN_METRICS = ROOT / "results" / "volare_log_input" / "metrics"


def log_input_table(S):
    g = S.set_index(["model", "spec", "h"])
    rows = []
    for m in TSFM_ROWS:
        cells = [NAME[m]]
        for h in H:
            v, lg = g.loc[(m, "vol", h)], g.loc[(m, "log", h)]
            cells += [f"{v.ratio:.3f}", f"{lg.ratio:.3f}", f"{100 * lg.beat / lg.n_assets:.0f}"]
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Foundation models applied to log volatility. Mean across the 50 assets of the per-asset "
           r"QLIKE loss ratio to Log-HAR for each foundation model with volatility as input, as in the "
           r"main comparison (Vol.), and for its log version, which receives log volatility over the "
           r"same context (Log), with the percentage of the 50 assets on which the log version has a "
           r"ratio below one (\% $<1$). We clip the log forecast $\hat{m}_{t+h}$ to the logarithm of the "
           r"bounds of Sec.~\ref{subsec:evaluation} and retransform it to volatility as "
           r"$\exp(\hat{m}_{t+h} + \hat{v}_t/2)$. Here $\hat{v}_t$ is the variance of the model's "
           r"one-step errors in log volatility over the 1{,}000 forecasts before the origin, or over all "
           r"earlier forecasts during the first 1{,}000 days of the evaluation period, with $\hat{v}_t = "
           r"0$ for the first 22 forecasts. Every model is scored on the dates common to every model and "
           r"both inputs.")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\caption{" + cap + "}",
             r"\label{tab:log_input}", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "rrr" * 3 + "}", r"\toprule",
             r"& \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Model & Vol. & Log & \% $<1$ & Vol. & Log & \% $<1$ & Vol. & Log & \% $<1$ \\", r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def body_rows(tex):
    """(row label, cells) of a table written by process_results.py, percent signs removed."""
    body = tex.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if "&" not in line or line.startswith("\\"):
            continue
        cells = [c.strip() for c in line.rstrip("\\").strip().split("&")]
        out.append((cells[0], [c.replace("\\%", "") for c in cells[1:]]))
    return out


def mcs_dm_table(metrics):
    import process_results as pr
    tsfm_labels = {NAME[m] for m in TSFM_ROWS}
    mcs = body_rows(pr.make_mcs_table(pd.read_csv(metrics / "mcs_all_results.csv"), VOLARE_ALL_TICKERS,
                                      caption="", label="x"))
    dm = body_rows(pr.make_dm_summary_table(metrics, VOLARE_ALL_TICKERS, caption="", label="y"))
    if [m for m, _ in mcs] != [m for m, _ in dm]:
        raise ValueError("model rows differ between the MCS and DM tables")
    if {m for m, _ in mcs} & tsfm_labels != tsfm_labels:
        raise ValueError("a foundation-model row label is missing")
    cap = (r"MCS inclusion rates and DM pairwise win rates for 50 assets with the log version of "
           r"every foundation model, retransformed to volatility as in Sec.~\ref{sec:robustness}, "
           r"and the econometric models as in Tab.~\ref{tab:mcs}. MCS columns report the percentage "
           r"of assets for which the model is included in the MCS at the 10\% significance level "
           r"(QLIKE loss, range statistic $T_R$ and its elimination rule, block bootstrap with $B = "
           r"10{,}000$). DM columns report the percentage of pairwise comparisons (across 50 assets "
           r"$\times$ 16 opponents = 800 tests) in which the row model achieves significantly lower "
           r"QLIKE at the 5\% level.")
    L = ["\\begin{table}[htbp]", "\\centering", "\\singlespacing", "\\caption{" + cap + "}",
         "\\label{tab:log_input_mcs_dm}", "\\small", "\\begin{tabular}{lrrrrrr}", "\\toprule",
         "& \\multicolumn{3}{c}{MCS inclusion (\\%)} & "
         "\\multicolumn{3}{c}{DM win rate (\\%)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\\\", "\\midrule"]
    for (m, a), (_, b) in zip(mcs, dm):
        name = f"Log-{m}" if m in tsfm_labels else m
        L.append(f"{name} & " + " & ".join(a + b) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", type=Path, default=METRICS / "log_input_summary.csv",
                    help="log_input_summary.csv of compute_log_input.py")
    ap.add_argument("--login-metrics", type=Path, default=LOGIN_METRICS,
                    help="metrics folder of run_evaluation_volare.py on the set of build_log_input_set.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_log_input.tex").write_text(log_input_table(pd.read_csv(a.summary)))
    (a.out / "table_log_input_mcs_dm.tex").write_text(mcs_dm_table(a.login_metrics))
    print("written to", a.out)


if __name__ == "__main__":
    main()
