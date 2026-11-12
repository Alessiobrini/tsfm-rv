"""The two combination diagnostics tables, from the output of compute_combination_diagnostics.py.

  table_combination_diagnostics.tex  why the equal-weight TTM and Log-HAR combination works, by horizon
  table_combination_pairs.tex        Log-HAR averaged with every other model, TTM with every
                                     econometric model, and HAR with every other model

Usage:  python code/gen_combination_diagnostics.py [--metrics DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, NAME, TABLES  # noqa: E402

H = [1, 5, 22]
ORDER = ["ttm", "chronos_bolt_small", "chronos_bolt_base", "timesfm_2_5", "moirai_2_0_small",
         "moirai_moe_small", "lag_llama", "toto", "sundial",
         "HAR", "ARFIMA", "ARMA", "MEM", "HAR_J", "HAR_RS", "HARQ"]


def row(label, vals, fmt):
    return " & ".join([label] + [fmt.format(v) for v in vals]) + r" \\"


def diagnostics_table(M):
    def by_h(f):
        return [f(M[M.h == h]) for h in H]

    rows = [r"\multicolumn{4}{l}{\textbf{Panel A: forecast errors on the volatility scale}} \\[2pt]",
            row("RMSE ratio $k$ (median)", by_h(lambda x: x.k.median()), "{:.3f}"),
            row("Error correlation $\\rho$ (median)", by_h(lambda x: x.rho.median()), "{:.3f}"),
            row("Threshold $(3-k^2)/(2k)$ (median)", by_h(lambda x: x.thr.median()), "{:.3f}"),
            row("Assets with $\\rho$ below the threshold", by_h(lambda x: int((x.rho < x.thr).sum())), "{:d}"),
            r"\midrule",
            r"\multicolumn{4}{l}{\textbf{Panel B: QLIKE loss ratio to Log-HAR}} \\[2pt]",
            row("TTM", by_h(lambda x: x.r_T.mean()), "{:.3f}"),
            row("Harmonic mean of the variance forecasts", by_h(lambda x: x.r_harm.mean()), "{:.3f}"),
            row("Average of the volatility forecasts", by_h(lambda x: x.r_vol.mean()), "{:.3f}"),
            row("Assets where the average beats both",
                by_h(lambda x: int((x.r_vol < np.minimum(1.0, x.r_T)).sum())), "{:d}"),
            row("Diversity share of the gain (\\%)",
                by_h(lambda x: 100 * (x.A / (x.A + x.combiner)).mean()), "{:.0f}"),
            r"\midrule",
            r"\multicolumn{4}{l}{\textbf{Panel C: after a real-time level correction of both models}} \\[2pt]",
            row("TTM", by_h(lambda x: x.rc_T.mean()), "{:.3f}"),
            row("Log-HAR", by_h(lambda x: x.rc_L.mean()), "{:.3f}"),
            row("Average of the volatility forecasts", by_h(lambda x: x.rc_vol.mean()), "{:.3f}"),
            row("Assets where the average beats both",
                by_h(lambda x: int((x.rc_vol < np.minimum(x.rc_T, x.rc_L)).sum())), "{:d}"),
            r"\midrule",
            r"\multicolumn{4}{l}{\textbf{Panel D: encompassing under QLIKE, assets with $t > 1.96$}} \\[2pt]",
            row("TTM adds information to Log-HAR", by_h(lambda x: int((x.foc_T_adds > 1.96).sum())), "{:d}"),
            row("Log-HAR adds information to TTM", by_h(lambda x: int((x.foc_L_adds > 1.96).sum())), "{:d}"),
            r"\midrule",
            r"\multicolumn{4}{l}{\textbf{Panel E: stability over time}} \\[2pt]",
            row("126-day blocks with TTM ahead (\\%)", by_h(lambda x: 100 * x.share_T_leads.mean()), "{:.0f}"),
            row("Changes of the leading model per asset", by_h(lambda x: x.switches.mean()), "{:.1f}"),
            row("Bates--Granger weight on TTM (mean)", by_h(lambda x: x.w_mean.mean()), "{:.2f}"),
            row("Bates--Granger combination, loss ratio", by_h(lambda x: x.r_bg.mean()), "{:.3f}")]
    cap = (r"Diagnostics of the equal-weight combination of TTM and Log-HAR. Panel~A: on the volatility "
           r"scale, $k$ is the ratio of the larger to the smaller root mean squared error (RMSE) of the "
           r"two models and $\rho$ the correlation of their errors, both computed around zero. Under "
           r"squared error the average of the two forecasts beats the better one when $\rho < "
           r"(3-k^2)/(2k)$. Panel~B: mean across the 50 assets of the per-asset QLIKE ratio to Log-HAR "
           r"for TTM and for two combinations, the harmonic mean of the two variance forecasts and the "
           r"square of the average of the two volatility forecasts, which is the combination of "
           r"Tab.~\ref{tab:combination}. The diversity share is the part of the combination's gain over "
           r"the average QLIKE of its two members that comes from the diversity term $D_t$ of "
           r"Eq.~\eqref{eq:diversity}, averaged across assets. Panel~C: each model's variance forecast "
           r"multiplied by its average ratio of realized to forecast variance over the previous 250 "
           r"forecasts whose targets are observed at the origin, as in "
           r"Appendix~\ref{app:forecast_level}, with ratios to the uncorrected Log-HAR. Panel~D: number "
           r"of assets, of 50, where a Newey--West test rejects at the 5\% level that moving weight from "
           r"one model toward the other leaves the expected QLIKE unchanged. Panel~E: share of 126-day "
           r"blocks in which TTM has the lower QLIKE, number of changes of the model with the lower "
           r"QLIKE between consecutive blocks, and the recursive Bates--Granger combination with its "
           r"weight estimated from the errors observed by each origin.")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\caption{" + cap + "}",
             r"\label{tab:combination_diagnostics}", r"\small",
             r"\begin{tabular}{lrrr}", r"\toprule", r" & $h=1$ & $h=5$ & $h=22$ \\", r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def pairs_table(P):
    g = P.groupby(["anchor", "partner", "h"]).agg(r_member=("r_member", "mean"), r_pair=("r_pair", "mean"),
                                                 r_harm=("r_pair_harm", "mean"), rho=("rho", "median"))

    def pair_row(anchor, m):
        cells = [NAME[m]]
        for h in H:
            v = g.loc[(anchor, m, h)]
            cells += [f"{v.r_member:.3f}", f"{v.r_pair:.3f}", f"{v.r_harm:.3f}", f"{v.rho:.3f}"]
        return " & ".join(cells) + r" \\"

    cap = (r"Equal-weight pairs. Mean across the 50 assets of the per-asset QLIKE ratio to Log-HAR for "
           r"the partner model alone, for the square of the average of the two volatility forecasts "
           r"(Avg.), and for the harmonic mean of the two variance forecasts (Harm.), with the median "
           r"correlation $\rho$ of the two models' errors on the volatility scale. Panel~A: Log-HAR "
           r"with every other model. Panel~B: TTM with every econometric model. Panel~C: HAR with every "
           r"model other than Log-HAR, whose pair with HAR is in Panel~A.")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\caption{" + cap + "}",
             r"\label{tab:combination_pairs}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "rrrr" * 3 + "}", r"\toprule",
             r"& \multicolumn{4}{c}{$h=1$} & \multicolumn{4}{c}{$h=5$} & \multicolumn{4}{c}{$h=22$} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
             r"Partner & Alone & Avg. & Harm. & $\rho$ & Alone & Avg. & Harm. & $\rho$ & Alone & Avg. & Harm. & $\rho$ \\",
             r"\midrule", r"\multicolumn{13}{l}{\textbf{Panel A: averaged with Log-HAR}} \\[2pt]"]
    lines += [pair_row("Log_HAR", m) for m in ORDER if ("Log_HAR", m, 1) in g.index]
    lines += [r"\midrule", r"\multicolumn{13}{l}{\textbf{Panel B: averaged with TTM}} \\[2pt]"]
    lines += [pair_row("ttm", m) for m in ORDER if ("ttm", m, 1) in g.index]
    lines += [r"\midrule", r"\multicolumn{13}{l}{\textbf{Panel C: averaged with HAR}} \\[2pt]"]
    lines += [pair_row("HAR", m) for m in ORDER if ("HAR", m, 1) in g.index]
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", type=Path, default=METRICS,
                    help="output folder of compute_combination_diagnostics.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    M = pd.read_csv(a.metrics / "combination_diagnostics_long.csv")
    P = pd.read_csv(a.metrics / "combination_pairs_long.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_combination_diagnostics.tex").write_text(diagnostics_table(M))
    (a.out / "table_combination_pairs.tex").write_text(pairs_table(P))
    print("written to", a.out)


if __name__ == "__main__":
    main()
