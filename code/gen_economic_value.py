"""table_economic_value.tex, from economic_value_per_asset.csv and economic_value_tests.csv of
compute_economic_value.py.

Usage:  python code/gen_economic_value.py [--metrics DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, NAME as MODEL_NAME, TABLES  # noqa: E402

ORDER = ["HAR", "Log_HAR", "HAR_J", "HAR_RS", "HARQ", "ARFIMA", "ARMA", "MEM", "chronos_bolt_small",
         "chronos_bolt_base", "moirai_2_0_small", "moirai_moe_small", "lag_llama", "timesfm_2_5",
         "toto", "sundial", "ttm", "ttm+Log_HAR", "perfect"]
NAME = dict(MODEL_NAME, **{"ttm+Log_HAR": "TTM and Log-HAR", "perfect": "Perfect forecast"})
H, C = [1, 5, 22], ["zero", "full", "gradual"]


def table(pa, tv):
    u = pa.groupby(["model", "h", "case"]).utility.mean().unstack(["h", "case"])
    gap = 1e4 * u.sub(u.loc["Log_HAR"], axis=1)
    t = tv[tv.group == "all"].set_index(["model", "h", "case"])
    stacked = gap.drop("Log_HAR").stack(["h", "case"])
    if not (abs(stacked - t.diff_bp.reindex(stacked.index)) < 0.05).all():
        raise ValueError("the utility gaps differ from the gaps of the panel tests")

    rows_a = [" & ".join([NAME[m]] + [f"{100 * u.loc[m, (h, c)]:.2f}" for h in H for c in C]) + r" \\"
              for m in ORDER]
    rows_b = []
    for m in ORDER:
        if m == "Log_HAR":
            continue
        cells = []
        for h in H:
            for c in C:
                g, tt = gap.loc[m, (h, c)], t.loc[(m, h, c), "t"]
                cells.append(f"{g:.1f}".replace("-", "$-$") + (r"$^{*}$" if abs(tt) > 1.96 else ""))
        rows_b.append(" & ".join([NAME[m]] + cells) + r" \\")
    cap = (r"Realized utility of the volatility forecasts. The investor holds the fraction $x_t = "
           r"(SR/\gamma)/\sqrt{252 F_t}$ of wealth in the asset, with $SR = 0.4$ and $\gamma = 2$, and "
           r"rebalances every day. At $h = 1$, $F_t$ is the square of the point forecast of the main "
           r"comparison, and at $h = 5$ and $h = 22$ it is the square of the forecast of average "
           r"volatility over the next $h$ days. Panel~A: realized utility of Eq.~\eqref{eq:utility}, in "
           r"percent of wealth per year, averaged over the forecast origins of each asset and then over "
           r"the 50 assets. Zero: no trading costs. Full: each change in position costs the asset's "
           r"bid-ask spread. Gradual (Grad.): the position moves 15\% of the way to its target each day "
           r"and pays the same spread. Panel~B: difference from Log-HAR in basis points per year. An "
           r"asterisk marks a DM test that rejects equal utility at the 5\% level, on a daily series "
           r"that weights the 50 assets equally, with Newey--West standard errors and $\max(22, 2h)$ "
           r"lags. TTM and Log-HAR is the equal-weight average of their volatility forecasts. The "
           r"perfect forecast sets $F_t$ equal to the realized variance.")
    head = r"& \multicolumn{3}{c}{$h=1$} & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=22$} \\"
    sub = r"Model & Zero & Full & Grad. & Zero & Full & Grad. & Zero & Full & Grad. \\"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\caption{" + cap + "}",
             r"\label{tab:economic_value}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "rrr" * 3 + "}", r"\toprule", head,
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}", sub, r"\midrule",
             r"\multicolumn{10}{l}{\textbf{Panel A: realized utility (\% per year)}} \\[2pt]"]
    lines += rows_a + [r"\midrule", r"\multicolumn{10}{l}{\textbf{Panel B: difference to Log-HAR (basis points per year)}} \\[2pt]"]
    lines += rows_b + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", type=Path, default=METRICS, help="output folder of compute_economic_value.py")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    pa = pd.read_csv(a.metrics / "economic_value_per_asset.csv")
    tv = pd.read_csv(a.metrics / "economic_value_tests.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_economic_value.tex").write_text(table(pa, tv))
    print("written to", a.out / "table_economic_value.tex")


if __name__ == "__main__":
    main()
