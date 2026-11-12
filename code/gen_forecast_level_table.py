"""table_forecast_level.tex, from forecast_level_summary.csv of compute_forecast_level.py: the
forecast level and the loss ratios after the two level adjustments.

Usage:  python code/gen_forecast_level_table.py [--summary CSV] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, NAME, TABLES  # noqa: E402

ORDER = ["Log_HAR", "HAR", "HAR_J", "HAR_RS", "HARQ", "ARFIMA", "ARMA", "MEM", "chronos_bolt_small",
         "chronos_bolt_base", "moirai_2_0_small", "moirai_moe_small", "lag_llama", "timesfm_2_5",
         "toto", "sundial", "ttm"]
H = [1, 5, 22]


def table(s):
    cap = (r"Forecast level and level adjustments. Level: mean across the 50 assets of the per-asset "
           r"average of $RV_{t+h}/\widehat{\sigma}^{2}_{t+h}$, one when the variance forecast is "
           r"right on average. This column is a ratio of variances, not a loss ratio. Log-normal: "
           r"QLIKE loss ratio when Log-HAR, ARFIMA and ARMA are scored on their log-normal expected "
           r"variance $\exp(2\hat{m}_{t+h} + 2\hat{v}_{t+h})$ and the other models are unchanged, "
           r"relative to Log-HAR scored the same way. Rescaled: QLIKE loss ratio when every variance "
           r"forecast is multiplied by its average ratio of realized to forecast variance over its "
           r"last 250 forecasts whose targets are observed before the origin, relative to the "
           r"rescaled Log-HAR. Log-HAR therefore has a ratio of one under both adjustments. Every "
           r"forecast is clipped to the window bounds of Sec.~\ref{subsec:evaluation}, and every "
           r"model is scored on the dates common to the 17 models.")
    L = ["\\begin{table}[htbp]", "\\centering", "\\singlespacing", "\\caption{" + cap + "}",
         "\\label{tab:forecast_level}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{l" + "rrr" * 3 + "}", "\\toprule",
         "& \\multicolumn{3}{c}{Level} & \\multicolumn{3}{c}{Log-normal} & \\multicolumn{3}{c}{Rescaled} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Model" + " & $h=1$ & $h=5$ & $h=22$" * 3 + " \\\\", "\\midrule"]
    for key in ORDER:
        cells = [f"{s.loc[(key, h), col]:.3f}" for col in ["level", "lognorm", "recal"] for h in H]
        L.append(f"{NAME[key]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", type=Path, default=METRICS / "forecast_level_summary.csv")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    s = pd.read_csv(a.summary, float_precision="round_trip").set_index(["model", "h"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_forecast_level.tex").write_text(table(s))
    print("written to", a.out / "table_forecast_level.tex")


if __name__ == "__main__":
    main()
