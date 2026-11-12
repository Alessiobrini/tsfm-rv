"""The two tables of the supervised ML baselines, from ml_baselines_summary.csv of
compute_ml_baselines.py: their loss ratios (table_ml_baselines.tex) and their level and tracking
diagnostics (table_ml_diag.tex).

Usage:  python code/gen_ml_tables.py [--summary CSV] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import METRICS, TABLES  # noqa: E402

MAIN = [("ttm", "TTM"), ("Log_HAR_refit22", "Log-HAR, refit every 22 days"), ("xgb_har", "XGBoost"),
        ("ffn_har", "Feedforward network"), ("lstm_22", "LSTM"), ("lstm_22_p100", "LSTM, patience 100")]
DIAG = [("Log_HAR", "Log-HAR"), ("ttm", "TTM"), ("xgb_har", "XGBoost"), ("ffn_har", "Feedforward network"),
        ("lstm_22", "LSTM"), ("lstm_22_p100", "LSTM, patience 100")]
H = (1, 5, 22)


def cell(s, m, h, col, fmt):
    r = s[(s.model == m) & (s.kind == "point") & (s.h == h)]
    if len(r) != 1:
        raise ValueError(f"{m} h={h}: {len(r)} rows")
    return fmt.format(r[col].iloc[0])


def main_table(s):
    rows = []
    for m, name in MAIN:
        cells = []
        for h in H:
            cells += [cell(s, m, h, "ratio", "{:.3f}"), cell(s, m, h, "beat", "{:d}")]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        (r"\caption{Supervised machine learning baselines. Mean across the 50 assets of the per-asset "
         r"QLIKE loss ratio to Log-HAR, with the number of assets, of 50, with a ratio below one "
         r"($<1$). XGBoost and the feedforward network take the logarithms of the average volatility "
         r"over the last 1, 5 and 22 days, and the LSTM takes the last 22 values of log volatility. "
         r"Each learner forecasts log volatility $h$ days ahead directly. We fit it on the 1{,}000 "
         r"days before the origin, refit it every 22 trading days and tune it every 252 trading days, "
         r"and we retransform its forecast to volatility with the rule of Log-HAR "
         r"(Appendix~\ref{app:ml}). LSTM, patience 100: the LSTM with early stopping at a patience of "
         r"100 epochs and at most 500 epochs. Log-HAR, refit every 22 days: Log-HAR refitted every 22 "
         r"trading days rather than at every origin. TTM is as in the main comparison. Every model is "
         r"scored on the dates common to every model, with forecasts clipped to the bounds of "
         r"Sec.~\ref{subsec:evaluation}.}"),
        r"\label{tab:ml_baselines}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=22$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Model & Ratio & $<1$ & Ratio & $<1$ & Ratio & $<1$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def diag_table(s):
    rows = []
    for m, name in DIAG:
        cells = []
        for col, fmt in (("bias", "{:.3f}"), ("sd_err", "{:.3f}"), ("rc_ratio", "{:.3f}")):
            cells += [cell(s, m, h, col, fmt) for h in H]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\singlespacing",
        (r"\caption{Level and tracking error of the supervised machine learning baselines. For the "
         r"volatility forecast $\widehat{\sigma}_{t+h}$ and the realized volatility $\sigma_{t+h}$, "
         r"the level error is the mean of $\log(\sigma_{t+h}/\widehat{\sigma}_{t+h})$, negative when "
         r"the forecasts are above the realized values, and the tracking error is the standard "
         r"deviation of $\log(\sigma_{t+h}/\widehat{\sigma}_{t+h})$ as a ratio to that of Log-HAR. "
         r"The last three columns give the QLIKE loss ratio to Log-HAR after every model, Log-HAR "
         r"included, is rescaled in real time as in Appendix~\ref{app:forecast_level}. Each measure "
         r"is computed per asset and averaged across the 50 assets. LSTM, patience 100: the LSTM with "
         r"early stopping at a patience of 100 epochs and at most 500 epochs.}"),
        r"\label{tab:ml_diag}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Level error} & \multicolumn{3}{c}{Tracking error} & \multicolumn{3}{c}{Ratio after rescaling} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Model & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ & $h=1$ & $h=5$ & $h=22$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", type=Path, default=METRICS / "ml_baselines_summary.csv")
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    s = pd.read_csv(a.summary, float_precision="round_trip")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_ml_baselines.tex").write_text(main_table(s))
    (a.out / "table_ml_diag.tex").write_text(diag_table(s))
    print("written to", a.out)


if __name__ == "__main__":
    main()
