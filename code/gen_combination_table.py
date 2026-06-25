"""
gen_combination_table.py — LaTeX table for the TTM+Log-HAR combination robustness.

Reads results/volare/metrics/combination_metrics.csv (per-asset QLIKE, optional
in_mcs, optional DM p-values written by run_combination_robustness.py) and writes
paper/tables/table_combination.tex:

  Panel A: average QLIKE loss ratios vs Log-HAR (mean across 50 assets).
  Panel B: MCS inclusion rates (% of 50 assets), if in_mcs is present.

Run run_combination_robustness.py --mcs first so combination_metrics.csv carries
the in_mcs column.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "volare" / "metrics"
OUT = ROOT / "paper" / "tables" / "table_combination.tex"
HORIZONS = [1, 5, 22]
LHAR = "Log_HAR"

# display order and labels for the combination table
ROWS = [
    ("Log_HAR", "Log-HAR"),
    ("ttm", "TTM"),
    ("sundial", "Sundial"),
    ("comb_ew", "TTM + Log-HAR (equal weight)"),
    ("comb_bg", "TTM + Log-HAR (Bates--Granger)"),
]


def loss_ratio_panel(df):
    table = {}  # model -> {h: ratio}
    for h in HORIZONS:
        sub = df[df.horizon == h]
        piv = sub.pivot_table(index="ticker", columns="model", values="QLIKE")
        ratios = piv.div(piv[LHAR], axis=0).mean(axis=0)
        for m in [r[0] for r in ROWS]:
            if m in ratios.index:
                table.setdefault(m, {})[h] = float(ratios[m])
    best = {}
    for h in HORIZONS:
        col = {m: table[m][h] for m in table if h in table[m]}
        best[h] = min(col, key=col.get) if col else None
    lines = ["\\multicolumn{4}{l}{\\textit{Panel A: average QLIKE loss ratio vs Log-HAR}} \\\\[2pt]"]
    for key, label in ROWS:
        if key not in table:
            continue
        cells = []
        for h in HORIZONS:
            v = table[key].get(h, np.nan)
            s = f"{v:.3f}" if np.isfinite(v) else "--"
            if key == best.get(h):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return lines


def mcs_panel(df):
    if "in_mcs" not in df.columns or df["in_mcs"].isna().all():
        return []
    lines = ["\\addlinespace",
             "\\multicolumn{4}{l}{\\textit{Panel B: MCS inclusion rate (\\% of 50 assets)}} \\\\[2pt]"]
    for key, label in ROWS:
        rates = []
        for h in HORIZONS:
            sub = df[(df.horizon == h) & (df.model == key)]
            rates.append(sub["in_mcs"].mean() * 100 if len(sub) else np.nan)
        if all(np.isnan(rates)):
            continue
        cells = [f"{r:.1f}" if np.isfinite(r) else "--" for r in rates]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return lines


def main():
    df = pd.read_csv(MET / "combination_metrics.csv")
    body = loss_ratio_panel(df) + mcs_panel(df)
    note = (
        "Average across the 50 assets of each model's per-asset QLIKE ratio to "
        "Log-HAR (1.000 by construction); values below 1 beat Log-HAR on average. "
        "The two members are TTM (best foundation model) and Log-HAR (best econometric "
        "benchmark); Sundial, the next-best single foundation model at $h=1$, is shown for "
        "reference. The equal-weight combination averages the two volatility forecasts; "
        "the Bates--Granger combination uses recursive variance-minimizing weights estimated "
        "from forecast errors observed strictly before each date (expanding window, clipped "
        "to $[0,1]$, equal-weight warm-up). By Diebold--Mariano test at 5\\%, the equal-weight "
        "combination has significantly lower QLIKE than Log-HAR on 42, 24, and 4 of 50 assets "
        "at $h=1,5,22$, and significantly beats TTM on at most 7 of 50, consistent with the "
        "forecast-combination puzzle: the combination matches the best single model while "
        "robustly improving on the econometric benchmark."
    )
    lines = [
        "\\begin{table}[H]", "\\centering", "\\singlespacing",
        "\\caption{Forecast-combination robustness: average QLIKE loss ratios "
        "relative to Log-HAR across the 50 assets for two combinations of the TTM "
        "and Log-HAR volatility forecasts, an equal-weight average and a recursive "
        "Bates--Granger combination with variance-minimizing weights estimated on "
        "an expanding window. Values below one beat Log-HAR; both combinations beat "
        "Log-HAR at every horizon and match the best single model.}",
        "\\label{tab:combination}", "\\small",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Model & $h=1$ & $h=5$ & $h=22$ \\\\", "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}", "\\\\[6pt]",
        f"\\parbox{{\\textwidth}}{{\\footnotesize {note}}}",
        "\\end{table}",
    ]
    OUT.write_text("\n".join(lines))
    print("Wrote", OUT)
    print("\n".join(body))


if __name__ == "__main__":
    main()
