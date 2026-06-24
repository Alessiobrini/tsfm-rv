"""Generate the Mincer-Zarnowitz bias-corrected QLIKE table across h=1,5,22.

Extends the original h=1-only robustness check (run_robustness.py) to all three
horizons. For each model and horizon we recompute the original QLIKE and the QLIKE
after a recursive affine MZ correction (alpha_t, beta_t estimated from daily-origin
forecasts strictly before t, expanding window, warm-up = 252), applied symmetrically
to all 17 models and winsorized to the in-sample volatility support. QLIKE uses
scale="vol" (squares the volatility forecast back to a variance), matching the
main-results QLIKE.

Outputs:
  results/volare/metrics/mz_bias_corrected_allh.csv  (per-asset, per-horizon)
  paper/tables/mz_bias_corrected.tex                 (cross-asset-mean table)

Run: python code/gen_mz_allh.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VOLARE_RESULTS_DIR, VOLARE_ALL_TICKERS
from evaluation.loss_functions import compute_loss_series
from evaluation.mz_regression import recursive_mz_correction
from run_evaluation import parse_forecast_filename, align_forecasts

FORECAST_DIR = VOLARE_RESULTS_DIR / "forecasts"
METRICS_DIR = VOLARE_RESULTS_DIR / "metrics"
TABLES_DIR = Path(__file__).resolve().parent.parent / "paper" / "tables"
MIN_WINDOW = 252
HORIZONS = [1, 5, 22]

MODEL_ORDER = ["Log_HAR", "HAR", "HAR_J", "HAR_RS", "HARQ", "ARFIMA", "ARMA", "MEM",
               "chronos_bolt_small", "chronos_bolt_base", "moirai_2_0_small",
               "moirai_moe_small", "lag_llama", "timesfm_2_5", "toto", "sundial", "ttm"]
DISPLAY = {"Log_HAR": "Log-HAR", "HAR": "HAR", "HAR_J": "HAR-J", "HAR_RS": "HAR-RS",
           "HARQ": "HARQ", "ARFIMA": "ARFIMA", "ARMA": "ARMA", "MEM": "MEM",
           "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
           "moirai_2_0_small": "Moirai-2.0-S", "moirai_moe_small": "Moirai-MoE-S",
           "lag_llama": "Lag-Llama", "timesfm_2_5": "TimesFM-2.5", "toto": "Toto",
           "sundial": "Sundial", "ttm": "TTM"}


def load_h(h):
    groups = defaultdict(dict)
    for fpath in FORECAST_DIR.glob("*.csv"):
        m, t, hor = parse_forecast_filename(fpath)
        if m is None or hor != h or t not in VOLARE_ALL_TICKERS:
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if "actual" not in df.columns or "forecast" not in df.columns:
            continue
        groups[t][m] = df.dropna(subset=["actual", "forecast"])
    return dict(groups)


def run_h(h):
    tf = load_h(h)
    rows = []
    for ticker in sorted(tf):
        ca, mf = align_forecasts(tf[ticker])
        if ca is None:
            continue
        a = ca.values
        if len(a) <= MIN_WINDOW:
            continue
        at = a[MIN_WINDOW:]
        lo = float(np.min(at[at > 0]))
        hi = float(np.max(at))
        for mn, fs in mf.items():
            if mn not in MODEL_ORDER:
                continue
            fa = fs.values
            ft = fa[MIN_WINDOW:]
            if len(ft) != len(at):
                continue
            q0 = float(np.mean(compute_loss_series(at, ft, loss_type="QLIKE", scale="vol")))
            try:
                corr = recursive_mz_correction(a, fa, min_window=MIN_WINDOW)
            except Exception:
                continue
            if len(corr) != len(at):
                continue
            corr = np.clip(corr, lo, hi)
            q1 = float(np.mean(compute_loss_series(at, corr, loss_type="QLIKE", scale="vol")))
            rows.append({"horizon": h, "ticker": ticker, "model": mn,
                         "QLIKE_orig": q0, "QLIKE_mz": q1})
    return pd.DataFrame(rows)


def fmt(v):
    if v > 1:
        return f"{v:.3f}$^{{\\dagger}}$"
    return f"{v:.3f}"


def main():
    all_df = pd.concat([run_h(h) for h in HORIZONS], ignore_index=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(METRICS_DIR / "mz_bias_corrected_allh.csv", index=False)

    # cross-asset means: {horizon: DataFrame[model -> (orig, mz)]}
    means = {}
    best_mz = {}
    for h in HORIZONS:
        sub = all_df[all_df.horizon == h]
        g = sub.groupby("model")[["QLIKE_orig", "QLIKE_mz"]].mean()
        means[h] = g
        best_mz[h] = g["QLIKE_mz"].idxmin()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{Mincer--Zarnowitz bias-corrected QLIKE across horizons "
                 r"(cross-asset mean over 50 assets). For each model and horizon we report the "
                 r"original QLIKE (Orig.) and the QLIKE after a recursive affine MZ correction "
                 r"(MZ), with $\hat\alpha_t,\hat\beta_t$ estimated from daily-origin forecasts "
                 r"strictly before each date on an expanding window and applied symmetrically to "
                 r"all 17 models. QLIKE is on the variance scale. The lowest corrected QLIKE in "
                 r"each horizon is in bold. $\dagger$ marks QLIKE $>1$.}")
    lines.append(r"\label{tab:mz_bias_corrected}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & "
                 r"\multicolumn{2}{c}{$h=22$} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"Model & Orig. & MZ & Orig. & MZ & Orig. & MZ \\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        cells = [DISPLAY[m]]
        for h in HORIZONS:
            if m not in means[h].index:
                cells += ["--", "--"]
                continue
            o = means[h].loc[m, "QLIKE_orig"]
            z = means[h].loc[m, "QLIKE_mz"]
            zc = fmt(z)
            if m == best_mz[h]:
                zc = r"\textbf{" + zc + "}"
            cells += [fmt(o), zc]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "mz_bias_corrected.tex").write_text("\n".join(lines) + "\n")
    print("Wrote", TABLES_DIR / "mz_bias_corrected.tex")
    for h in HORIZONS:
        ranked = means[h].sort_values("QLIKE_mz")
        print(f"h={h}: best MZ = {DISPLAY[best_mz[h]]} ({ranked.iloc[0]['QLIKE_mz']:.4f}); "
              f"TTM MZ rank = {list(ranked.index).index('ttm')+1}")


if __name__ == "__main__":
    main()
