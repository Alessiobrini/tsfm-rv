"""Diebold-Mariano test of each foundation model's log version against the same model with volatility
as input, asset by asset, and table_log_vs_vol_dm.tex.

The forecasts, the retransformation, the clipping and the common dates are those of
compute_log_input.py. When its log_input_long.csv is given with --check, the per-asset QLIKE ratios
must equal it before anything is written. The test is the one of evaluation/dm_test.py: two-sided,
Newey-West variance with max(1, h - 1) lags, on the QLIKE loss of the variance forecasts. A log
version is significantly better on an asset when p < 0.05 and its mean QLIKE is the lower one, as in
the DM win rates of the main MCS and DM table.

Output: dm_log_vs_vol_long.csv and dm_log_vs_vol_summary.csv in --metrics-out, and the table in --out.

Usage:  python code/dm_log_vs_vol.py [--forecasts DIR] [--log-forecasts DIR] [--check CSV]
                                     [--metrics-out DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_log_input import (LOG_FORECASTS, back_to_vol, clip_log, common_forecast_dates,  # noqa: E402
                               load_log, past_error_variance)
from evaluation.dm_test import dm_test  # noqa: E402
from evaluation.forecast_sets import (ALL, FORECASTS, L, METRICS, MODELS, NAME, TABLES, TSFM,  # noqa: E402
                                      TSFM_ROWS, load_clipped, load_sigma_all, qlike_series as ql)
from forecasting.bounds import clip_to_window  # noqa: E402

H = (1, 5, 22)
ALPHA = 0.05


def run(fc, log_fc, sigma_all):
    """Per asset, horizon and model, the DM test of the log version against the volatility one."""
    rows = []
    for t in ALL:
        sig = sigma_all[t]
        logs1 = {m: clip_log(d, sig) for m, d in load_log(log_fc, t, 1).items()}
        var1 = {m: past_error_variance(logs1[m]) for m in TSFM}
        for h in H:
            vol = load_clipped(fc, MODELS, t, h, sig)
            logs = logs1 if h == 1 else {m: clip_log(d, sig) for m, d in load_log(log_fc, t, h).items()}
            idx = common_forecast_dates(list(vol.values()) + list(logs.values()))
            rv = vol[L].loc[idx, "actual"].values ** 2
            base = ql(rv, vol[L].loc[idx, "forecast"].values ** 2).mean()
            for m in TSFM:
                l_vol = ql(rv, vol[m].loc[idx, "forecast"].values ** 2)
                f_log = clip_to_window(back_to_vol(logs[m], var1[m]), sig).loc[idx].values
                l_log = ql(rv, f_log ** 2)
                r = dm_test(l_log, l_vol, h=h, alternative="two-sided")
                rows.append(dict(ticker=t, h=h, model=m, n=len(idx), ratio_vol=l_vol.mean() / base,
                                 ratio_log=l_log.mean() / base, dm_stat=r.statistic, p=r.p_value,
                                 mean_diff=r.mean_loss_diff))
        print("done", t, flush=True)
    return pd.DataFrame(rows)


def check_against(res, long_csv):
    """The per-asset ratios must equal those of compute_log_input.py."""
    ref = pd.read_csv(long_csv).set_index(["ticker", "h", "model", "spec"])["ratio"]
    for spec, col in (("vol", "ratio_vol"), ("log", "ratio_log")):
        r = ref.xs(spec, level="spec").reindex(pd.MultiIndex.from_frame(res[["ticker", "h", "model"]]))
        if r.isna().any() or not np.allclose(r.values, res[col].values, rtol=1e-10, atol=0):
            raise AssertionError(f"{spec}: ratios differ from {long_csv}, max gap "
                                 f"{np.nanmax(np.abs(r.values - res[col].values)):.2e}")


def summarize(res):
    sig = res["p"] < ALPHA
    res = res.assign(lower=sig & (res["mean_diff"] < 0), higher=sig & (res["mean_diff"] > 0))
    return (res.groupby(["model", "h"])
               .agg(lower=("lower", "mean"), higher=("higher", "mean"), n_assets=("ticker", "size"))
               .mul({"lower": 100, "higher": 100, "n_assets": 1}).reset_index())


def table(summ):
    g = summ.set_index(["model", "h"])
    cap = (r"DM tests of the log version of each foundation model against the same model with "
           r"volatility as input. Percentage of the 50 assets on which the log version has a "
           r"significantly lower QLIKE (Lower) or a significantly higher QLIKE (Higher) at the 5\% "
           r"level, with the two-sided test of Sec.~\ref{sec:significance}, on the dates and "
           r"forecasts of Tab.~\ref{tab:log_input}.")
    lines = [r"\begin{table}[htbp]", r"\centering", r"\singlespacing", r"\caption{" + cap + "}",
             r"\label{tab:log_vs_vol_dm}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{$h=1$} & \multicolumn{2}{c}{$h=5$} & \multicolumn{2}{c}{$h=22$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & Lower & Higher & Lower & Higher & Lower & Higher \\", r"\midrule"]
    for m in TSFM_ROWS:
        cells = []
        for h in H:
            row = g.loc[(m, h)]
            if row.n_assets != len(ALL):
                raise ValueError(f"{m} h={h}: {row.n_assets} assets")
            cells += [f"{row.lower:.0f}", f"{row.higher:.0f}"]
        lines.append(f"Log-{NAME[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--log-forecasts", type=Path, default=LOG_FORECASTS,
                    help="log-volatility forecasts of the nine foundation models")
    ap.add_argument("--check", type=Path, default=None,
                    help="log_input_long.csv of compute_log_input.py, which the ratios must reproduce")
    ap.add_argument("--metrics-out", type=Path, default=METRICS)
    ap.add_argument("--out", type=Path, default=TABLES)
    a = ap.parse_args()
    res = run(a.forecasts, a.log_forecasts, load_sigma_all())
    if a.check is not None:
        check_against(res, a.check)
    a.metrics_out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.metrics_out / "dm_log_vs_vol_long.csv", index=False)
    summ = summarize(res)
    summ.to_csv(a.metrics_out / "dm_log_vs_vol_summary.csv", index=False)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table_log_vs_vol_dm.tex").write_text(table(summ))
    print("written to", a.out / "table_log_vs_vol_dm.tex")


if __name__ == "__main__":
    main()
