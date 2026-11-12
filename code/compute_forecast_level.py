"""Forecast level and two level adjustments, the input of the forecast-level table.

For each asset and horizon, on the dates common to the 17 models:

  level    mean of RV / F^2, with F the volatility forecast as issued (clipped to the window bounds).
  issued   QLIKE ratio to Log-HAR as issued, which equals the main loss-ratio table.
  lognorm  Log-HAR, ARFIMA and ARMA scored on their log-normal expected variance
           exp(2 m_h + 2 v_h), clipped to the squared window bounds, with m_h and v_h from
           refit_lognormal_moments.py, and the other models unchanged. Ratios to the adjusted Log-HAR.
  recal    every model's variance forecast multiplied by its mean RV / F^2 over the last 250
           forecasts of its own file whose targets are observed by the origin
           (evaluation.forecast_sets.rescale_level). Ratios to the rescaled Log-HAR.

The checks file records the largest relative gap between each refit's volatility forecast and the
main forecast file, so the adjustment is applied to the forecasts the main tables score.

Output: forecast_level_long.csv, forecast_level_checks.csv and forecast_level_summary.csv.

Usage:  python code/compute_forecast_level.py [--forecasts DIR] [--refit DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import (ALL, FORECASTS, L, METRICS, MODELS, load_sigma_all,  # noqa: E402
                                      qlike_series as ql, read, rescale_level)
from evaluation.targets import common_dates  # noqa: E402
from forecasting.bounds import window_bounds  # noqa: E402
from refit_lognormal_moments import OUT as REFIT  # noqa: E402

LOGM = ["Log_HAR", "ARFIMA", "ARMA"]


def lognormal_variance(refit, t, h, idx, lo2, hi2):
    """{model: log-normal expected variance on idx, clipped to the squared window bounds} and
    {model: refit forecast on idx}."""
    out, fc = {}, {}
    for m in LOGM:
        p = Path(refit) / f"{m}_{t}_h{h}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing: run refit_lognormal_moments.py {t} {m} first")
        d = read(p).loc[idx]
        ln = d["lognormal_var"].values if m == L else np.exp(2 * d["m"].values + 2 * d["v"].values)
        out[m] = np.clip(ln, lo2, hi2)
        fc[m] = d["forecast"].values
    return out, fc


def run(folder, refit, sigma_all, horizons=(1, 5, 22)):
    rows, checks = [], []
    for t in ALL:
        sig = sigma_all[t]
        lo, hi = window_bounds(sig)
        for h in horizons:
            full = {m: read(Path(folder) / f"{m}_{t}_h{h}.csv") for m in MODELS}
            idx = common_dates(full)
            rv = full[L].loc[idx, "actual"].values ** 2
            for m, d in full.items():
                if not np.allclose(d.loc[idx, "actual"].values ** 2, rv, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h{h} {m}: actuals differ from Log-HAR")
            F2 = {m: d.loc[idx, "forecast"].values ** 2 for m, d in full.items()}
            lo2, hi2 = lo.reindex(idx).values ** 2, hi.reindex(idx).values ** 2
            ln, fc = lognormal_variance(refit, t, h, idx, lo2, hi2)
            checks.append(dict(ticker=t, h=h, **{f"gap_{m}": float(np.max(np.abs(
                fc[m] / full[m].loc[idx, "forecast"].values - 1))) for m in LOGM}))
            rc = {m: rescale_level(d, h).loc[idx].values for m, d in full.items()}
            q_iss = {m: ql(rv, F2[m]).mean() for m in MODELS}
            q_ln = dict(q_iss, **{m: ql(rv, ln[m]).mean() for m in LOGM})
            q_rc = {m: ql(rv, rc[m]).mean() for m in MODELS}
            for m in MODELS:
                rows.append(dict(ticker=t, h=h, model=m, n=len(idx),
                                 level=float(np.mean(rv / F2[m])),
                                 issued=q_iss[m] / q_iss[L],
                                 lognorm=q_ln[m] / q_ln[L],
                                 lognorm_vs_issued_LH=q_ln[m] / q_iss[L],
                                 recal=q_rc[m] / q_rc[L],
                                 recal_vs_issued_LH=q_rc[m] / q_iss[L]))
        print("done", t, flush=True)
    return pd.DataFrame(rows), pd.DataFrame(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--refit", type=Path, default=REFIT, help="output folder of refit_lognormal_moments.py")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res, chk = run(a.forecasts, a.refit, load_sigma_all())
    res.to_csv(a.out / "forecast_level_long.csv", index=False)
    chk.to_csv(a.out / "forecast_level_checks.csv", index=False)
    cols = ["level", "issued", "lognorm", "lognorm_vs_issued_LH", "recal", "recal_vs_issued_LH"]
    s = res.groupby(["h", "model"])[cols].mean()
    beat = res.assign(b_iss=res.issued < 1, b_rc=res.recal < 1).groupby(["h", "model"])[["b_iss", "b_rc"]].sum()
    s.join(beat).reset_index().to_csv(a.out / "forecast_level_summary.csv", index=False)
    print(chk.drop(columns=["ticker"]).groupby("h").max().to_string())
    print("written to", a.out)


if __name__ == "__main__":
    main()
