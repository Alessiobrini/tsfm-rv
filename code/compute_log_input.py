"""The nine foundation models applied to log volatility, the input of the log-input table.

Each foundation model also forecasts log volatility over the same context (run_foundation_volare.py
--scale logvol, cluster/run_tsfm_logvol.slurm). Its log forecast m is clipped to the log of the
window bounds and converted back to volatility with the rule of Log-HAR, exp(m + v / 2). Here v is
the variance of the model's own one-step errors in log volatility, from the clipped one-step
forecasts, over the 1,000 origins before the forecast origin, or over all earlier origins when fewer
exist, with v = 0 until 22 errors are observed. The volatility forecast is then clipped to the window
bounds and squared for QLIKE. Every model is scored on the dates common to the 17 models and both
inputs, and the per-asset QLIKE ratio to Log-HAR is averaged across assets.

The spec column of the output is "vol" for the main forecasts, "log" for the log versions and
"log_raw" for the log versions without the first clip. Output: log_input_long.csv (per asset) and
log_input_summary.csv (mean ratio and number of assets with a ratio below one).

Usage:  python code/compute_log_input.py [--forecasts DIR] [--log-forecasts DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import (ALL, FORECASTS, L, METRICS, MODELS, ROOT, TSFM,  # noqa: E402
                                      load_clipped, load_sigma_all, qlike_series as ql, read)
from forecasting.bounds import clip_to_window, window_bounds  # noqa: E402

LOG_FORECASTS = ROOT / "results" / "tsfm_logvol" / "forecasts"
VAR_WINDOW, VAR_MIN = 1000, 22


def past_error_variance(one_step: pd.DataFrame, window=VAR_WINDOW, min_periods=VAR_MIN) -> pd.Series:
    """Variance of the one-step log errors e_q = actual_q - forecast_q over the ``window`` origins q
    before each origin date, indexed by that date. The target of origin q is dated q, so every error
    used at origin t is observed by t. NaN until ``min_periods`` errors exist."""
    e = one_step["actual"] - one_step["forecast"]
    return e.rolling(window, min_periods=min_periods).var().shift(1)


def clip_log(df: pd.DataFrame, sig: pd.Series) -> pd.DataFrame:
    """The log forecast clipped to the log of the volatility window bounds at its origin."""
    lo, hi = window_bounds(sig)
    lo, hi = lo.reindex(df.index), hi.reindex(df.index)
    if lo.isna().any() or hi.isna().any():
        raise ValueError("a forecast date has no bounds")
    out = df.copy()
    out["forecast"] = df["forecast"].clip(lower=np.log(lo), upper=np.log(hi))
    return out


def back_to_vol(df: pd.DataFrame, v: pd.Series) -> pd.Series:
    """exp(m + v / 2) at each origin, with v from past_error_variance (0 before its burn-in)."""
    vv = v.reindex(df.index)
    if vv.index.has_duplicates or len(vv) != len(df):
        raise ValueError("origin dates do not align")
    return np.exp(df["forecast"] + 0.5 * vv.fillna(0.0))


def load_log(folder, t, h):
    return {m: read(Path(folder) / f"{m}_{t}_h{h}.csv") for m in TSFM}


def common_forecast_dates(frames):
    idx = None
    for x in frames:
        s = set(x["forecast"].dropna().index)
        idx = s if idx is None else idx & s
    return pd.DatetimeIndex(sorted(idx))


def run(fc, log_fc, sigma_all, horizons=(1, 5, 22)):
    rows = []
    for t in ALL:
        sig = sigma_all[t]
        raw1 = load_log(log_fc, t, 1)
        logs1 = {m: clip_log(d, sig) for m, d in raw1.items()}
        var1 = {m: past_error_variance(logs1[m]) for m in TSFM}
        var1_raw = {m: past_error_variance(raw1[m]) for m in TSFM}
        for h in horizons:
            vol = load_clipped(fc, MODELS, t, h, sig)
            raws = raw1 if h == 1 else load_log(log_fc, t, h)
            logs = logs1 if h == 1 else {m: clip_log(d, sig) for m, d in raws.items()}
            idx = common_forecast_dates(list(vol.values()) + list(logs.values()))
            y = vol[L].loc[idx, "actual"].values
            for m, d in vol.items():
                if not np.allclose(d.loc[idx, "actual"].values, y, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h={h} {m}: actuals differ from Log-HAR")
            for m, d in logs.items():
                if not np.allclose(np.exp(d.loc[idx, "actual"].values), y, rtol=1e-8, atol=0):
                    raise ValueError(f"{t} h={h} {m}: log actuals differ from Log-HAR")
            rv = y ** 2
            base = ql(rv, vol[L].loc[idx, "forecast"].values ** 2).mean()
            for m, d in vol.items():
                rows.append(dict(ticker=t, h=h, model=m, spec="vol", n=len(idx),
                                 ratio=ql(rv, d.loc[idx, "forecast"].values ** 2).mean() / base))
            for spec, ds, vs in (("log", logs, var1), ("log_raw", raws, var1_raw)):
                for m, d in ds.items():
                    f = clip_to_window(back_to_vol(d, vs[m]), sig).loc[idx].values
                    rows.append(dict(ticker=t, h=h, model=m, spec=spec, n=len(idx),
                                     ratio=ql(rv, f ** 2).mean() / base,
                                     mean_v=float(vs[m].reindex(idx).mean())))
        print("done", t, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--log-forecasts", type=Path, default=LOG_FORECASTS,
                    help="log-volatility forecasts of the nine foundation models")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = run(a.forecasts, a.log_forecasts, load_sigma_all())
    res.to_csv(a.out / "log_input_long.csv", index=False)
    s = (res.groupby(["model", "spec", "h"])
            .agg(ratio=("ratio", "mean"), beat=("ratio", lambda r: int((r < 1).sum())),
                 n_assets=("ratio", "size"), mean_v=("mean_v", "mean"))
            .reset_index())
    s.to_csv(a.out / "log_input_summary.csv", index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
