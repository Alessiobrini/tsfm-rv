"""Refit Log-HAR, ARMA or ARFIMA at every origin and record what the log-normal expected variance
needs, the input of the log-normal column of the forecast-level table.

At origin position i the model is fit on sigma[i - 1000 : i], the 1,000 most recent observations,
as in the main forecasts (fixed window, re-estimated daily). For each horizon the script records the
volatility forecast clipped to the window bounds, which must reproduce the main forecast file, and
the moments of log volatility:

  Log_HAR  the regressors are logs of the averages of sigma, as in models/har.py with use_log. The
           conditional mean path m of log sigma carries the recursion on log sigma, with the future
           sigmas inside the averages replaced by exp(m). The h-step forecast-error variance v_h is
           s2 * sum_{j<h} psi_j^2, with psi the MA weights of the AR(22) linearization of the
           regressors and s2 the one-step residual variance. The column lognormal_var holds
           exp(2 m_h + 2 v_h), exact at h = 1 and a first-order approximation at h > 1.
  ARMA     m_h, s2 and v_h = s2 * sum_{j<h} psi_j^2, with psi the MA(inf) weights of the fitted ARMA.
  ARFIMA   m_h, s2 and v_h = s2 * sum_{j<h} c_j^2, with c the convolution of the (1-L)^{-d} weights
           with psi.

Output: {model}_{ticker}_h{h}.csv and repro_{model}_{ticker}.csv, the largest relative gap to the
main forecast file, in the output folder. One asset and model per call, so the 150 calls can run as
a job array.

Usage:  python code/refit_lognormal_moments.py TICKER {Log_HAR,ARMA,ARFIMA} [--forecasts DIR] [--out DIR]
                                               [--skip-existing]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import arfima_cfg  # noqa: E402
from evaluation.forecast_sets import FORECASTS, H, ROOT, load_sigma_all, read  # noqa: E402
from features import align_features_target, build_har_features, build_target  # noqa: E402
from forecasting.bounds import window_bounds  # noqa: E402
from models.arfima import ARFIMAModel  # noqa: E402
from models.arma import ARMAModel  # noqa: E402
from models.har import HARModel  # noqa: E402

OUT = ROOT / "results" / "lognormal_refit"
TRAIN, W, M = 1000, 5, 22
HMAX = max(H)


# ---------------------------------------------------------------- Log-HAR
def fit_log_har(fit_series):
    """The one-step Log-HAR on ``fit_series`` as the forecasting engine fits it:
    (b0, bd, bw, bm, s2)."""
    feats = build_har_features(fit_series, monthly_lag=M, weekly_lag=W)
    tgt = build_target(fit_series, horizon=1, target_kind="point")
    X, y = align_features_target(feats, tgt)
    mdl = HARModel(use_log=True)
    mdl.fit(X, y)
    b0, bd, bw, bm, _, _ = mdl.linear_coef()
    return b0, bd, bw, bm, float(mdl._resid_var)


def ar22_psi(bd, bw, bm, n):
    """MA weights of the AR(22) in log sigma that the HAR regressors imply."""
    phi = np.zeros(M + 1)
    phi[1] = bd + bw / W + bm / M
    phi[2:W + 1] = bw / W + bm / M
    phi[W + 1:M + 1] = bm / M
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        k = np.arange(1, min(j, M) + 1)
        psi[j] = np.dot(phi[k], psi[j - k])
    return psi


def rollout_forecast(vals_hist, coefs, s2, hmax):
    """The iterated Log-HAR forecast on the volatility scale, exp(lp + s2 / 2) at every step."""
    b0, bd, bw, bm = coefs
    work = list(vals_hist)
    out = []
    for _ in range(hmax):
        d, w, m = work[-1], float(np.mean(work[-W:])), float(np.mean(work[-M:]))
        lp = b0 + bd * np.log(max(d, 1e-10)) + bw * np.log(max(w, 1e-10)) + bm * np.log(max(m, 1e-10))
        nxt = float(np.exp(lp + 0.5 * s2))
        work.append(nxt)
        out.append(nxt)
    return np.array(out)


def rollout_log_mean(vals_hist, coefs, hmax):
    """The conditional mean path of log sigma, with exp(m) inside the averages."""
    b0, bd, bw, bm = coefs
    work = list(vals_hist)
    out = []
    for _ in range(hmax):
        d, w, m = work[-1], float(np.mean(work[-W:])), float(np.mean(work[-M:]))
        lp = b0 + bd * np.log(d) + bw * np.log(w) + bm * np.log(m)
        work.append(float(np.exp(lp)))
        out.append(lp)
    return np.array(out)


def log_har_origin(sig, i):
    b0, bd, bw, bm, s2 = fit_log_har(sig.iloc[i - TRAIN:i])
    hist = sig.values[i - M:i]
    fc = rollout_forecast(hist, (b0, bd, bw, bm), s2, HMAX)
    m = rollout_log_mean(hist, (b0, bd, bw, bm), HMAX)
    v = s2 * np.cumsum(ar22_psi(bd, bw, bm, HMAX) ** 2)
    return fc, {"lognormal_var": np.exp(2 * m + 2 * v)}


# ---------------------------------------------------------------- ARMA and ARFIMA
def series_model(model):
    if model == "ARMA":
        return ARMAModel(max_p=2, max_q=2, use_log=True, ic="bic")
    return ARFIMAModel(p=arfima_cfg.max_ar, q=arfima_cfg.max_ma, use_log=arfima_cfg.use_log_rv,
                       d_method=arfima_cfg.d_method)


def arma_ma_weights(res, n):
    from statsmodels.tsa.arima_process import arma2ma
    ar = np.r_[1.0, -np.asarray(res.arparams)] if len(res.arparams) else np.array([1.0])
    ma = np.r_[1.0, np.asarray(res.maparams)] if len(res.maparams) else np.array([1.0])
    return arma2ma(ar, ma, lags=n)


def series_origin(sig, i, model):
    mdl = series_model(model)
    mdl.fit(sig.iloc[i - TRAIN:i])
    fc = np.asarray(mdl.predict(steps=HMAX), dtype=float)   # exp(m + s2 / 2)
    res = mdl._result
    psi_arma = arma_ma_weights(res, HMAX)
    if model == "ARMA":
        s2 = float(res.mse)
        c = psi_arma
    else:
        s2 = float(mdl._sigma2)
        c = np.convolve(mdl._psi[:HMAX], psi_arma)[:HMAX]
    m = np.log(fc) - 0.5 * s2
    return fc, {"m": m, "s2": np.full(HMAX, s2), "v": s2 * np.cumsum(c ** 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ticker")
    ap.add_argument("model", choices=["Log_HAR", "ARMA", "ARFIMA"])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS,
                    help="main point forecasts, against which the refit is checked")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip the asset and model when its repro file exists")
    a = ap.parse_args()
    t, model = a.ticker, a.model
    a.out.mkdir(parents=True, exist_ok=True)
    if a.skip_existing and (a.out / f"repro_{model}_{t}.csv").exists():
        print("exists, skipping", model, t)
        return
    t0 = time.time()
    sig = load_sigma_all([t])[t]
    lo, hi = window_bounds(sig)
    n = len(sig)
    recs = {h: [] for h in H}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(TRAIN, n):
            fc, extra = log_har_origin(sig, i) if model == "Log_HAR" else series_origin(sig, i, model)
            d = sig.index[i]
            for h in H:
                if i + h - 1 >= n:
                    continue
                k = h - 1
                rec = dict(date=d, pos=i, actual=float(sig.values[i + h - 1]))
                rec.update({c: float(x[k]) for c, x in extra.items()})
                rec.update(lo=float(lo.loc[d]), hi=float(hi.loc[d]),
                           forecast=float(np.clip(fc[k], lo.loc[d], hi.loc[d])))
                recs[h].append(rec)
    log = []
    for h in H:
        df = pd.DataFrame(recs[h]).set_index("date")
        fin = read(a.forecasts / f"{model}_{t}_h{h}.csv")
        c = df.index.intersection(fin.index)
        gap = np.abs(df.loc[c, "forecast"].values - fin.loc[c, "forecast"].values)
        log.append(dict(ticker=t, model=model, h=h, n_refit=len(df), n_main=len(fin), n_common=len(c),
                        maxabs=float(gap.max()), maxrel=float((gap / fin.loc[c, "forecast"].values).max()),
                        actual_ok=bool(np.allclose(df.loc[c, "actual"].values, fin.loc[c, "actual"].values,
                                                   rtol=1e-10, atol=0))))
        df.to_csv(a.out / f"{model}_{t}_h{h}.csv")
    lg = pd.DataFrame(log)
    lg.to_csv(a.out / f"repro_{model}_{t}.csv", index=False)
    print(lg.to_string())
    print(f"{model} {t}: {n - TRAIN} origins in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
