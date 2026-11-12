"""Realized utility of the volatility forecasts, the input of the economic-value table, following
Bollerslev, Hood, Huss and Pedersen (2018, RFS 31, Sec. 5 and Table 6).

An investor with mean-variance preferences holds one asset with a constant Sharpe ratio SR = 0.4 and
relative risk aversion gamma = 2. With a variance forecast F_t for the holding period that starts at
origin t, the position is x_t = (SR / gamma) / sqrt(252 F_t), a 20% annual volatility target. With
RV_t the mean daily realized variance over the holding period, the realized utility per year is

    u_t = SR x_t sqrt(252 RV_t) - (gamma / 2) x_t^2 252 RV_t  =  8% sqrt(RV_t / F_t) - 4% RV_t / F_t,

which equals 4% for a perfect forecast. Positions are rebalanced at every origin (daily), so at
h = 5 and 22 the holding windows overlap. Three cost cases:
  zero     no trading costs.
  full     the asset's proportional bid-ask spread times |x_t - x_{t-1}| each day.
  gradual  the position moves 15% of the way to x_t each day (Garleanu and Pedersen, 2013), with the
           same costs on its changes.
A cost per day is multiplied by 252 to match the annual utility.

Spreads: each asset's median bid-ask spread over the last nine months of the sample (2025-05-01 to
2026-01-30), as a fraction of the price. Futures: one minimum price increment of the CME contract
divided by the median close. FX: one pip divided by the median close. Stocks: the median daily
time-weighted quoted spread, read from the file given with --stock-spreads. That file is not
distributed with the code, since it comes from licensed data: the WRDS TAQ intraday indicators
(tables taqmsec.wrds_iid_2025 and taqmsec.wrds_iid_2026, column quotedspread_percent_tw, already a
fraction of the price), with GOOGL under sym_root GOOG and sym_suffix L. It is a CSV indexed by
ticker, with one row for each of the 40 stocks and a column `quoted` that holds the median over
the nine months above.

Forecasts as in the main comparison: the 17 models and the equal-weight average of the TTM and
Log-HAR volatility forecasts, clipped to the window bounds, on dates common to every model. h = 1
uses the point forecasts. h = 5 and 22 use the averaged-target forecasts, squared, and the realized
side is the mean of raw daily RV over the h days from the origin, the same for every model.

Output: economic_value_spreads.csv, economic_value_per_asset.csv, economic_value_tests.csv and
economic_value_summary.csv.

Usage:  python code/compute_economic_value.py --stock-spreads CSV [--forecasts DIR] [--forecasts-avg DIR]
                                              [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE, VOLARE_FUTURES_TICKERS as FUT,  # noqa: E402
                    VOLARE_FX_TICKERS as FX)
from evaluation.forecast_sets import (ALL, FORECASTS, FORECASTS_AVG, L, METRICS, MODELS,  # noqa: E402
                                      asset_class, equal_weight_series, hac_t, load_sigma_all, read)
from evaluation.targets import holding_rv  # noqa: E402
from forecasting.bounds import clip_to_window  # noqa: E402

SR, GAMMA, ANN = 0.4, 2.0, 252
TARGET = SR / GAMMA
SPEED = 0.15
COMBO = "ttm+Log_HAR"


# ---------------------------------------------------------------- utility and positions
def utility(rv, F):
    """Realized utility per year of the position targeted with forecast F."""
    r = rv / F
    return (SR ** 2 / GAMMA) * (np.sqrt(r) - 0.5 * r)


def positions(F):
    return TARGET / np.sqrt(ANN * F)


def utility_of_position(x, rv):
    return SR * x * np.sqrt(ANN * rv) - 0.5 * GAMMA * x ** 2 * ANN * rv


def gradual(x, speed=SPEED):
    g = np.empty_like(x)
    g[0] = x[0]
    for i in range(1, len(x)):
        g[i] = g[i - 1] + speed * (x[i] - g[i - 1])
    return g


def net_series(F, rv, spread):
    """Daily net utility per year for the three cost cases. The first position is taken as given."""
    x = positions(F)
    dx = np.abs(np.diff(x, prepend=x[0]))
    g = gradual(x)
    dg = np.abs(np.diff(g, prepend=g[0]))
    u = utility(rv, F)
    return {"zero": u, "full": u - ANN * spread * dx,
            "gradual": utility_of_position(g, rv) - ANN * spread * dg}


# ---------------------------------------------------------------- spreads
TICK = {"ES": 0.25, "CL": 0.01, "GC": 0.10, "NG": 0.001, "C": 0.25}      # C quoted in cents per bushel
PIP = {"AUDUSD": 1e-4, "EURUSD": 1e-4, "GBPUSD": 1e-4, "USDCAD": 1e-4, "USDJPY": 1e-2}
SPREAD_START, SPREAD_END = "2025-05-01", "2026-01-30"


def median_close(tickers):
    out = {}
    for path in (VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE):
        df = pd.read_csv(path, usecols=["date", "symbol", "close_price"], parse_dates=["date"])
        df = df[(df.date >= SPREAD_START) & (df.date <= SPREAD_END) & df.symbol.isin(tickers)]
        out.update(df.groupby("symbol").close_price.median().to_dict())
    return out


def spreads_all(stock_csv):
    taq = pd.read_csv(stock_csv, index_col=0)["quoted"]
    px = median_close(set(TICK) | set(PIP))
    s = {t: float(taq[t]) for t in taq.index}
    s.update({t: TICK[t] / px[t] for t in TICK})
    s.update({t: PIP[t] / px[t] for t in PIP})
    out = pd.Series(s, name="spread")
    missing = set(ALL) - set(out.index)
    if missing:
        raise ValueError(f"no spread for {sorted(missing)}")
    return out.loc[list(ALL)]


# ---------------------------------------------------------------- forecasts
def load(t, h, sig, dirs):
    folder = dirs["point" if h == 1 else "avg"]
    out = {}
    for m in MODELS:
        d = read(folder / f"{m}_{t}_h{h}.csv")
        out[m] = clip_to_window(d["forecast"], sig)
    idx = None
    for s in out.values():
        k = set(s.dropna().index)
        idx = k if idx is None else idx & k
    idx = pd.DatetimeIndex(sorted(idx))
    f = pd.DataFrame({m: s.loc[idx] for m, s in out.items()})
    f[COMBO] = 0.5 * (f["ttm"] + f[L])
    return f


def run(sigma_all, spreads, dirs, horizons=(1, 5, 22)):
    """Per-asset mean utilities, and DM t-statistics of the utility difference to Log-HAR: per asset,
    and on the cross-asset daily mean by asset class and for all assets (Newey-West, max(22, 2h) lags)."""
    per_asset, tests = [], []
    groups = [("all", list(ALL)), ("stocks", [x for x in ALL if asset_class(x) == "stocks"]),
              ("fx", list(FX)), ("futures", list(FUT))]
    for h in horizons:
        lags = max(22, 2 * h)
        diffs = {}                                   # (case, model) -> {ticker: Series of u - u_LogHAR}
        for t in ALL:
            sig = sigma_all[t]
            rv = sig ** 2
            f = load(t, h, sig, dirs)
            rbar = holding_rv(rv, f.index, h)
            if h == 1 and not np.allclose(rbar, rv.loc[f.index].values, rtol=1e-12):
                raise ValueError(f"{t}: the one-day holding RV differs from RV")
            series = {}
            for m in list(f.columns) + ["perfect"]:
                F = rbar if m == "perfect" else f[m].values ** 2
                series[m] = net_series(F, rbar, spreads[t])
                to = float(ANN * np.abs(np.diff(positions(F))).mean())
                for case, v in series[m].items():
                    per_asset.append(dict(h=h, ticker=t, cls=asset_class(t), model=m, case=case,
                                          utility=float(v.mean()), n=len(v), turnover=to))
            for m in series:
                if m == L:
                    continue
                for case in series[m]:
                    d = pd.Series(series[m][case] - series[L][case], index=f.index)
                    diffs.setdefault((case, m), {})[t] = d
                    tests.append(dict(h=h, case=case, model=m, group="asset", ticker=t, t=hac_t(d.values, lags)))
        for (case, m), dd in diffs.items():
            panel = pd.DataFrame(dd)
            for grp, tick in groups:
                s_t = equal_weight_series(panel[tick])
                tests.append(dict(h=h, case=case, model=m, group=grp, ticker="", t=hac_t(s_t.values, lags),
                                  diff_bp=1e4 * float(s_t.sum())))
        print("done h", h, flush=True)
    return pd.DataFrame(per_asset), pd.DataFrame(tests)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stock-spreads", type=Path, default=None,
                    help="CSV of the median quoted spreads of the 40 stocks (see the module docstring)")
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--forecasts-avg", type=Path, default=FORECASTS_AVG,
                    help="averaged-target forecasts of the 17 models")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    if a.stock_spreads is None or not a.stock_spreads.exists():
        ap.error("--stock-spreads must name the CSV of median quoted spreads of the 40 stocks, which "
                 "comes from the WRDS TAQ intraday indicators and is not distributed with the code. "
                 "The module docstring describes its content.")
    a.out.mkdir(parents=True, exist_ok=True)
    spreads = spreads_all(a.stock_spreads)
    spreads.to_csv(a.out / "economic_value_spreads.csv")
    pa, tv = run(load_sigma_all(), spreads, {"point": a.forecasts, "avg": a.forecasts_avg})
    pa.to_csv(a.out / "economic_value_per_asset.csv", index=False)
    tv.to_csv(a.out / "economic_value_tests.csv", index=False)
    s = pa.groupby(["h", "case", "model"]).agg(utility=("utility", "mean"),
                                                turnover=("turnover", "mean")).reset_index()
    s.to_csv(a.out / "economic_value_summary.csv", index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
