"""Loading and scoring helpers shared by the scripts behind the appendix and robustness tables.

Every script scores the forecasts the same way as the main comparison. Each volatility forecast is
clipped to the window bounds of `forecasting/bounds.py`, every model is scored on the dates common
to the models of the comparison, QLIKE is computed on the variance scale, and per-asset ratios to
Log-HAR are averaged across the 50 assets.

The forecast files follow the layout of `results/volare/forecasts/`: one file per model, asset and
horizon, named `{model}_{ticker}_h{h}.csv`, indexed by the first day of the target, with the
columns `actual` and `forecast` on the volatility scale.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import VOLARE_ALL_TICKERS, VOLARE_FX_TICKERS as FX, VOLARE_FUTURES_TICKERS as FUT
from evaluation.targets import rv_series_all
from forecasting.bounds import clip_to_window

ROOT = Path(__file__).resolve().parent.parent.parent
FORECASTS = ROOT / "results" / "volare" / "forecasts"
FORECASTS_AVG = ROOT / "results" / "volare_avg" / "forecasts"
METRICS = ROOT / "results" / "volare" / "metrics"
TABLES = ROOT / "paper" / "tables"

ALL = list(VOLARE_ALL_TICKERS)
H = (1, 5, 22)
L, T = "Log_HAR", "ttm"
# The 17 models of the comparison, grouped as the scripts load and test them
ECON = ["HAR", "Log_HAR", "ARFIMA", "ARMA", "MEM"]
DIRECT = ["HAR_J", "HAR_RS", "HARQ"]
TSFM = ["chronos_bolt_small", "chronos_bolt_base", "timesfm_2_5", "moirai_2_0_small",
        "moirai_moe_small", "lag_llama", "toto", "sundial", "ttm"]
MODELS = ECON + DIRECT + TSFM
# Foundation models in the row order of the paper's tables
TSFM_ROWS = ["chronos_bolt_small", "chronos_bolt_base", "moirai_2_0_small", "moirai_moe_small",
             "lag_llama", "timesfm_2_5", "toto", "sundial", "ttm"]
NAME = {"HAR": "HAR", "ARFIMA": "ARFIMA", "ARMA": "ARMA", "MEM": "MEM", "HAR_J": "HAR-J",
        "HAR_RS": "HAR-RS", "HARQ": "HARQ", "Log_HAR": "Log-HAR",
        "chronos_bolt_small": "Chronos-Bolt-S", "chronos_bolt_base": "Chronos-Bolt-B",
        "moirai_2_0_small": "Moirai-2.0-S", "moirai_moe_small": "Moirai-MoE-S",
        "lag_llama": "Lag-Llama", "timesfm_2_5": "TimesFM-2.5", "toto": "Toto",
        "sundial": "Sundial", "ttm": "TTM"}


def asset_class(t):
    return "fx" if t in FX else "futures" if t in FUT else "stocks"


def load_sigma_all(tickers=None):
    """Daily volatility sqrt(RV) per asset, missing days dropped: the series the models forecast."""
    return {t: np.sqrt(rv) for t, rv in rv_series_all(tickers).items()}


def read(path, round_trip=True):
    """One forecast file, indexed by date."""
    if round_trip:
        return pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_clipped(folder, models, t, h, sigma, round_trip=True, suffix=""):
    """{model: file} for one asset and horizon, each forecast clipped to the window bounds."""
    out = {}
    for m in models:
        d = read(Path(folder) / f"{m}_{t}_h{h}{suffix}.csv", round_trip)
        d["forecast"] = clip_to_window(d["forecast"], sigma)
        out[m] = d
    return out


def common_index(frames, column=None):
    """Dates present in every frame: rows without a missing value in ``column``, or in any column
    when ``column`` is None."""
    idx = None
    for x in frames.values():
        s = set((x if column is None else x[column]).dropna().index)
        idx = s if idx is None else idx & s
    return pd.DatetimeIndex(sorted(idx))


def qlike_series(rv, var):
    """Daily QLIKE loss of the variance forecast ``var`` for the realized variance ``rv``."""
    r = rv / var
    return r - np.log(r) - 1.0


def hac_t(x, lags, z=None):
    """Newey-West t-statistic of the mean of ``x``, or of the slope on ``z`` when it is given."""
    X = np.ones((len(x), 1)) if z is None else sm.add_constant(z)
    res = sm.OLS(np.asarray(x), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[-1])


def equal_weight_series(panel: pd.DataFrame) -> pd.Series:
    """Daily series s_t = sum over assets of d_it / (N T_i), with T_i the number of dates of asset i
    and N the number of assets. Its sum over dates is the mean across assets of each asset's mean
    difference, so a test on it weights every asset equally."""
    n_dates = panel.notna().sum()
    return (panel / (len(panel.columns) * n_dates)).sum(axis=1, min_count=1).dropna()


def rescale_level(df, h, window=250, burn=22):
    """Variance forecast multiplied in real time by the mean of RV / F over the last ``window`` rows
    q <= p - h of the file, the rows whose targets are observed by origin p. Unchanged until
    ``burn`` such rows exist."""
    rv = df["actual"].values ** 2
    F = df["forecast"].values ** 2
    ratio = rv / F
    c = np.ones(len(df))
    for p in range(len(df)):
        hi = p - h + 1
        if hi >= burn:
            c[p] = ratio[max(0, hi - window):hi].mean()
    return pd.Series(F * c, index=df.index)
