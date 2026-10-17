"""gen_descriptive_stats.py — Descriptive-statistics table for daily RV (Table tab:descriptive).

Recomputes the per-asset moments of 5-minute realized variance (rv5) directly
from the raw VOLARE files, using the STANDARD excess-kurtosis definition
(Fisher, bias-corrected sample estimator = pandas Series.kurt(); normal => 0).
Skewness uses the matching sample estimator (pandas Series.skew()).

Equity panel = cross-sectional average of each statistic across the 40 stocks;
FX and futures are reported per asset. Mean and Median are in units of 1e-4
(decimal squared returns). rho_k is the sample autocorrelation at lag k.

Prints the table rows and the kurtosis range for the prose. Verification mode
also prints the non-excess (Pearson) kurtosis so the change is auditable.

Usage:  python code/gen_descriptive_stats.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import (VOLARE_STOCKS_FILE, VOLARE_FOREX_FILE, VOLARE_FUTURES_FILE,
                    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS)

RV = "rv5"
FX_NAMES = {t: t for t in VOLARE_FX_TICKERS}
FUT_NAMES = {"C": "Corn (C)", "CL": "Crude Oil (CL)", "ES": "E-mini S\\&P (ES)",
             "GC": "Gold (GC)", "NG": "Natural Gas (NG)"}


def asset_stats(s):
    """Moments for one rv5 series s (already dropna). Kurtosis is EXCESS (Fisher)."""
    return {
        "n": int(len(s)),
        "mean": s.mean() * 1e4,
        "median": s.median() * 1e4,
        "skew": s.skew(),                 # sample skewness (G1)
        "kurt": s.kurt(),                 # sample EXCESS kurtosis (G2, Fisher)
        "kurt_pearson": s.kurt() + 3.0,   # non-excess, for audit only
        "rho1": s.autocorr(1),
        "rho22": s.autocorr(22),
    }


def load_long(path):
    d = pd.read_csv(path, usecols=["date", "symbol", RV])
    d = d.dropna(subset=[RV])
    d = d[d[RV] > 0]
    return d


def per_ticker(df, tickers):
    out = {}
    for t in tickers:
        s = df.loc[df["symbol"] == t, RV].reset_index(drop=True)
        if len(s):
            out[t] = asset_stats(s)
    return out


def main():
    stocks = per_ticker(load_long(VOLARE_STOCKS_FILE), VOLARE_STOCK_TICKERS)
    fx = per_ticker(load_long(VOLARE_FOREX_FILE), VOLARE_FX_TICKERS)
    fut = per_ticker(load_long(VOLARE_FUTURES_FILE), VOLARE_FUTURES_TICKERS)

    # equity panel: cross-sectional average of each statistic
    keys = ["n", "mean", "median", "skew", "kurt", "kurt_pearson", "rho1", "rho22"]
    eq_avg = {k: float(np.mean([stocks[t][k] for t in stocks])) for k in keys}

    def row(label, st, n_int=True):
        n = f"{int(round(st['n'])):,}"
        return (f"{label} & {n} & {st['mean']:.2f} & {st['median']:.2f} & "
                f"{st['skew']:.1f} & {fmt_k(st['kurt'])} & "
                f"{fmt_ac(st['rho1'])} & {fmt_ac(st['rho22'])} \\\\")

    def fmt_k(v):
        return f"{v:,.0f}"

    def fmt_ac(v):
        s = f"{v:.3f}"
        return s.replace("-", "$-$")

    print("=== EXCESS-KURTOSIS (standard) table rows ===\n")
    print("Panel A: Equities (cross-sectional average, 40 stocks)")
    print(row("Average", eq_avg))
    print("\nPanel B: FX")
    for t in VOLARE_FX_TICKERS:
        print(row(t, fx[t]))
    print("\nPanel C: Futures")
    for t in VOLARE_FUTURES_TICKERS:
        print(row(FUT_NAMES[t], fut[t]))

    # kurtosis range for prose
    all_k = ([eq_avg["kurt"]] + [fx[t]["kurt"] for t in fx] + [fut[t]["kurt"] for t in fut])
    print(f"\nEXCESS kurtosis range (panel rows): {min(all_k):.0f} to {max(all_k):.0f}")

    # audit: Pearson (non-excess) values for the same rows
    print("\n=== AUDIT: non-excess (Pearson) kurtosis, current-table convention check ===")
    print(f"Equity avg: excess={eq_avg['kurt']:.0f}  pearson={eq_avg['kurt_pearson']:.0f}")
    for t in VOLARE_FX_TICKERS:
        print(f"{t}: excess={fx[t]['kurt']:.0f}  pearson={fx[t]['kurt_pearson']:.0f}")
    for t in VOLARE_FUTURES_TICKERS:
        print(f"{FUT_NAMES[t]}: excess={fut[t]['kurt']:.0f}  pearson={fut[t]['kurt_pearson']:.0f}")


if __name__ == "__main__":
    main()
