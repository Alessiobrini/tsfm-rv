"""
compute_bh_dm.py — Benjamini--Hochberg-adjusted Diebold--Mariano win rates.

Backs the multiple-comparison footnote in Sec.~6: the win rates in the DM table
are unadjusted; here we apply a Benjamini--Hochberg false-discovery-rate
correction at 5% within each model's 800 pairwise tests (50 assets x 16
opponents) and report the adjusted win rate. A "win" requires the DM test to
survive BH at 5% AND the model to have the lower per-asset QLIKE.

Reads results/volare/metrics/dm_pvalues_<ticker>_h<h>.csv (two-sided DM p-value
matrices) and metrics_by_asset_h<h>.csv (per-asset QLIKE). Prints raw and
BH-adjusted win rates for the requested models.

    python code/compute_bh_dm.py            # TTM and Log-HAR, h = 1, 5, 22
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from config import (  # noqa: E402
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)

MET = ROOT / "results" / "volare" / "metrics"
TICKERS = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
HORIZONS = [1, 5, 22]
ALPHA = 0.05


def win_rates(model, h):
    """Return (n_tests, raw_win_%, bh_win_%) for one model at horizon h."""
    q = pd.read_csv(MET / f"metrics_by_asset_h{h}.csv")
    qd = {(r.ticker, r.model): r.QLIKE for r in q.itertuples()}
    pv, better = [], []
    for t in TICKERS:
        f = MET / f"dm_pvalues_{t}_h{h}.csv"
        if not f.exists():
            continue
        M = pd.read_csv(f, index_col=0)
        if model not in M.index:
            continue
        for opp in M.columns:
            if opp == model:
                continue
            p = M.loc[model, opp]
            if pd.isna(p):
                continue
            pv.append(float(p))
            better.append(qd.get((t, model), np.inf) < qd.get((t, opp), np.inf))
    pv, better = np.array(pv), np.array(better)
    raw = float(np.mean((pv < ALPHA) & better) * 100)
    rej = multipletests(pv, alpha=ALPHA, method="fdr_bh")[0]
    bh = float(np.mean(rej & better) * 100)
    return len(pv), raw, bh


def main():
    models = sys.argv[1:] or ["ttm", "Log_HAR"]
    for m in models:
        print(f"--- {m} ---")
        for h in HORIZONS:
            n, raw, bh = win_rates(m, h)
            print(f"  h={h:2d}: n={n} raw_win={raw:.1f}% BH_win={bh:.1f}%")


if __name__ == "__main__":
    main()
