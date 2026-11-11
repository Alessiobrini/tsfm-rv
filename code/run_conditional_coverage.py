"""
run_conditional_coverage.py — Conditional coverage of raw and CQR intervals.

For every density file under <root>/<model>/<ticker>_h<h>.csv and each nominal
level (50/80/95%; 95% only for models with native tails), builds the miss
sequence of the central interval and tests:

    - Christoffersen (1998) unconditional coverage, independence and joint LR
    - dynamic-quantile regression of misses on a lagged miss and the
      volatility state (Engle and Manganelli 2004)
    - coverage in calm / normal / turbulent states (terciles of the state)

The volatility state at an origin is the mean RV over the 22 days before it,
known when the forecast is made. At h > 1 the tests use every h-th origin.

Three interval sets per file:
    raw             the model's own interval, full sample
    raw_cqr_window  the same interval on the dates CQR covers
    cqr             split-conformal CQR intervals from run_density_recalibration

Output: <root>/conditional_coverage.xlsx with sheets summary and by_asset.

Usage:
    python run_conditional_coverage.py
    python run_conditional_coverage.py --root ../results/volare/density_avg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR, VOLARE_STOCK_TICKERS
from data_loader import load_data
from evaluation.conditional_coverage import (
    christoffersen_test,
    coverage_by_regime,
    dq_test,
    volatility_regimes,
)
from evaluation.density_io import DECILE_ONLY_MODELS, discover_density_files, read_density_csv
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
CQR_DIR = VOLARE_RESULTS_DIR / "density_cqr"
LEVELS = (0.50, 0.80, 0.95)
STATE_WINDOW = 22


def raw_interval(df: pd.DataFrame, level: float) -> pd.DataFrame:
    lo, hi = round((1 - level) / 2 * 1000), round((1 + level) / 2 * 1000)
    return df[[f"q_{lo:04d}", f"q_{hi:04d}"]].set_axis(["lower", "upper"], axis=1)


def test_interval(actual: pd.Series, bounds: pd.DataFrame, state: pd.Series,
                  level: float, horizon: int) -> Dict[str, float]:
    covered = ((actual >= bounds["lower"]) & (actual <= bounds["upper"])).to_numpy()
    s = state.loc[actual.index]
    out = {"n_obs": int(len(covered)), "coverage": float(covered.mean())}
    out.update(coverage_by_regime(covered, volatility_regimes(s)))
    out.update(christoffersen_test(~covered, level, subsample=horizon))
    out.update(dq_test(~covered, s.to_numpy(), level, subsample=horizon))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--root", default=str(DENSITY_DIR))
    p.add_argument("--cqr-root", default=str(CQR_DIR))
    p.add_argument("--output", default=None, help="Defaults to <root>/conditional_coverage.xlsx")
    args = p.parse_args()

    root, cqr_root = Path(args.root), Path(args.cqr_root)
    output = Path(args.output) if args.output else root / "conditional_coverage.xlsx"
    logger = setup_logger("conditional_coverage")

    files = [f for f in discover_density_files(root)
             if not f.model.startswith("_") and f.context_length is None]
    tickers = sorted({f.ticker for f in files})
    rv = load_data(dataset="volare", tickers=tickers).rv
    states = {tk: rv[tk].dropna().rolling(STATE_WINDOW).mean().shift(1) for tk in tickers}
    logger.info(f"{len(files)} density files, {len(tickers)} tickers; CQR from {cqr_root}")

    rows: List[Dict[str, object]] = []
    for f in files:
        df = read_density_csv(f.path)
        actual, state = df["actual"], states[f.ticker]
        cqr_path = cqr_root / f.model / f.path.name
        cqr = read_density_csv(cqr_path) if cqr_path.exists() else None
        for level in LEVELS:
            if level == 0.95 and f.model in DECILE_ONLY_MODELS:
                continue
            tag = int(round(level * 100))
            sets = {"raw": (actual, raw_interval(df, level))}
            if cqr is not None:
                win = cqr.index.intersection(df.index)
                sets["raw_cqr_window"] = (actual.loc[win], raw_interval(df, level).loc[win])
                sets["cqr"] = (cqr.loc[win, "actual"],
                               cqr.loc[win, [f"lower_{tag}", f"upper_{tag}"]].set_axis(["lower", "upper"], axis=1))
            for method, (a, b) in sets.items():
                rows.append({"model": f.model, "ticker": f.ticker, "horizon": f.horizon,
                             "level": tag, "method": method,
                             **test_interval(a, b, state, level, f.horizon)})
    by_asset = pd.DataFrame(rows)

    keys = ["model", "horizon", "level", "method"]
    g = by_asset.groupby(keys)
    summary = g[["coverage", "cov_calm", "cov_normal", "cov_turbulent", "beta_state"]].mean()
    summary["turbulent_minus_calm"] = summary["cov_turbulent"] - summary["cov_calm"]
    for test in ("uc", "ind", "cc", "dq"):
        summary[f"reject_{test}_pct"] = g[f"p_{test}"].apply(lambda p: 100 * (p < 0.05).mean())
    summary["beta_state_pos_sig_pct"] = g["t_state"].apply(lambda t: 100 * (t > 1.96).mean())
    summary["n_assets"] = g.size()
    summary = summary.reset_index()

    with pd.ExcelWriter(output, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        by_asset.sort_values(keys + ["ticker"]).to_excel(w, sheet_name="by_asset", index=False)
    logger.info(f"Wrote {output}")

    head = summary[summary["level"] == 80]
    for _, r in head.iterrows():
        logger.info(f"  {r['model']:>20s} h={int(r['horizon']):>2d} {r['method']:<15s} "
                    f"cov80={r['coverage']:.2f} calm/normal/turb={r['cov_calm']:.2f}/"
                    f"{r['cov_normal']:.2f}/{r['cov_turbulent']:.2f} "
                    f"reject uc/ind/cc/dq={r['reject_uc_pct']:.0f}/{r['reject_ind_pct']:.0f}/"
                    f"{r['reject_cc_pct']:.0f}/{r['reject_dq_pct']:.0f}")


if __name__ == "__main__":
    main()
