"""
run_density_significance.py — DM tests and Model Confidence Sets on density losses.

Per-date log-space CRPS and 80% interval score are computed from every
density CSV, aligned on common dates within each asset, then tested two ways:

    1. Per-asset MCS (Hansen, Lunde & Nason 2011), reported as the share of
       assets where each model is in the 90% MCS -- same convention as the
       paper's point-forecast evaluation.
    2. Panel tests on the cross-asset average loss per date: a 90% MCS, and
       Diebold-Mariano of HAR-Empirical against every other model with
       Newey-West HAC (h-1 lags) and Holm correction across comparisons.

Output: results/volare/density/density_significance.xlsx

Usage:
    python run_density_significance.py
    python run_density_significance.py --horizons 1 --n-bootstrap 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))

from config import VOLARE_RESULTS_DIR, eval_cfg, forecast_cfg
from evaluation.density import (
    DEFAULT_QUANTILE_LEVELS,
    crps_from_quantiles,
    interval_score_per_obs,
)
from evaluation.density_io import discover_density_files, read_density_csv, split_actual_and_grid
from evaluation.dm_test import dm_test
from evaluation.mcs import model_confidence_set
from utils import setup_logger

DENSITY_DIR = VOLARE_RESULTS_DIR / "density"
BENCHMARK = "har_logn_empirical"
LOSSES = ("crps", "is80")


def per_date_losses(path: Path, levels: np.ndarray) -> pd.DataFrame:
    df = read_density_csv(path)
    actuals, q_grid = split_actual_and_grid(df, levels)
    log_a = np.log(np.clip(actuals, 1e-30, None))
    log_q = np.log(np.clip(q_grid, 1e-12, None))
    return pd.DataFrame(
        {
            "crps": crps_from_quantiles(log_a, log_q, levels),
            "is80": interval_score_per_obs(log_a, np.maximum.accumulate(log_q, axis=1), 0.80, levels),
        },
        index=df.index,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--horizons", nargs="+", type=int, default=forecast_cfg.horizons)
    p.add_argument("--alpha", type=float, default=eval_cfg.mcs_alpha)
    p.add_argument("--n-bootstrap", type=int, default=eval_cfg.mcs_n_bootstrap)
    p.add_argument("--block-length", type=int, default=eval_cfg.mcs_block_length)
    p.add_argument("--output", default=str(DENSITY_DIR / "density_significance.xlsx"))
    args = p.parse_args()

    logger = setup_logger("density_significance")
    levels = DEFAULT_QUANTILE_LEVELS
    files = [f for f in discover_density_files(DENSITY_DIR, horizons=args.horizons)
             if not f.model.startswith("_") and f.context_length is None]

    by_asset_rows, panel_rows, dm_rows = [], [], []
    for h in args.horizons:
        # losses[loss][ticker] -> DataFrame(date x model)
        losses: Dict[str, Dict[str, pd.DataFrame]] = {l: {} for l in LOSSES}
        for f in (f for f in files if f.horizon == h):
            ld = per_date_losses(f.path, levels)
            for l in LOSSES:
                losses[l].setdefault(f.ticker, {})[f.model] = ld[l]
        for l in LOSSES:
            per_asset = {tk: pd.DataFrame(d).dropna() for tk, d in losses[l].items()}
            models = sorted(set.intersection(*(set(d.columns) for d in per_asset.values())))
            logger.info(f"h={h} {l}: {len(per_asset)} assets, {len(models)} models")

            for tk, d in per_asset.items():
                res = model_confidence_set({m: d[m].to_numpy() for m in models}, alpha=args.alpha,
                                           n_bootstrap=args.n_bootstrap, block_length=args.block_length)
                for m in models:
                    by_asset_rows.append({"horizon": h, "loss": l, "ticker": tk, "model": m,
                                          "in_mcs": m in res.surviving_models})

            panel = pd.concat([d[models] for d in per_asset.values()]).groupby(level=0).mean().sort_index()
            res = model_confidence_set({m: panel[m].to_numpy() for m in models}, alpha=args.alpha,
                                       n_bootstrap=args.n_bootstrap, block_length=args.block_length)
            for m in models:
                panel_rows.append({"horizon": h, "loss": l, "model": m, "mean_loss": panel[m].mean(),
                                   "in_mcs": m in res.surviving_models,
                                   "mcs_pvalue": res.p_values.get(m, np.nan), "n_dates": len(panel)})

            rivals = [m for m in models if m != BENCHMARK]
            tests = [dm_test(panel[BENCHMARK].to_numpy(), panel[m].to_numpy(), h=h) for m in rivals]
            p_holm = multipletests([t.p_value for t in tests], method="holm")[1]
            for m, t, ph in zip(rivals, tests, p_holm):
                dm_rows.append({"horizon": h, "loss": l, "benchmark": BENCHMARK, "rival": m,
                                "mean_diff_bench_minus_rival": t.mean_loss_diff, "dm_stat": t.statistic,
                                "p_value": t.p_value, "p_holm": ph})

    by_asset = pd.DataFrame(by_asset_rows)
    rate = (by_asset.groupby(["horizon", "loss", "model"])["in_mcs"].mean().mul(100).round(1)
            .rename("pct_assets_in_mcs").reset_index())
    panel = pd.DataFrame(panel_rows)
    dm = pd.DataFrame(dm_rows)

    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        panel.to_excel(w, sheet_name="panel_mcs", index=False)
        dm.to_excel(w, sheet_name="dm_vs_har_empirical", index=False)
        rate.to_excel(w, sheet_name="mcs_rate_by_asset", index=False)
        by_asset.to_excel(w, sheet_name="mcs_by_asset", index=False)
    logger.info(f"Wrote {args.output}")

    for (h, l), g in panel.groupby(["horizon", "loss"]):
        logger.info(f"h={h} {l}: panel {int((1 - args.alpha) * 100)}% MCS = "
                    f"{sorted(g.loc[g.in_mcs, 'model'])}")
    for (h, l), g in dm.groupby(["horizon", "loss"]):
        worse = g.loc[(g.p_holm < 0.05) & (g.mean_diff_bench_minus_rival < 0), "rival"].tolist()
        logger.info(f"h={h} {l}: HAR-Empirical significantly better (Holm 5%) than {worse}")


if __name__ == "__main__":
    main()
