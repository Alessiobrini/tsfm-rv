"""Foundation models at an equal context, the input of the equal-context table.

TTM uses 512 days of context and the other foundation models 1,000. Two comparisons, with every
forecast clipped to the window bounds, QLIKE on the variance scale, and the per-asset ratio to a
Log-HAR averaged across the 50 assets:

  Panel 1, on the dates of the main comparison. Each foundation model at its main context and at
  512 days, against Log-HAR estimated on 1,000 days and on 512 days.
  Panel 2, on the dates where the 1,536-day TTM has forecasts. TTM r2 at 512, 1,024 and 1,536 days
  (release branches, the 512-day branch with a 192-day head and the others with a 96-day head), and
  the r2.1 model of the main comparison, against Log-HAR on 1,000 days.

Inputs: the main forecasts and their `_ctx512` files (run_foundation_volare.py --context-length
512), Log-HAR on a 512-day window and TTM r2 by context (cluster/run_context_length.slurm).
Output: equal_context_panel1.csv and equal_context_panel2.csv, with their per-asset files.

Usage:  python code/compute_equal_context.py [--forecasts DIR] [--loghar512 DIR] [--ttm-r2 DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import (ALL, FORECASTS, METRICS, MODELS, ROOT, TSFM,  # noqa: E402
                                      common_index, load_clipped, load_sigma_all)

TTM_R2_CONTEXTS = [512, 1024, 1536]


def qlike_mean(rv, f):
    r = rv / f ** 2
    return float(np.mean(r - np.log(r) - 1.0))


def run(fc, lh512, ttm_r2, sigma):
    rows1, rows2 = [], []
    for h in [1, 5, 22]:
        for t in ALL:
            sig = sigma[t]
            main_set = load_clipped(fc, MODELS, t, h, sig, round_trip=False)
            extra = {f"{m}@512": load_clipped(fc, [m], t, h, sig, round_trip=False, suffix="_ctx512")[m]
                     for m in TSFM}
            extra["Log_HAR@512"] = load_clipped(lh512, ["Log_HAR"], t, h, sig, round_trip=False)["Log_HAR"]
            for c in TTM_R2_CONTEXTS:
                extra[f"ttm_r2@{c}"] = load_clipped(ttm_r2, ["ttm_r2"], t, h, sig, round_trip=False,
                                                    suffix=f"_ctx{c}")["ttm_r2"]

            # Panel 1: dates of the main comparison
            p1 = {**main_set, **{k: v for k, v in extra.items() if not k.startswith("ttm_r2")}}
            idx = common_index(p1)
            act = main_set["Log_HAR"].loc[idx, "actual"].values
            for k, d in p1.items():
                if not np.allclose(d.loc[idx, "actual"].values, act, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h={h} {k}: actuals differ from Log-HAR")
            rv = act ** 2
            base = {"Log_HAR@1000": qlike_mean(rv, main_set["Log_HAR"].loc[idx, "forecast"].values),
                    "Log_HAR@512": qlike_mean(rv, extra["Log_HAR@512"].loc[idx, "forecast"].values)}
            cands = {f"{m}@main": main_set[m] for m in TSFM}
            cands.update({f"{m}@512": extra[f"{m}@512"] for m in TSFM})
            cands["Log_HAR@512"] = extra["Log_HAR@512"]
            for k, d in cands.items():
                q = qlike_mean(rv, d.loc[idx, "forecast"].values)
                for b, qb in base.items():
                    rows1.append(dict(h=h, ticker=t, model=k, base=b, ratio=q / qb, n=len(idx)))

            # Panel 2: dates where the 1,536-day TTM has forecasts
            p2 = {"Log_HAR": main_set["Log_HAR"], "ttm_r2.1@512": main_set["ttm"],
                  **{k: v for k, v in extra.items() if k.startswith("ttm_r2")}}
            idx2 = common_index(p2)
            act2 = p2["Log_HAR"].loc[idx2, "actual"].values
            for k, d in p2.items():
                if not np.allclose(d.loc[idx2, "actual"].values, act2, rtol=1e-10, atol=0):
                    raise ValueError(f"{t} h={h} {k}: actuals differ from Log-HAR")
            rv2 = act2 ** 2
            qb2 = qlike_mean(rv2, p2["Log_HAR"].loc[idx2, "forecast"].values)
            for k, d in p2.items():
                if k != "Log_HAR":
                    rows2.append(dict(h=h, ticker=t, model=k, n=len(idx2), first=idx2[0].date(),
                                      ratio=qlike_mean(rv2, d.loc[idx2, "forecast"].values) / qb2))
        print(f"h={h} done", flush=True)
    return pd.DataFrame(rows1), pd.DataFrame(rows2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS,
                    help="point forecasts of the 17 models, with the _ctx512 files of the foundation models")
    ap.add_argument("--loghar512", type=Path, default=ROOT / "results/ctx_loghar512/forecasts",
                    help="Log-HAR estimated on a 512-day window")
    ap.add_argument("--ttm-r2", type=Path, default=ROOT / "results/ctx_ttm_r2/forecasts",
                    help="TTM r2 at 512, 1,024 and 1,536 days (files ttm_r2_{ticker}_h{h}_ctx{c}.csv)")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    r1, r2 = run(a.forecasts, a.loghar512, a.ttm_r2, load_sigma_all())
    r1.to_csv(a.out / "equal_context_panel1_long.csv", index=False)
    r2.to_csv(a.out / "equal_context_panel2_long.csv", index=False)
    r1.pivot_table(index="model", columns=["base", "h"], values="ratio", aggfunc="mean").to_csv(
        a.out / "equal_context_panel1.csv")
    r2.pivot_table(index="model", columns="h", values="ratio", aggfunc="mean").to_csv(
        a.out / "equal_context_panel2.csv")
    print("written to", a.out)


if __name__ == "__main__":
    main()
