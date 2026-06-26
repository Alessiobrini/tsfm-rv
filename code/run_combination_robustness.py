"""
run_combination_robustness.py — TTM + Log-HAR forecast-combination robustness.

Motivated by the referee/self-attack note that the paper reports a skeptical
"only TTM beats Log-HAR" result without testing whether a simple combination of
the best foundation model (TTM) and the best econometric benchmark (Log-HAR)
does better than either alone. The Mincer--Zarnowitz bias-corrected evidence
(§6) shows TTM is well calibrated while the HAR family is affine-efficient,
suggesting the two carry complementary information.

Two combinations, both formed on the volatility scale:
  * comb_ew  — equal weight, f = 0.5*f_TTM + 0.5*f_LogHAR.
  * comb_bg  — recursive Bates--Granger optimal weight, estimated from squared
    forecast errors observed strictly up to t-1 (expanding window, no look-ahead),
    clipped to [0, 1]; an EW warm-up of WARMUP observations.

The combinations are injected into the existing headline forecast set per
(ticker, horizon) and evaluated through the SAME pipeline functions
(align_forecasts + compute_metrics_for_group) used to build the published
metrics, so QLIKE, DM, and MCS are computed identically. Because each
combination's date index is a superset of the all-model common sample, adding it
does not change the common sample, so the existing models' per-asset QLIKE (and
hence the published loss ratios) are unchanged; this is asserted as a validation
check (TTM loss ratios must reproduce 0.972 / 0.965 / 0.983).

Writes results/volare/metrics/combination_metrics.csv (per-asset) and
combination_summary.csv. Does NOT overwrite any canonical metrics file.

Usage:
    python code/run_combination_robustness.py            # loss ratios + DM (fast)
    python code/run_combination_robustness.py --mcs      # also recompute MCS (slow)
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from run_evaluation_volare import load_all_forecasts, FORECAST_DIR  # noqa: E402
from run_evaluation import align_forecasts, compute_metrics_for_group  # noqa: E402
from config import (  # noqa: E402
    VOLARE_STOCK_TICKERS, VOLARE_FX_TICKERS, VOLARE_FUTURES_TICKERS,
)

METRICS_DIR = ROOT / "results" / "volare" / "metrics"
HORIZONS = [1, 5, 22]
TICKERS = VOLARE_STOCK_TICKERS + VOLARE_FX_TICKERS + VOLARE_FUTURES_TICKERS
SCALE = "vol"
WARMUP = 100          # observations using equal weight before BG kicks in
TTM, LHAR, ARMA = "ttm", "Log_HAR", "ARMA"
EW, BG = "comb_ew", "comb_bg"            # two-way TTM + Log-HAR
EW3, BG3 = "comb3_ew", "comb3_bg"        # three-way TTM + Log-HAR + ARMA


def bates_granger_recursive(actual, f1, f2, warmup=WARMUP):
    """Recursive Bates--Granger combination weight on the volatility scale.

    At each t, weights are estimated from the error (co)variances of f1, f2 over
    observations strictly before t (expanding window), so the combined forecast
    at t uses no contemporaneous information. Weight on f1 is
        w = (s22 - s12) / (s11 + s22 - 2 s12),
    clipped to [0, 1]; degenerate denominators fall back to 0.5.
    """
    a = np.asarray(actual, float)
    f1 = np.asarray(f1, float)
    f2 = np.asarray(f2, float)
    e1, e2 = a - f1, a - f2
    n = len(a)
    comb = np.empty(n)
    for t in range(n):
        if t < warmup:
            w = 0.5
        else:
            x1, x2 = e1[:t], e2[:t]
            s11 = np.mean(x1 * x1)
            s22 = np.mean(x2 * x2)
            s12 = np.mean(x1 * x2)
            denom = s11 + s22 - 2.0 * s12
            w = 0.5 if abs(denom) < 1e-18 else (s22 - s12) / denom
            w = min(1.0, max(0.0, w))
        comb[t] = w * f1[t] + (1.0 - w) * f2[t]
    return comb


def min_variance_recursive(actual, members, warmup=WARMUP):
    """Recursive minimum-variance (generalized Bates--Granger) combination of K
    forecasts on the volatility scale. At each t the weights minimize the error
    variance using the K x K error covariance over observations strictly before t
    (expanding window): w = Sigma^{-1} 1 / (1' Sigma^{-1} 1), with negative
    weights clipped to 0 and renormalized; equal weight during the warm-up or
    when the covariance is singular. No look-ahead."""
    a = np.asarray(actual, float)
    F = np.column_stack([np.asarray(f, float) for f in members])   # n x K
    n, K = F.shape
    E = a[:, None] - F
    ones = np.ones(K)
    comb = np.empty(n)
    for t in range(n):
        if t < warmup:
            w = ones / K
        else:
            try:
                Sig = np.atleast_2d(np.cov(E[:t].T, bias=True))
                raw = np.linalg.solve(Sig, ones)
                w = raw / raw.sum()
                if not np.all(np.isfinite(w)):
                    raise np.linalg.LinAlgError
                w = np.clip(w, 0.0, None)
                s = w.sum()
                w = w / s if s > 0 else ones / K
            except np.linalg.LinAlgError:
                w = ones / K
        comb[t] = float(F[t] @ w)
    return comb


def build_combinations(model_dfs):
    """Build the combination forecasts: two-way TTM + Log-HAR (EW, BG) and
    three-way TTM + Log-HAR + ARMA (EW, BG), each aligned on its members' common
    dates. Returns a dict, or None if no combination could be formed."""
    out = {}
    if TTM in model_dfs and LHAR in model_dfs:
        d1, d2 = model_dfs[TTM], model_dfs[LHAR]
        common = d1.index.intersection(d2.index).sort_values()
        if len(common) > 0:
            a = d1.loc[common, "actual"].values
            f1 = d1.loc[common, "forecast"].values
            f2 = d2.loc[common, "forecast"].values
            out[EW] = pd.DataFrame({"actual": a, "forecast": 0.5 * f1 + 0.5 * f2}, index=common)
            out[BG] = pd.DataFrame({"actual": a, "forecast": bates_granger_recursive(a, f1, f2)}, index=common)
    if TTM in model_dfs and LHAR in model_dfs and ARMA in model_dfs:
        d1, d2, d3 = model_dfs[TTM], model_dfs[LHAR], model_dfs[ARMA]
        common = d1.index.intersection(d2.index).intersection(d3.index).sort_values()
        if len(common) > 0:
            a = d1.loc[common, "actual"].values
            f1 = d1.loc[common, "forecast"].values
            f2 = d2.loc[common, "forecast"].values
            f3 = d3.loc[common, "forecast"].values
            out[EW3] = pd.DataFrame({"actual": a, "forecast": (f1 + f2 + f3) / 3.0}, index=common)
            out[BG3] = pd.DataFrame({"actual": a, "forecast": min_variance_recursive(a, [f1, f2, f3])}, index=common)
    return out if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcs", action="store_true", help="also recompute MCS (slow)")
    args = ap.parse_args()

    groups = load_all_forecasts()
    rows = []          # per (model, ticker, h): QLIKE, in_mcs, dm vs ttm / loghar
    for h in HORIZONS:
        for tic in TICKERS:
            key = (tic, h)
            if key not in groups:
                continue
            model_dfs = dict(groups[key])     # copy
            combos = build_combinations(model_dfs)
            if combos is None:
                print(f"  skip {tic} h{h}: missing member forecast")
                continue
            model_dfs.update(combos)

            actual, forecasts = align_forecasts(model_dfs)
            if actual is None or len(actual) == 0:
                continue
            metrics_df, dm_pvals, mcs_res = compute_metrics_for_group(
                actual, forecasts, h, scale=SCALE,
            )
            if not args.mcs:
                mcs_res = None  # compute_metrics_for_group still ran it; ignore

            for m in metrics_df.index:
                rec = {
                    "model": m, "ticker": tic, "horizon": h,
                    "QLIKE": float(metrics_df.loc[m, "QLIKE"]),
                    "n_obs": len(actual),
                }
                if mcs_res is not None:
                    rec["in_mcs"] = int(m in mcs_res.surviving_models)
                if dm_pvals is not None and m in (EW, BG, EW3, BG3):
                    rec["dm_p_vs_ttm"] = float(dm_pvals.loc[m, TTM]) if TTM in dm_pvals.columns else np.nan
                    rec["dm_p_vs_loghar"] = float(dm_pvals.loc[m, LHAR]) if LHAR in dm_pvals.columns else np.nan
                rows.append(rec)
        print(f"h={h} done ({len([r for r in rows if r['horizon']==h])} model-asset rows)")

    per_asset = pd.DataFrame(rows)
    per_asset.to_csv(METRICS_DIR / "combination_metrics.csv", index=False)

    # ---- loss ratios vs Log-HAR (mean per-asset QLIKE ratio across assets) ----
    print("\n" + "=" * 64)
    print("AVERAGE QLIKE LOSS RATIOS vs Log-HAR (mean across 50 assets)")
    print("=" * 64)
    summ = []
    report_models = [TTM, LHAR, "sundial", EW, BG, EW3, BG3]
    for h in HORIZONS:
        sub = per_asset[per_asset["horizon"] == h]
        piv = sub.pivot_table(index="ticker", columns="model", values="QLIKE")
        ratios = piv.div(piv[LHAR], axis=0).mean(axis=0)
        line = {f"h{h}": {m: float(ratios.get(m, np.nan)) for m in report_models}}
        print(f"\nh={h}:")
        for m in report_models:
            print(f"  {m:10s} {ratios.get(m, np.nan):.4f}")
        # win rates for combos
        for m in (EW, BG, EW3, BG3):
            if m not in piv.columns:
                continue
            wr_lh = float((piv[m] < piv[LHAR]).mean())
            wr_ttm = float((piv[m] < piv[TTM]).mean())
            summ.append({"horizon": h, "model": m,
                         "loss_ratio": float(ratios.get(m, np.nan)),
                         "beats_loghar_frac": wr_lh, "beats_ttm_frac": wr_ttm})
        summ.append({"horizon": h, "model": TTM, "loss_ratio": float(ratios.get(TTM, np.nan)),
                     "beats_loghar_frac": float((piv[TTM] < piv[LHAR]).mean()), "beats_ttm_frac": np.nan})

    summ_df = pd.DataFrame(summ)

    # ---- DM significance: combos vs ttm / loghar ----
    print("\n" + "=" * 64)
    print("DM TEST (5%): combo significantly beats member, # of 50 assets")
    print("=" * 64)
    for h in HORIZONS:
        for m in (EW, BG, EW3, BG3):
            sub = per_asset[(per_asset.horizon == h) & (per_asset.model == m)]
            if sub.empty:
                continue
            # combo beats X if dm p<0.05 AND combo QLIKE < X QLIKE
            piv = per_asset[per_asset.horizon == h].pivot_table(index="ticker", columns="model", values="QLIKE")
            beat_ttm = ((sub.set_index("ticker")["dm_p_vs_ttm"] < 0.05) &
                        (piv[m] < piv[TTM])).sum()
            beat_lh = ((sub.set_index("ticker")["dm_p_vs_loghar"] < 0.05) &
                       (piv[m] < piv[LHAR])).sum()
            print(f"  h={h:2d} {m}: beats TTM on {int(beat_ttm)}/50, beats Log-HAR on {int(beat_lh)}/50")

    # ---- MCS inclusion ----
    if args.mcs and "in_mcs" in per_asset.columns:
        print("\n" + "=" * 64)
        print("MCS INCLUSION RATE (augmented candidate set, % of 50 assets)")
        print("=" * 64)
        for h in HORIZONS:
            sub = per_asset[per_asset.horizon == h]
            for m in report_models:
                rate = sub[sub.model == m]["in_mcs"].mean() * 100
                print(f"  h={h:2d} {m:10s} {rate:.1f}%")

    summ_df.to_csv(METRICS_DIR / "combination_summary.csv", index=False)

    # ---- validation ----
    print("\n" + "=" * 64)
    print("VALIDATION (adding combos must not change TTM loss ratios: 0.982/0.986/0.987)")
    print("=" * 64)
    ok = True
    expect = {1: 0.982, 5: 0.986, 22: 0.987}
    for h in HORIZONS:
        sub = per_asset[per_asset.horizon == h]
        piv = sub.pivot_table(index="ticker", columns="model", values="QLIKE")
        ttm_lr = float(piv[TTM].div(piv[LHAR]).mean())
        diff = abs(ttm_lr - expect[h])
        flag = "OK" if diff < 0.002 else "**MISMATCH**"
        if diff >= 0.002:
            ok = False
        print(f"  h={h:2d}: TTM loss ratio = {ttm_lr:.4f} (published {expect[h]}) [{flag}]")
    print("\nVALIDATION", "PASSED" if ok else "FAILED")
    print("Wrote", METRICS_DIR / "combination_metrics.csv")
    print("Wrote", METRICS_DIR / "combination_summary.csv")


if __name__ == "__main__":
    main()
