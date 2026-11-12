"""The supervised ML baselines scored against the 17 models of the main comparison, the input of the
two ML tables.

Learners (run_ml_volare.py, cluster/run_ml_baselines.slurm): XGBoost and a feedforward network on
the HAR inputs, and an LSTM on 22 or 252 days of log volatility, with Log-HAR refitted every 22
origins (run_log_har_refit22.py) for the same cadence. The ML forecasts are stored before the clip,
so the clip to the window bounds is applied here as for every other model. Every model is scored on
the dates common to every model, with QLIKE on the variance scale, on the point target at
h = 1, 5, 22 and on the averaged target at h = 5, 22. On the averaged target every squared forecast
is scored against the mean daily RV over the h days.

For each model, target, horizon and asset, on the volatility forecast f that is scored:
  ratio      QLIKE ratio to Log-HAR
  bias       mean log(sigma / f), the level error (negative when the forecasts are too high)
  sd_err     standard deviation of log(sigma / f), as a ratio to Log-HAR's
  rc_ratio   QLIKE ratio after the real-time level correction applied to every model, Log-HAR
             included (evaluation.forecast_sets.rescale_level)
The panel test is the equal-weight Diebold-Mariano test of the QLIKE difference to Log-HAR and to
TTM, with Newey-West standard errors and max(22, 2h) lags.

Output: ml_baselines_long.csv, ml_baselines_tests.csv, ml_baselines_configs.csv,
ml_baselines_summary.csv and ml_baselines_by_class.csv.

Usage:  python code/compute_ml_baselines.py [--forecasts DIR] [--forecasts-avg DIR] [--ml DIR]
                                            [--lh22 DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VOLARE_FUTURES_TICKERS as FUT, VOLARE_FX_TICKERS as FX  # noqa: E402
from evaluation.forecast_sets import (ALL, DIRECT, FORECASTS, FORECASTS_AVG, L, METRICS, MODELS,  # noqa: E402
                                      ROOT, T, asset_class, common_index, equal_weight_series, hac_t,
                                      load_sigma_all, qlike_series as ql, read, rescale_level)
from evaluation.targets import holding_rv  # noqa: E402
from forecasting.bounds import clip_to_window  # noqa: E402

ML = ["xgb_har", "ffn_har", "lstm_22", "lstm_22_p100", "lstm_252"]
LH22 = "Log_HAR_refit22"
KINDS = {"point": (1, 5, 22), "avg": (5, 22)}


def load(t, h, kind, sig, dirs, ml_dir, lh22_dir, ml):
    """Every model's file for one asset, horizon and target, forecasts clipped to the window bounds."""
    out = {m: read(dirs[kind] / f"{m}_{t}_h{h}.csv") for m in MODELS}
    for m in ml:
        out[m] = read(Path(ml_dir.format(kind=kind)) / f"{m}_{t}_h{h}.csv")
    if kind == "point":
        out[LH22] = read(Path(lh22_dir) / f"Log_HAR_{t}_h{h}.csv")
    for d in out.values():
        d["forecast"] = clip_to_window(d["forecast"], sig)
    return out


def check_ml_files(files, ml):
    """The ML files must have Log-HAR's dates and actuals exactly (to rounding), with no gaps."""
    ref = files[L]
    for m in ml + ([LH22] if LH22 in files else []):
        d = files[m]
        if not d.index.equals(ref.index):
            raise AssertionError(f"{m}: dates differ from Log-HAR ({len(d)} against {len(ref)})")
        if not np.allclose(d["actual"].values, ref["actual"].values, rtol=1e-10, atol=0):
            raise AssertionError(f"{m}: actuals differ from Log-HAR")
        if d["forecast"].isna().any():
            raise AssertionError(f"{m}: missing forecasts")


def on_variance_target(d, kind, h, sig):
    """The file with its actual replaced by the square root of the target variance. On the averaged
    target that is the mean daily RV over the h days starting at the origin, computed from raw RV
    for every model, since the stored actual of HAR-J, HAR-RS and HARQ is the square root of that
    mean while the other files store the mean of volatility. On the point target the stored actual
    already is the volatility of the target day."""
    if kind == "point":
        return d
    out = d.copy()
    out["actual"] = np.sqrt(holding_rv(sig ** 2, d.index, h))
    return out


def run(sigma_all, dirs, ml_dir, lh22_dir, ml):
    rows, diffs = [], {}
    for t in ALL:
        sig = sigma_all[t]
        for kind, horizons in KINDS.items():
            for h in horizons:
                files = load(t, h, kind, sig, dirs, ml_dir, lh22_dir, ml)
                check_ml_files(files, ml)
                models = list(files)
                idx = common_index(files, "forecast")
                idx17 = common_index({m: files[m] for m in MODELS}, "forecast")
                y = files[L].loc[idx, "actual"].values
                for m in models:
                    if (kind == "point" or m not in DIRECT) and not np.allclose(
                            files[m].loc[idx, "actual"].values, y, rtol=1e-10, atol=0):
                        raise ValueError(f"{t} {kind} h={h} {m}: actuals differ from Log-HAR")
                files = {m: on_variance_target(d, kind, h, sig) for m, d in files.items()}
                y = files[L].loc[idx, "actual"].values
                rv = y ** 2
                fl = files[L].loc[idx, "forecast"].values
                base_q = ql(rv, fl ** 2).mean()
                base_sd = np.log(y / fl).std()
                rc_base = ql(rv, rescale_level(files[L], h).loc[idx].values).mean()
                loss = {}
                for m in models:
                    f = files[m].loc[idx, "forecast"].values
                    e = np.log(y / f)
                    loss[m] = ql(rv, f ** 2)
                    rows.append(dict(
                        ticker=t, cls=asset_class(t), kind=kind, h=h, model=m, n=len(idx),
                        n_lost=len(idx17) - len(idx), ratio=loss[m].mean() / base_q, bias=e.mean(),
                        sd_err=e.std() / base_sd,
                        rc_ratio=ql(rv, rescale_level(files[m], h).loc[idx].values).mean() / rc_base,
                        mean_v=float(files[m].loc[idx, "v"].mean()) if "v" in files[m] else np.nan))
                for m in models:
                    for ref in (L, T):
                        if m != ref:
                            diffs.setdefault((kind, h, m, ref), {})[t] = pd.Series(loss[m] - loss[ref], index=idx)
        print("done", t, flush=True)
    tests = []
    groups = [("all", list(ALL)), ("stocks", [x for x in ALL if asset_class(x) == "stocks"]),
              ("fx", list(FX)), ("futures", list(FUT))]
    for (kind, h, m, ref), dd in diffs.items():
        panel = pd.DataFrame(dd)
        for grp, tick in groups:
            s_t = equal_weight_series(panel[tick])
            tests.append(dict(kind=kind, h=h, model=m, ref=ref, group=grp,
                              t=hac_t(s_t.values, max(22, 2 * h)), gap=float(s_t.sum())))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def ml_configs(ml_dir, ml):
    """The share of origins at which each grid setting was chosen, by model and target."""
    rows = []
    for kind, horizons in KINDS.items():
        for m in ml:
            for t in ALL:
                for h in horizons:
                    d = pd.read_csv(Path(ml_dir.format(kind=kind)) / f"{m}_{t}_h{h}.csv")
                    for cfg, n in d["config"].value_counts().items():
                        rows.append(dict(kind=kind, model=m, h=h, config=cfg, n=n,
                                         n_fit_med=float(d["n_fit"].median())))
    df = pd.DataFrame(rows)
    return df.groupby(["kind", "model", "h", "config"]).agg(n=("n", "sum"), n_fit_med=("n_fit_med", "median")).reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--forecasts-avg", type=Path, default=FORECASTS_AVG,
                    help="averaged-target forecasts of the 17 models")
    ap.add_argument("--ml", default=str(ROOT / "results/ml_{kind}/forecasts"),
                    help="folder of the ML forecasts, with {kind} standing for point or avg")
    ap.add_argument("--lh22", type=Path, default=ROOT / "results/log_har_refit22/forecasts",
                    help="Log-HAR refitted every 22 origins (run_log_har_refit22.py)")
    ap.add_argument("--ml-models", nargs="+", default=ML, help="score a subset of the ML models")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = {"point": a.forecasts, "avg": a.forecasts_avg}
    res, tests = run(load_sigma_all(), dirs, a.ml, a.lh22, a.ml_models)
    res.to_csv(a.out / "ml_baselines_long.csv", index=False)
    tests.to_csv(a.out / "ml_baselines_tests.csv", index=False)
    ml_configs(a.ml, a.ml_models).to_csv(a.out / "ml_baselines_configs.csv", index=False)
    s = res.groupby(["kind", "h", "model"]).agg(
        ratio=("ratio", "mean"), beat=("ratio", lambda r: int((r < 1).sum())),
        rc_ratio=("rc_ratio", "mean"), rc_beat=("rc_ratio", lambda r: int((r < 1).sum())),
        bias=("bias", "mean"), sd_err=("sd_err", "mean"), mean_v=("mean_v", "mean"),
        n=("n", "mean"), n_lost=("n_lost", "max")).reset_index()
    s.to_csv(a.out / "ml_baselines_summary.csv", index=False)
    res.groupby(["kind", "h", "cls", "model"]).ratio.mean().reset_index().to_csv(
        a.out / "ml_baselines_by_class.csv", index=False)
    print("written to", a.out)


if __name__ == "__main__":
    main()
