"""Contamination check by sample split, the inputs of the two contamination tables.

If pretraining on market data that overlaps the evaluation period helped a foundation model, its
loss relative to Log-HAR would be lower in the part of the sample its pretraining could have seen
than in the part it could not. Every split is by the date of the forecast target, the trading day
h - 1 rows after the origin. Two splits:

  A_2024     1 January 2024, the same date for every model.
  B_weights  the day after the date each foundation model's pinned weight file was last changed on
             Hugging Face. A target after that date cannot be in the model's pretraining data.

The script writes three groups of files to the output folder.

  contamination_long.csv, contamination_summary.csv
      Per model, horizon and side of the split, the mean across assets of the per-asset QLIKE ratio
      to Log-HAR. Per asset, the daily loss differential to Log-HAR is regressed on a constant and a
      dummy for the later side, with Newey-West standard errors (22 lags).
  contamination_control_long.csv, contamination_control.csv
      HAR, ARFIMA, ARMA and MEM are estimated only on the 1,000 days before each origin, so a change
      in their ratio across a split measures what the period alone does. For each foundation model
      and split, the excess is the change of log(ratio) for the model minus the mean change for the
      four controls. The panel test averages the daily differential of the model to the mean of the
      four controls across the assets observed on each target date, and regresses the average on a
      constant and the later-side dummy with Newey-West standard errors (44 lags). Since the 50
      assets share the same market days, this test accounts for shocks common to all assets.
  contamination_years_long.csv, contamination_years.csv
      The per-asset QLIKE ratio to Log-HAR by calendar year of the target, 2019 to January 2026,
      averaged across assets.

Every forecast is clipped to the window bounds, and every model is scored on the dates common to
the 17 models.

Usage:  python code/compute_contamination.py [--forecasts DIR] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.forecast_sets import (ALL, FORECASTS, METRICS, MODELS, TSFM, common_index,  # noqa: E402
                                      load_clipped, load_sigma_all, qlike_series)
from evaluation.targets import target_dates  # noqa: E402

SPLIT_A = pd.Timestamp("2024-01-01")
# Date the pinned weight file was last changed on Hugging Face (tree API, lastCommit of the weight
# file at the pinned commit), queried on 2 October 2026. TTM: the 512-96-ft-r2.1 branch.
WEIGHTS_DATE = {
    "lag_llama": "2024-02-07", "moirai_moe_small": "2024-11-01",
    "chronos_bolt_small": "2024-11-13", "chronos_bolt_base": "2024-11-13",
    "ttm": "2025-01-31", "toto": "2025-05-06", "sundial": "2025-05-14",
    "moirai_2_0_small": "2025-08-06", "timesfm_2_5": "2025-10-01",
}
CONTROLS = ["HAR", "ARFIMA", "ARMA", "MEM"]
NW_LAGS, NW_PANEL = 22, 44
MIN_SIDE = 22
YEARS = range(2019, 2027)
MIN_YEAR_DAYS = 15


def weights_cut(m):
    return pd.Timestamp(WEIGHTS_DATE[m]) + pd.Timedelta(days=1)


def split_test(d, later):
    """Coefficient and Newey-West t-statistic of the later-side dummy in an OLS of ``d``."""
    X = sm.add_constant(later.astype(float))
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return float(res.params[1]), float(res.tvalues[1])


def panel_t(series_by_asset, cut):
    """Newey-West t-statistic of the later-side dummy for the cross-asset daily mean."""
    m = pd.concat(series_by_asset, axis=1).mean(axis=1).sort_index()
    later = (m.index >= cut).astype(float)
    X = sm.add_constant(np.asarray(later))
    res = sm.OLS(m.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_PANEL})
    return float(res.tvalues[1])


def daily_losses(t, h, folder, sig):
    """Target dates and daily QLIKE of every model for one asset and horizon, on common dates."""
    dfs = load_clipped(folder, MODELS, t, h, sig, round_trip=False)
    idx = common_index(dfs)
    act = dfs["Log_HAR"].loc[idx, "actual"].values
    for m in MODELS:
        if not np.allclose(dfs[m].loc[idx, "actual"].values, act, rtol=1e-10, atol=0):
            raise ValueError(f"{t} h={h} {m}: actuals differ from Log-HAR")
    tdate = target_dates(sig.index, idx, h)
    if not np.allclose(sig.loc[tdate].values, act, rtol=1e-10, atol=0):
        raise ValueError(f"{t} h={h}: the target dates do not match the actuals")
    rv = act ** 2
    return tdate, {m: qlike_series(rv, dfs[m].loc[idx, "forecast"].values ** 2) for m in MODELS}


def split_rows(t, h, tdate, Lq):
    rows = []
    splits = {"A_2024": {m: SPLIT_A for m in MODELS}, "B_weights": {m: weights_cut(m) for m in TSFM}}
    for name, cut in splits.items():
        for m, c in cut.items():
            later = np.asarray(tdate >= c)
            if later.sum() < MIN_SIDE or (~later).sum() < MIN_SIDE:
                continue
            rec = dict(split=name, h=h, ticker=t, model=m, cut=c.date(),
                       n_before=int((~later).sum()), n_after=int(later.sum()))
            for side, mask in [("before", ~later), ("after", later)]:
                rec[f"ratio_{side}"] = Lq[m][mask].mean() / Lq["Log_HAR"][mask].mean()
            if m != "Log_HAR":
                rec["dummy"], rec["t_dummy"] = split_test(Lq[m] - Lq["Log_HAR"], later)
            rows.append(rec)
    return rows


def summarize_splits(res):
    out = []
    for (split, h, m), g in res[res.model != "Log_HAR"].groupby(["split", "h", "model"]):
        dl = np.log(g.ratio_after) - np.log(g.ratio_before)
        p = wilcoxon(dl).pvalue if len(g) > 5 and (dl != 0).any() else np.nan
        out.append(dict(split=split, h=h, model=m, cut=g.cut.iloc[0], n_assets=len(g),
                        days_before=int(g.n_before.median()), days_after=int(g.n_after.median()),
                        ratio_before=g.ratio_before.mean(), ratio_after=g.ratio_after.mean(),
                        n_worse_sig=int((g.t_dummy > 1.96).sum()),
                        n_better_sig=int((g.t_dummy < -1.96).sum()), wilcoxon_p=p))
    return pd.DataFrame(out)


def control_rows(t, h, tdate, Lq, panel):
    """Per-asset change of each foundation model against the four controls, and the daily series
    of the panel test, collected in ``panel``."""
    rows = []

    def dlog(m, later):
        a = Lq[m][later].mean() / Lq["Log_HAR"][later].mean()
        b = Lq[m][~later].mean() / Lq["Log_HAR"][~later].mean()
        return np.log(a) - np.log(b)

    ctrl = np.mean([Lq[c] for c in CONTROLS], axis=0)
    for m in TSFM:
        for split, cut in [("A_2024", SPLIT_A), ("B_weights", weights_cut(m))]:
            later = np.asarray(tdate >= cut)
            if later.sum() < MIN_SIDE or (~later).sum() < MIN_SIDE:
                continue
            d_model = dlog(m, later)
            d_ctrl = float(np.mean([dlog(c, later) for c in CONTROLS]))
            _, tt = split_test(Lq[m] - ctrl, later)
            key = (split, h, m)
            panel.setdefault(key, {"cut": cut, "excess": [], "vs_loghar": []})
            panel[key]["excess"].append(pd.Series(Lq[m] - ctrl, index=tdate))
            panel[key]["vs_loghar"].append(pd.Series(Lq[m] - Lq["Log_HAR"], index=tdate))
            rows.append(dict(split=split, h=h, ticker=t, model=m, cut=cut.date(),
                             n_after=int(later.sum()), d_model=d_model, d_control=d_ctrl,
                             excess=d_model - d_ctrl, t_dummy=tt))
    return rows


def summarize_control(r, panel):
    summ = []
    for (split, h, m), g in r.groupby(["split", "h", "model"]):
        p = panel[(split, h, m)]
        summ.append(dict(split=split, h=h, model=m, cut=g.cut.iloc[0],
                         days_after=int(g.n_after.median()),
                         pct_change_model=100 * np.expm1(g.d_model.mean()),
                         pct_change_control=100 * np.expm1(g.d_control.mean()),
                         excess_pct=100 * np.expm1(g.excess.mean()),
                         wilcoxon_p=wilcoxon(g.excess).pvalue,
                         n_worse_sig=int((g.t_dummy > 1.96).sum()),
                         n_better_sig=int((g.t_dummy < -1.96).sum()),
                         t_panel_excess=panel_t(p["excess"], p["cut"]),
                         t_panel_vs_loghar=panel_t(p["vs_loghar"], p["cut"])))
    s = pd.DataFrame(summ)
    order = {m: i for i, m in enumerate(TSFM)}
    return s.sort_values(["split", "h", "model"], key=lambda c: c.map(order) if c.name == "model" else c)


def year_rows(t, h, tdate, Lq):
    rows = []
    yr = np.asarray(tdate.year)
    for y in YEARS:
        mk = yr == y
        if mk.sum() < MIN_YEAR_DAYS:
            continue
        for m in MODELS:
            rows.append(dict(h=h, ticker=t, model=m, year=y, n=int(mk.sum()),
                             ratio=Lq[m][mk].mean() / Lq["Log_HAR"][mk].mean()))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--forecasts", type=Path, default=FORECASTS, help="point forecasts of the 17 models")
    ap.add_argument("--out", type=Path, default=METRICS)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sigma = load_sigma_all()
    splits, control, years, panel = [], [], [], {}
    for h in (1, 5, 22):
        for t in ALL:
            tdate, Lq = daily_losses(t, h, a.forecasts, sigma[t])
            splits += split_rows(t, h, tdate, Lq)
            control += control_rows(t, h, tdate, Lq, panel)
            years += year_rows(t, h, tdate, Lq)
        print(f"h={h} done", flush=True)

    res = pd.DataFrame(splits)
    res.to_csv(a.out / "contamination_long.csv", index=False)
    summarize_splits(res).to_csv(a.out / "contamination_summary.csv", index=False)

    r = pd.DataFrame(control)
    r.to_csv(a.out / "contamination_control_long.csv", index=False)
    summarize_control(r, panel).to_csv(a.out / "contamination_control.csv", index=False)

    y = pd.DataFrame(years)
    y.to_csv(a.out / "contamination_years_long.csv", index=False)
    tab = y.pivot_table(index=["h", "model"], columns="year", values="ratio", aggfunc="mean")
    tab = tab.reindex(pd.MultiIndex.from_product([[1, 5, 22], MODELS], names=["h", "model"]))
    tab.to_csv(a.out / "contamination_years.csv")
    print("written to", a.out)


if __name__ == "__main__":
    main()
