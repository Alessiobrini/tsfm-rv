# Audit fix progress (REVISION_AUDIT_2026-06-11.md)

Status key: done (rendered/looked where applicable) · DEFERRED (needs cluster) · DISPUTED

## B — blockers
- B1 50-asset vs equity aggregation in 5.1 — **DONE**. Verified text numbers = 50-asset pooled blend. Added explicit all-50 pooled table (`gen_pooled50.py` -> Table 6 `tab:pooled50`, QLIKE+R2 at h=1/5/22), relabeled 5.1 to all-50, re-pointed loss-ratio caption to Table 6. Fixed a NEW error found en route: Toto mean R2 is +0.151 at h=5 (text said "negative at every horizon"). Rendered + verified.
- B2 fig_qlike_boxplot — **DONE**. Denominator HAR->Log-HAR, dropped Log-HAR box, y-label fixed. Regenerated; TTM median 0.972/0.967/0.977 below 1 at all 3 horizons. Rendered + verified.
- B3 fig_persistence_drivers — **DONE**. Regenerated from current results (HAR denominator kept per caption). Text fixed: 6/8 insignificant, Sundial (r=0.32,p=0.02) & Lag-Llama (r=0.30,p=0.03) significant. Rendered + verified.
- B4 fig1_forecast_vs_actual — **DONE**. Variance->volatility labels; Moirai-MoE -> TTM (matches caption). Rendered + verified (vol scale).
- B5 GR fluctuation — **DONE**. Code sign confirmed (+DM = model beats Log-HAR); caption+text were flipped, now corrected. Plot rewritten to fixed 9-FM set + consistent colors + TTM bold (was top-8-by-sup_stat, which dropped TTM). Regenerated 3 panels; narrative rewritten to match. Rendered + verified.
- B6 table_context_sensitivity — **DONE**. Cluster sweep finished (all 9 TSFMs, ctx 128/256/512 + 1000 from main); `gen_context_sensitivity.py` built Table 18; paragraph rewritten with real model-specific patterns (TTM/Sundial best at 512-1000; TTM low+stable so headline unaffected). Rendered + verified.
- B7 table_computational_cost — **DONE**. Caption count 2,274->1,786; added ARMA (0.013 s/step) & MEM (0.005) rows from real local relative timing scaled to the table's ARFIMA value.
- B8 tab:main_results caption — **DONE**. Removed "near-zero forecasts from levels-based OLS" dagger claim; harmonized with FX/futures table; added scale note. Regenerated + rendered.
- B9 averaged-target appendix — **DONE**. Cluster run finished (all 17 models, --target-kind avg); `gen_avg_target_table.py` built Table 19. Ranking reproduced: TTM only FM beating Log-HAR at every horizon (0.973/0.960/0.959 vs 0.972/0.965/0.983 point-in-time); Sundial only at h=1. Rendered + verified. NOTE: lag-llama+sundial first failed (env drift to pandas 3.0.3/transformers 4.57.6); fixed by pinning pandas 2.1.4 + transformers 4.43.4 + granite-tsfm 0.3.1 (see requirements.txt; all 9 models verified to run).

## C — referee minors
- C1 "far more complex alternatives" — **DONE**. Named ARFIMA + ML forecasters \citet{christensen2023ml}.
- C2 "multi-scale" — **DONE**. -> "aggregate lagged RV over daily/weekly/monthly horizons" / "multi-horizon".
- C3 ARFIMA d range — **DONE**. (0,0.5)->[0,0.5); noted d=0 nests ARMA (testable).
- C4 conclusion point-forecast wording — **DONE**. "i.e." -> "specifically the conditional mean of each model's predictive distribution".
- C5 test indicators in tables — **DONE**. Added MCS-majority marker (^*) to equity + combined tables (model in MCS for >50% of panel assets at that horizon); captions define it. Rendered + verified.

## D — smaller items
- D1 FX "two orders of magnitude" -> one order (~factor of 8) — **DONE**.
- D2 futures h=1 Sundial/TTM tie — **DONE**. Generator now bolds displayed-precision ties; both 0.184 bold.
- D3 Tab 1 variance moments — **DONE**. Added variance-vs-volatility consistency sentence.
- D4 stale REVIEW NOTE comments — **DONE**. Both deleted.
- D5 winsorization binding rate — **DONE**. Computed: 0.05% of 5.3M pairs (upper cap 0.01%), 0 outside bounds. Stated + answered the COVID-cap concern. Also reconciled methods text (code uses full-sample support, not the "1000-day window" the text claimed).
- D6 abstract "97%" — **DONE**. Added "on average across horizons" in abstract/intro/conclusion.
- D7 orphan files — **DONE**. Confirmed none input; pruned 11 tables + 2 figures (window_512*, qlike_floor, capire*, cov*, portfolio, fx/futures_metrics, bootstrap_ci, cumulative_qlike).
- D8 dashes + MZ acronym — **DONE**. Prose ranges -> "X to Y" (126,315,382,388,456); conclusion "Mincer--Zarnowitz bias correction" -> "MZ bias correction".

## E — process
- compile clean (50 pp, 0 overfull, 0 undefined); rendered+inspected all changed figures/tables — **DONE**.
- number sweep (pooled50, Toto R2, persistence r/p, winsor rate, factor-of-8) — **DONE**.
- main_blue.tex regenerate, commit/push — in progress.

## Remaining (cluster, user action)
- Submit `cluster/run_rev_context_sensitivity.slurm` (B6) and `cluster/run_rev_avg_target.slurm` (B9) on DCC; pull results; then `python code/gen_context_sensitivity.py` + build avg-target appendix tables, and replace the two red pending notes with the real numbers.
