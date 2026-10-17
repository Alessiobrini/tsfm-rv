# Revision audit, 2026-06-11 (post-IJF rejection rewrite)

Independent read of `paper/main.tex` against the two IJF referee reports
(`paper/archive_ijf_rejected/reviews/`), the editor letter, and the revision plan
(`~/.claude/plans/i-got-a-review-velvety-conway.md`). Every numeric claim in the text was
cross-checked against the current files in `paper/tables/`, and the five figures referenced in
the manuscript were rasterized and inspected visually.

**Bottom line:** the methodology redesign (point-in-time target, iterated multistep, mean point
forecast, 1000-day window, constrained HAR, winsorization, ARMA/MEM/ARFIMA-MLE, loss ratios,
symmetric MZ, contamination section, re-narrated abstract) is in place and the prose matches the
refreshed loss-ratio, MCS, and DM tables. But the figure layer and two tables are stale from the
pre-revision run and directly contradict the new text, and Section 5.1 quotes pooled all-50
numbers while attributing them to the 40-equity table. Referee 2 caught exactly this kind of
text-vs-table slip last time (the 0.188 < 0.191 error), so these are resubmission blockers.

---

## A. Verified as correctly addressed (no action needed)

- R1 1.3 / 1.3.1: point-in-time RV_{t+h} target, iterated multistep for pure-RV models, direct
  estimation for augmented HAR variants with literature justification (Sec 4.1, 4.4).
- R2 major 1: conditional mean point forecast, QLIKE primary; alignment stated explicitly.
- R1 1.5 / R2 major 3: 1000-day window, matched TSFM context, footnoted 512 exceptions
  (Moirai-MoE, TTM); average loss ratios vs Log-HAR added (Tab. loss_ratios).
- R1 1.4.1: non-negativity-constrained HAR, winsorization to in-sample support replacing the
  1e-10 floor; Log-HAR promoted to headline benchmark.
- R1 1.4.2 / 1.4.3: ARMA on log-RV with BIC selection and MEM (Engle 2002) added; ARFIMA now
  local-Whittle d plus ARMA on the fractionally differenced series.
- R2 major 2: abstract fully re-narrated ("not a free lunch"); the 0.188 < 0.191 slip is moot
  because the matched-information check is superseded by the matched 1000-day design.
- R2 contamination: Sec 4.3 is a genuine strengthening (burden-shift argument, TTM
  smallest-capacity argument, explicit concession for undisclosed corpora, carried to conclusion).
- R2 min2: MZ correction applied symmetrically to all models.
- R1 minors fixed: NW standard error claim removed; OLS efficiency wording now conditional;
  RV_t double role resolved by the point target.
- Numeric spot checks that PASS: all loss-ratio values quoted in abstract, intro, Sec 5.1, and
  conclusion (TTM 0.972/0.965/0.983, Sundial 0.988/1.060/1.178, HAR 0.996/1.010/1.065, HARQ
  5.078) match `table_loss_ratios.tex`. MCS all-horizon averages quoted in Sec 6 (TTM 0.97,
  Log-HAR 0.81, HAR 0.61, Sundial 0.51, Toto 0.31, Lag-Llama 0.29, Moirai 0.20, MoE 0.17, Bolt
  0.05 to 0.06, TimesFM 0.03) match `table_equity_mcs.tex`. DM narrative matches
  `table_dm_summary.tex` (TTM 70.1/69.6/27.0, Log-HAR 30.1 at h=22). FX and futures panel
  claims match `table_fx_futures_metrics.tex`.

---

## B. Blockers: text, tables, and figures that contradict each other

### B1. Section 5.1 quotes all-50 pooled numbers but labels them "equity" and cites the 40-equity table
The paragraph beginning "On the pooled cross-sectional average, several foundation models post
low QLIKE" (main.tex ~line 378) says "TTM achieves the lowest equity QLIKE (0.190)" and cites
Tab. main_results, which is the 40-equity table where TTM h=1 QLIKE is 0.194. Every number in
that paragraph (0.190, 0.193, 0.195, 0.196, 0.202, 0.211, 0.216 to 0.218; the h=5 set 0.285,
0.296, 0.299, 0.314, 0.319, 0.323; the h=22 set 0.499, 0.510, 0.513, 0.573, 0.608) is in fact
the 50-asset weighted blend of the equity, FX, and futures panels. Check: TTM h=1 blend =
(40(0.194) + 5(0.163) + 5(0.184))/50 = 0.190. Same arithmetic reproduces every quoted value,
including the Toto paragraph (pooled 0.234, 0.383, 0.588 and mean R2 of about -0.04 at h=1,
which only holds with the futures Toto R2 of -3.848 in the blend; the equity-only Toto R2 is
+0.400).

Fix: pick one basis and make text and citation agree. Either (a) re-quote from the equity table
(then re-derive the ranking sentences: e.g. at h=22 the equity table has Lag-Llama 0.532 BELOW
Log-HAR 0.539, so "Log-HAR second (0.510); Lag-Llama (0.513) follows" reverses), or (b) present
the all-50 pooled average explicitly, add it as a tabulated column or panel, and label the
discussion as such. Re-verify every ordering claim against whichever basis is chosen, and
re-verify the Toto GC R2 of about -21 claim against the per-asset results CSVs.

### B2. fig_qlike_boxplot is stale and contradicts both its caption and the text
The figure y-label reads "QLIKE ratio (model / HAR)" while the caption and the text (~line 408)
say the denominator is Log-HAR. Log-HAR appears as a box that is not identically 1, so the
plotted denominator is HAR. Worse, the TTM box at h=1 sits with median around 1.4, contradicting
"TTM is the only model whose distribution sits predominantly below one at all three horizons"
and contradicting the loss-ratio table (TTM 0.972). The h=22 panel has medians near 0.1, which
is implausible under the current results. This figure was not regenerated from the revised run.
Fix: regenerate from `results/volare` with Log-HAR as denominator via the committed generator
script, then re-verify the caption and the sentence describing it.

### B3. fig_persistence_drivers is stale
The TTM panel shows equity points clustered around ratio 1.5 and FX points at 3 to 6.5 (vs HAR),
which is impossible given the current tables (TTM FX QLIKE 0.163 vs HAR 0.167). The Sundial
panel shows most points near 0.85, also inconsistent with a 0.988 loss ratio. Regenerate from
current results, then re-verify the quoted correlations and the claim "TTM beating HAR on the
largest share of assets in both high- and low-persistence groups" (~line 422), which the current
panel visibly contradicts.

### B4. fig1_forecast_vs_actual repeats the exact variance-vs-volatility mix Referee 2 flagged
The y-axes read "Realized Variance" and the legend says "Actual RV" while the caption says
"Forecast vs. actual realized volatility". R2 minor 3 cited Figure 1 by name for mixing the two;
this cannot survive a second look. Also the legend shows Log-HAR, Sundial, and Moirai-MoE-S, but
the caption claims the two representative TSFMs are "TTM and Sundial". Fix: regenerate on the
volatility scale with axis label "Realized volatility", include TTM (the headline model), and
make caption and legend agree.

### B5. GR fluctuation figures: sign convention and legend
In gr_fluctuation_h1 every model's rolling DM path sits between -2 and -8, and the caption says
"Negative values indicate the comparison model outperforms Log-HAR". Under that convention
ARFIMA, ARMA, MEM, and HARQ (loss ratios 1.5 to 5.1) would be significantly BEATING Log-HAR for
the whole sample, contradicting every table in the paper. The sign convention in the generator
or the caption is flipped; verify which and fix caption, axis annotation, and the in-text
narrative (~lines 465 and 485: "move toward and across the critical-value bands around the
2019 to 2021 period" does not describe a panel where all paths are beyond the band at all
times). Separately, the legend identifies only 8 series plus the critical-value line, several
legend swatches render as near-identical gray, and the legend colors do not match the plotted
lines. R2 minor 6 (colors barely distinguishable) is therefore still effectively open. If the
legend is intentionally split across the three horizon subfigures, that fails when a reader
looks at one panel; give each panel a complete, color-matched legend or move to a small-multiples
layout.

### B6. table_context_sensitivity.tex is the OLD pre-revision table
It contains exactly the rejected draft's numbers (Sundial h=5 ctx512 = 0.153 and TTM h=22
ctx512 = 0.191, the very values Referee 2 quoted), and TTM h=1 ctx512 = 0.373 is inconsistent
with the new equity TTM of 0.194. The grid is 128/256/512 with no 1000 column even though 1000
is now the default, so the robustness paragraph (~line 509) rests on stale evidence. A
context-sensitivity SLURM generator was recently committed in the code repo; regenerate under
the revised pipeline (point target, mean forecast, vol scale) including ctx = 1000, then rewrite
the paragraph and the appendix description from the new numbers.

### B7. table_computational_cost: stale caption and missing models
The caption says "n approximately 2,274 out-of-sample steps", which is the old 512-day-window
count (2786 - 512); the new design gives about 1,786, the number the text itself states at
~line 325. The table also lists only 6 econometric models; ARMA and MEM are missing although the
paper claims 8 econometric specifications and 17 models total. Extend and re-caption.

### B8. tab:main_results caption contradicts the new methodology
The equity table caption and footnote still say the dagger "reflects near-zero forecasts from
levels-based OLS", but the text (~lines 386 and 506) now states that constrained estimation plus
winsorization means no near-zero forecasts occur. No dagger even appears in the equity panels.
Reword the caption and footnote (the FX/futures table's plain "dagger marks QLIKE > 1" is fine;
harmonize the two).

### B9. Dangling appendix promise: averaged-target results
The robustness paragraph "Forecast target: point-in-time vs. averaged" (~line 503) says the
averaged-target arm is "reported in the appendix", and Sec 4.4 says the same, but no such table
exists in `paper/tables/` or in the appendix input list. R1 made the target choice his top
concern and the plan was to keep both, following Patton and Sheppard. Generate the appendix
table(s) from `results/volare_avg` and input them, or, if the run is not available, change the
text to state the result without promising a table (weaker; prefer adding the table).

---

## C. Referee minor comments still unaddressed in the current text

- C1 (R1 minor 3:43): "yet achieves accuracy comparable to far more complex alternatives"
  (~line 86) survives verbatim. Name the alternatives (e.g. ARFIMA and the ML methods surveyed
  in Christensen et al.) or delete the clause.
- C2 (R1 minor): "multi-scale" still used (~lines 166, 171, 174 and possibly elsewhere). Either
  switch to "multi-horizon aggregation of lagged RV" / "multi-lag" or define the term once;
  R1 explicitly queried it.
- C3 (R1 minor 8:48): ARFIMA text still writes d in (0, 0.5), excluding zero, which is the exact
  point R1 raised. Write d in [0, 0.5) and note that d = 0 nests the ARMA benchmark, which the
  new ARMA row makes testable.
- C4 (R2 min5): the conclusion (~line 527) says "restricted to point forecasts, i.e., the
  conditional mean", reproducing the imprecision R2 flagged (a point forecast need not be any
  particular functional). Write "restricted to point forecasts, specifically the conditional
  mean of each model's predictive distribution".
- C5 (R2 min1, second half): the combined FX/futures table and the equity table still carry no
  statistical-test indicators; the dagger is not a test. Add MCS-membership marks or
  DM-vs-Log-HAR significance stars to the metric tables, which is what the referee asked for
  when requesting the merge.

---

## D. Smaller factual and consistency items

- D1: "approximately two orders of magnitude smaller" for FX RV (~line 129) is wrong; the factor
  is about 8 (3e-5 vs 2.5e-4), one order of magnitude.
- D2: futures h=1, text says Sundial and TTM "tie for the lowest QLIKE" (both 0.184) but the
  table bolds only TTM; bold both or note the tie at displayed precision.
- D3: Tab. 1 reports realized variance moments while the paper's headline object is volatility.
  Defensible for raw-data description, but add one sentence stating that descriptives are for
  RV as distributed in VOLARE while all modeling and evaluation are on the square root, or
  convert the moments to the volatility scale. R1 1.2 and R2 min3 demanded strict consistency.
- D4: two stale "REVIEW NOTE (Comment 7)" and "(Comment 23)" LaTeX comments in main.tex refer to
  a different review's numbering; delete before resubmission.
- D5: the winsorization paragraph claims the bound "binds for only a negligible fraction of
  forecast-date pairs" with no number. Compute the binding rate from the results and state it;
  the upper cap (in-sample max) is attackable as preventing record-volatility forecasts around
  COVID, so back the claim with data.
- D6: abstract says "TTM enters for 97% of assets" without noting this is the average across
  horizons (98/98/94); add "averaged across horizons" or quote the per-horizon range.
- D7: orphan files in the paper repo (table_window_512*, qlike_floor_sensitivity,
  table_capire_*, table_portfolio, table_cov_*, fig3_cumulative_qlike_diff, fig_bootstrap_ci,
  plus the now-superseded table_fx_metrics / table_futures_metrics) are unreferenced; prune or
  leave, but confirm none is input anywhere.
- D8: author style rule (global CLAUDE.md): no dashes anywhere, including en-dash ranges. The
  draft uses "0.39--0.46", "2024--2025", "2--3.5\%", "2015--2026" style ranges throughout the
  prose. Sweep prose ranges to "X to Y" form (tables can be discussed with the author if
  needed). Also, after an acronym is defined, use only the acronym: the conclusion writes
  "Mincer--Zarnowitz bias correction" although MZ is defined in Sec 4.4.

---

## E. Process checklist before resubmission (per project CLAUDE.md)

1. Regenerate every stale figure via committed scripts (no hand-edited PDFs), on the volatility
   scale, sized via the set_size helper.
2. Recompile with latexmk, grep the log for overfull boxes, rasterize changed pages, and visually
   inspect (compile-and-render.sh helper). Fig 1, the GR panels, and the boxplot must be
   re-inspected after regeneration.
3. Re-run the text-vs-table number sweep after B1 is resolved (the /fact-check or
   paper_verification skill covers this).
4. Keep main_blue.tex in sync with main.tex or regenerate it after edits.
5. Verify any new bib entries via OpenAlex + Crossref (keys engle2002, marcellino2006, box1970,
   nelson1992, robinson1995, taylor2017, geweke1983 all exist in references.bib; run
   verify_bib.sh if not yet done this round).
6. Commit and push the paper repo so Overleaf stays current.
