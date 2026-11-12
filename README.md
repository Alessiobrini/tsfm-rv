# Forecasting Realized Volatility with Time Series Foundation Models: A Comparison with Econometric Benchmarks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.05291-b31b1b.svg)](https://arxiv.org/abs/2607.05291)

**Author:** Alessio Brini ([alessio.brini@duke.edu](mailto:alessio.brini@duke.edu))

**Paper:** [arXiv:2607.05291](https://arxiv.org/abs/2607.05291)

---

## Overview

We evaluate nine zero-shot time series foundation models (TSFMs), spanning eight distinct architectures, against eight econometric specifications for forecasting realized volatility on 50 assets from the [VOLARE](https://arxiv.org/abs/2602.19732) dataset (equities, FX, futures) at horizons of 1, 5, and 22 days. We apply Diebold-Mariano tests, the Model Confidence Set, Mincer-Zarnowitz regressions, and Giacomini-Rossi fluctuation tests.

**Key findings:**
- The foundation models do not outperform the econometric benchmarks as a class.
- Only Tiny Time Mixers (TTM), the smallest model in the study (<1M parameters), has a lower average QLIKE loss ratio than Log-HAR at every horizon, by a narrow margin of 1.1 to 1.6%. For most assets the Model Confidence Set contains TTM together with Log-HAR and several other models.
- At the daily and weekly horizons TTM's edge comes mostly from the level of its forecasts. At the daily horizon, a Mincer-Zarnowitz recalibration estimated on past forecasts gives several foundation models the accuracy that TTM reaches without one. At h = 22 the recalibration raises the loss of every model.
- Accuracy differs far more across the foundation models than between the best of them and Log-HAR, so which foundation model one chooses matters more than whether one uses a foundation model or an econometric one.
- For an investor who targets a constant volatility, TTM's edge is worth a few basis points a year before trading costs.
- TTM and Log-HAR make different errors, so their equal-weight average matches the best single model and enters the Model Confidence Set for 98 to 100% of assets.

## Repository Structure

```
code/
  config.py                       # Central configuration (paths, walk-forward params, model IDs)
  data_loader.py                  # VOLARE long-format CSV loader
  features.py                     # HAR regressor construction
  utils.py                        # Logging / save helpers
  models/
    har.py                        # HAR, HAR-J, HAR-RS, HARQ, Log-HAR
    arfima.py                     # ARFIMA (long memory)
    arma.py                       # ARMA on log realized volatility
    mem.py                        # Multiplicative error model (Engle, 2002)
    foundation.py                 # TSFM wrappers (Chronos-Bolt, TimesFM, Moirai,
                                  #   Moirai-MoE, Lag-Llama, Toto, Sundial, TTM)
    ml.py                         # Supervised ML baselines (XGBoost, feed-forward network, LSTM)
  finetune/
    windows.py                    # Training and validation windows from a realized-variance panel
    common.py                     # Warmup, early stopping, resumable checkpoints
    train.py                      # Fine-tune TTM (all weights) or Sundial (LoRA)
  forecasting/
    rolling_forecast.py           # Walk-forward and zero-shot forecast drivers
  evaluation/
    loss_functions.py             # MSE, MAE, QLIKE, R2_OOS
    dm_test.py                    # Diebold-Mariano test
    mcs.py                        # Model Confidence Set
    mz_regression.py              # Mincer-Zarnowitz regressions
    gr_fluctuation.py             # Giacomini-Rossi fluctuation test

  # Pipeline entry points (run in the order documented below)
  run_baselines_volare.py         # 1. Econometric baselines
  run_foundation_volare.py        # 2. TSFM zero-shot forecasts (and fine-tuned, with --checkpoint)
  run_ml_volare.py                # Supervised ML baselines on the rolling window
  winsorize_stored_forecasts.py   # 2b. Clip stored forecast files to the origin-date bounds
  run_evaluation_volare.py        # 3. Metrics, DM tests, MCS
  run_advanced_evaluation.py      # 4. MZ regressions, Giacomini-Rossi tests
  compute_subsample_metrics.py    # 5. Pre/post-COVID subsample metrics
  process_results.py              # 6. Main accuracy and loss-ratio tables
  generate_figures.py             # 7. fig1_forecast_vs_actual, fig2_mcs_heatmap
  gen_fig_qlike_boxplot.py        # 8. fig_qlike_boxplot
  gen_fig_persistence_drivers.py  # 9. fig_persistence_drivers
  gen_pooled50.py                 # 10. table_pooled50
  gen_mcs_dm.py                   # 11. table_mcs_dm (merges two process_results tables)
  gen_mz_allh.py                  # 12. mz_bias_corrected (all horizons)
  run_combination_robustness.py   # 13. combination forecasts and metrics
  gen_combination_table.py        # 14. table_combination
  gen_context_sensitivity.py      # 15. table_context_sensitivity
  gen_avg_target_table.py         # 16. table_avg_target
  gen_descriptive_stats.py        # 17. descriptive statistics rows (pasted into main.tex)
  compute_bh_dm.py                # 18. Benjamini-Hochberg DM values quoted in a footnote

  # Robustness and appendix tables (each compute_* script writes metrics, each gen_* the table)
  compute_contamination.py, gen_contamination_tables.py           # table_contamination_split, table_contamination_years
  compute_economic_value.py, gen_economic_value.py                # table_economic_value
  compute_log_input.py, build_log_input_set.py, gen_log_input_tables.py   # table_log_input, table_log_input_mcs_dm
  dm_log_vs_vol.py                                                # table_log_vs_vol_dm
  compute_equal_context.py, gen_equal_context_table.py            # table_equal_context
  compute_combination_diagnostics.py, gen_combination_diagnostics.py      # table_combination_diagnostics, table_combination_pairs
  refit_lognormal_moments.py, compute_forecast_level.py, gen_forecast_level_table.py  # table_forecast_level
  run_log_har_refit22.py, compute_ml_baselines.py, gen_ml_tables.py       # table_ml_baselines, table_ml_diag
  score_finetune.py, gen_finetune_table.py                        # table_finetune

  _archive/                       # scripts no longer used by the paper, kept for reference

  # Helper modules (imported by entry points; do not run directly)
  run_baselines.py                # Helpers shared with run_baselines_volare.py
  run_evaluation.py               # Helpers shared with run_evaluation_volare.py
                                  #   and run_advanced_evaluation.py

cluster/
  *.slurm                         # SLURM scripts for GPU cluster execution
  setup_models.sh                 # Environment setup for cluster
```

## Data

This repository does **not** include the underlying data. Realized volatility series are from the [VOLARE dataset](https://arxiv.org/abs/2602.19732) (VOLatility Archive for Realized Estimates). Download the bulk dataset and place files under `data/raw/volare/`.

## Models Evaluated

| Econometric | Time Series Foundation Models |
|---|---|
| HAR (Corsi, 2009) | Chronos-Bolt-Small (Amazon) |
| Log-HAR (Corsi, 2009) | Chronos-Bolt-Base (Amazon) |
| HAR-J (Andersen et al., 2007) | Moirai 2.0-Small (Salesforce) |
| HAR-RS (Patton and Sheppard, 2015) | Moirai-MoE-Small (Salesforce) |
| HARQ (Bollerslev et al., 2016) | TimesFM 2.5 (Google) |
| ARFIMA (Granger, 1980) | Toto (Datadog) |
| ARMA (on log RV) | Sundial (Tsinghua University) |
| MEM (Engle, 2002) | Lag-Llama (Rasul et al., 2024) |
| | TTM (IBM) |

Log-HAR is the headline econometric benchmark: all relative loss ratios are computed against it.

## Reproduction

### Environment Setup

```bash
conda create -n tsfm-rv python=3.11
conda activate tsfm-rv
pip install -r requirements.txt
```

Some TSFMs require a CUDA-capable GPU for inference. Econometric baselines run on CPU.

The Hugging Face revisions of the foundation models are pinned in `code/config.py`, except TTM, which loads by branch name (512-96-ft-r2.1). `code/config.py` records the commit the runs used. Sundial and
Moirai-MoE draw their samples with a fixed seed. Toto's point forecast is the mean of 20 draws that
are not seeded, and Lag-Llama is seeded once per predictor, so a rerun of these two models gives
forecasts that differ from the stored ones. The difference is large for Toto, whose predictive
distribution is heavy-tailed.

### Pipeline

Run in order:

```bash
# Forecasts
python code/run_local.py --stages baselines foundation   # 1, 2. Econometric baselines (CPU) and TSFM zero-shot forecasts (GPU), all 50 assets
python code/run_local.py --stages baselines foundation --target-kind avg   # 1b. The same forecasts for the averaged target of step 16
python code/winsorize_stored_forecasts.py --apply  # 2b. Clip stored forecast files to the origin-date bounds

# Evaluation
python code/run_evaluation_volare.py       # 3. Metrics, DM tests, MCS
python code/run_advanced_evaluation.py --dataset volare --horizons 1 5 22 \
    --benchmark Log_HAR --gr-plot --latex  # 4. MZ regressions, Giacomini-Rossi tests
python code/compute_subsample_metrics.py   # 5. Pre/post-COVID subsample metrics

# Tables and figures for the paper
python code/process_results.py             # 6. table_equity_metrics, table_equity_metrics_median, table_fx_futures_metrics, table_loss_ratios
python code/generate_figures.py            # 7. fig1, fig2 in paper/figures/
python code/gen_fig_qlike_boxplot.py       # 8. fig_qlike_boxplot
python code/gen_fig_persistence_drivers.py # 9. fig_persistence_drivers
python code/gen_pooled50.py                # 10. table_pooled50
python code/gen_mcs_dm.py                  # 11. table_mcs_dm (after step 6)
python code/gen_mz_allh.py                 # 12. mz_bias_corrected
python code/run_combination_robustness.py --mcs  # 13. combination forecasts
python code/gen_combination_table.py       # 14. table_combination
python code/gen_context_sensitivity.py     # 15. table_context_sensitivity
python code/gen_avg_target_table.py        # 16. table_avg_target
python code/gen_descriptive_stats.py       # 17. descriptive statistics rows
python code/compute_bh_dm.py ttm Log_HAR   # 18. Benjamini-Hochberg DM footnote values
```

### Robustness and appendix tables

Each table has a `compute_*` script, which writes its metrics to `results/volare/metrics/`, and a
`gen_*` script, which writes the table to `paper/tables/`. Every script takes its input folders as
options and defaults to the layout above; run `python code/<script>.py --help` for the inputs. The
forecasts these scripts read come from the SLURM scripts in `cluster/` (log-volatility input,
512-day contexts, supervised learners, fine-tuned models, refits every 22 days). The economic-value
table needs the median daily quoted spread of each stock, which comes from the TAQ intraday
indicators licensed through WRDS and is not distributed. `compute_economic_value.py` reads it from
the path given by `--stock-spreads` and its docstring states the format.

### Fine-tuned TTM and Sundial

`code/finetune/train.py` fine-tunes TTM (all weights) and Sundial (LoRA adapters) on a panel of
daily realized variance for US stocks, with volatility as the input and the contexts of the
zero-shot runs (512 days for TTM, 1,000 for Sundial). Training windows end by 2020-12-31 and
validation windows in 2021, so the fine-tuned models are evaluated from 2022 on: their forecast files
start earlier, and only origins from 2022-01-03 are scored. The forecasts go to their own folder,
so they never enter the evaluation of the 17 models. The panel is built
from TAQ data licensed through WRDS and is not distributed; it needs the columns `permno`, `date`
and `rv5`.

```bash
python code/finetune/train.py --model ttm --panel panel.parquet --out checkpoints/ttm-ft
python code/finetune/train.py --model sundial --panel panel.parquet --out checkpoints/sundial-ft
python code/run_foundation_volare.py --models ttm-ft --checkpoint checkpoints/ttm-ft --all-tickers --results-dir results/tsfm_ft
python code/run_foundation_volare.py --models sundial-ft --checkpoint checkpoints/sundial-ft --all-tickers --results-dir results/tsfm_ft
```

Forecasts and metrics land in `results/volare/`. LaTeX tables and PDF
figures land in `paper/tables/` and `paper/figures/`.

Two tables in the paper (`table_pretraining_data.tex` and the TSFM summary
table inside `main.tex`) are authored by hand, as is `table_computational_cost.tex`,
which the paper does not currently include. `mz_regression_all.tex` is written
by step 4 to `paper/tables/`, and the descriptive statistics printed by step 17 are pasted into
`main.tex`. Every generated table matches the paper's copy byte for byte.
