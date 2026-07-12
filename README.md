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
- Foundation models do not deliver a uniform gain over the econometric benchmarks. Pooled QLIKE losses appear to favor several TSFMs, but the advantage is concentrated in a few outlier assets.
- Under average per-asset QLIKE loss ratios relative to Log-HAR, which weight each asset equally, only Tiny Time Mixers (TTM), the smallest model in the study (<1M parameters), beats Log-HAR at every horizon, and by a narrow margin (roughly 1.3-1.8%). The other eight TSFMs do not improve on a well-specified Log-HAR, and the econometric benchmarks (HAR, ARFIMA, ARMA, MEM) remain competitive throughout.
- A Mincer-Zarnowitz recalibration shows that much of the short-horizon edge reflects better-scaled forecasts rather than better prediction of volatility dynamics; a genuine informational gain remains only at the monthly horizon.
- A simple equal-weight average of TTM and Log-HAR matches the best single model and enters the Model Confidence Set for 98-100% of assets, so a forecaster need not identify the best model for each asset in advance.
- Performance varies so widely across TSFM architectures that which TSFM one chooses matters more than the broader choice between foundation and econometric models.

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
  run_foundation_volare.py        # 2. TSFM zero-shot forecasts
  run_evaluation_volare.py        # 3. Metrics, DM tests, MCS
  run_advanced_evaluation.py      # 4. MZ regressions, Giacomini-Rossi tests
  run_robustness.py               # 5. MZ bias correction, 252- vs 512-day window
  compute_subsample_metrics.py    # 6. Pre/post-COVID subsample metrics
  process_results.py              # 7. LaTeX tables for the paper
  generate_figures.py             # 8. fig1_forecast_vs_actual, fig2_mcs_heatmap
  gen_fig_qlike_boxplot.py        # 9. fig_qlike_boxplot
  gen_fig_persistence_drivers.py  # 10. fig_persistence_drivers

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

### Pipeline

Run in order:

```bash
# Forecasts
python code/run_baselines_volare.py        # 1. Econometric baselines (CPU)
python code/run_foundation_volare.py       # 2. TSFM zero-shot forecasts (GPU)

# Evaluation
python code/run_evaluation_volare.py       # 3. Metrics, DM tests, MCS
python code/run_advanced_evaluation.py     # 4. MZ regressions, Giacomini-Rossi tests
python code/run_robustness.py              # 5. MZ bias correction, 512-day window
python code/compute_subsample_metrics.py   # 6. Pre/post-COVID subsample metrics

# Tables and figures for the paper
python code/process_results.py             # 7. Most LaTeX tables in paper/tables/
python code/generate_figures.py            # 8. fig1, fig2 in paper/figures/
python code/gen_fig_qlike_boxplot.py       # 9. fig_qlike_boxplot
python code/gen_fig_persistence_drivers.py # 10. fig_persistence_drivers
```

Forecasts and metrics land in `results/volare/`. LaTeX tables and PDF
figures land in `paper/tables/` and `paper/figures/`.

Three tables in the paper (`table_computational_cost.tex`,
`table_pretraining_data.tex`, `mz_regression_all.tex`) are authored by hand
and are not regenerated by any script.
