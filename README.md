# Can Time Series Foundation Models Forecast Realized Volatility?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

**Author:** Alessio Brini ([alessio.brini@duke.edu](mailto:alessio.brini@duke.edu))

---

## Overview

We evaluate nine zero-shot time series foundation models (TSFMs) against six econometric specifications for forecasting realized volatility on 50 assets from the [VOLARE](https://arxiv.org/abs/2602.19732) dataset (equities, FX, futures; 2015-2026) at horizons of 1, 5, and 22 days.

**Key findings:**
- The best TSFMs outperform the best econometric benchmarks across all horizons and asset classes.
- Sundial, a flow-matching generative model, achieves the lowest QLIKE loss at every horizon and enters the Model Confidence Set for 98-100% of equities; its advantage over Log-HAR grows from 11% at h=1 to 37% at h=22.
- Performance is highly heterogeneous across architectures: model selection within the TSFM class matters as much as the choice between TSFMs and econometric benchmarks.

## Repository Structure

```
code/
  config.py              # Central configuration (walk-forward params, context length)
  data_loader.py         # Data loading for VOLARE and CAPIRe datasets
  features.py            # HAR regressor construction
  models/                # Model implementations
    har.py               #   HAR, HAR-J, HAR-RS, HARQ, Log-HAR
    arfima.py            #   ARFIMA (long memory)
    foundation.py        #   TSFM wrappers (Chronos-Bolt, TimesFM, Moirai, Sundial, Toto, etc.)
  forecasting/
    rolling_forecast.py  # Fixed-window walk-forward and zero-shot forecast drivers
  evaluation/            # Loss functions, Diebold-Mariano test, Model Confidence Set, MZ regression
  run_baselines_volare.py      # Run econometric baselines
  run_foundation_volare.py     # Run TSFM zero-shot forecasts
  run_evaluation_volare.py     # Compute metrics, DM tests, MCS
  run_advanced_evaluation.py   # MZ regressions, Giacomini-Rossi fluctuation tests
  process_results.py           # Generate LaTeX tables
  generate_figures.py          # Generate PDF figures
cluster/
  *.slurm                # SLURM scripts for GPU cluster execution
  setup_models.sh        # Environment setup for cluster
```

## Data

This repository does **not** include the underlying data. Realized volatility series are from the [VOLARE dataset](https://arxiv.org/abs/2602.19732) (VOLatility Archive for Realized Estimates). Download the bulk dataset and place files under `data/raw/volare/`.

## Models Evaluated

| Econometric | Time Series Foundation Models |
|---|---|
| HAR (Corsi, 2009) | Chronos-Bolt-Small (Amazon) |
| HAR-J (Andersen et al., 2007) | Chronos-Bolt-Base (Amazon) |
| HAR-RS (Patton and Sheppard, 2015) | Moirai 2.0-Small (Salesforce) |
| HARQ (Bollerslev et al., 2016) | Moirai-MoE-Small (Salesforce) |
| Log-HAR (Corsi, 2009) | TimesFM 2.5 (Google) |
| ARFIMA (Granger, 1980) | Toto (Datadog) |
| | Sundial (Tsinghua University) |
| | Lag-Llama (Rasul et al., 2024) |
| | TTM (IBM) |

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
# 1. Econometric baselines
python code/run_baselines_volare.py

# 2. Foundation model zero-shot forecasts (GPU)
python code/run_foundation_volare.py

# 3. Evaluation (metrics, DM tests, MCS)
python code/run_evaluation_volare.py

# 4. Advanced evaluation (MZ regressions, GR tests)
python code/run_advanced_evaluation.py

# 5. Generate tables and figures
python code/process_results.py
python code/generate_figures.py
```

Results are saved to `results/`.
