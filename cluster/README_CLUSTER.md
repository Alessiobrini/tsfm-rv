# Cluster Setup — Realized Covariance Forecasting

## Conda Environment

```bash
# On the cluster:
conda create -n human-x-ai python=3.11 -y
conda activate human-x-ai

# Core packages
pip install pandas numpy scipy statsmodels matplotlib seaborn scikit-learn arch openpyxl

# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Foundation models
pip install chronos-forecasting transformers
pip install uni2ts

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from chronos import ChronosBoltPipeline; print('Chronos OK')"
```

## Running Jobs

### Quick test (FX, ~15 pairs)
```bash
sbatch cluster/run_cov_baselines.slurm       # baselines on forex
sbatch cluster/run_cov_foundation_small.slurm # TSFMs on forex + futures
```

### Full stock run (~820 pairs)
```bash
sbatch cluster/run_cov_baselines_stocks.slurm        # CPU baselines
sbatch cluster/run_cov_foundation_stocks.slurm        # GPU array job
```

### Portfolio evaluation (after forecasts complete)
```bash
sbatch cluster/run_portfolio_eval.slurm
```

## Collecting Results

After array jobs complete, results are in `results/covariance/{asset_class}/forecasts/`.
For stock TSFM array jobs, each chunk produces a separate npz file that needs merging
before portfolio evaluation.

---

# Realized Variance Forecasting

## Submission Order

All scripts use `--skip-existing` so re-runs safely skip completed CSVs.

### 1. Baselines (CPU-only)
```bash
# CAPIRe: 29 tickers as array job
BASE_CAP=$(sbatch --parsable cluster/run_rv_baselines_capire.slurm)

# VOLARE stocks: 40 tickers as array job
BASE_VOL_STK=$(sbatch --parsable cluster/run_rv_baselines_volare_stocks.slurm)

# VOLARE FX + futures: single job
BASE_VOL_SM=$(sbatch --parsable cluster/run_rv_baselines_volare_small.slurm)
```

### 2. Foundation models (GPU)
```bash
# CAPIRe: 29 tickers, 5 models each
TSFM_CAP=$(sbatch --parsable cluster/run_rv_foundation_capire.slurm)

# VOLARE stocks: 40 tickers, 5 models each
TSFM_VOL_STK=$(sbatch --parsable cluster/run_rv_foundation_volare_stocks.slurm)

# VOLARE FX + futures: single GPU job
TSFM_VOL_SM=$(sbatch --parsable cluster/run_rv_foundation_volare_small.slurm)
```

### 3. Evaluation (after all forecasts complete)
```bash
sbatch --dependency=afterok:${BASE_CAP}:${BASE_VOL_STK}:${BASE_VOL_SM}:${TSFM_CAP}:${TSFM_VOL_STK}:${TSFM_VOL_SM} \
    cluster/run_rv_evaluation.slurm
```

## Expected Output

| Dataset | Tickers | Models | Horizons | CSVs |
|---------|---------|--------|----------|------|
| CAPIRe | 29 | 11 | 3 | 957 |
| VOLARE stocks | 40 | 11 | 3 | 1,320 |
| VOLARE FX | 5 | 11 | 3 | 165 |
| VOLARE futures | 5 | 11 | 3 | 165 |
| **Total** | | | | **2,607** |

Results: `results/forecasts/` (CAPIRe) and `results/volare/forecasts/` (VOLARE).
