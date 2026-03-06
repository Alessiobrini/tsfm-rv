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

## Submission Order

Steps 1-4 are forecast jobs and can run in parallel (no dependencies between them).
Step 5 is portfolio evaluation and must wait for all forecast jobs to finish.

### Step 1 — Forex baselines (CPU, single job)
Runs covariance baselines (element-HAR, HAR-DRD) for all 15 forex pairs.
```bash
sbatch cluster/run_cov_baselines.slurm
```

### Step 2 — Stock baselines (CPU, single job)
Runs covariance baselines (element-HAR, HAR-DRD) for all 820 stock pairs.
```bash
sbatch cluster/run_cov_baselines_stocks.slurm
```

### Step 3 — Forex + futures foundation models (GPU, single job)
Runs TSFMs (Chronos-Bolt, Moirai) element-wise on 15 forex pairs and 15 futures pairs.
```bash
sbatch cluster/run_cov_foundation_small.slurm
```

### Step 4 — Stock foundation models (GPU, array job: 82 tasks)
Runs TSFMs element-wise on all 820 stock pairs, split across 82 array tasks.
```bash
sbatch cluster/run_cov_foundation_stocks.slurm
```

### Step 5 — Portfolio evaluation (after steps 1-4 finish)
Computes GMV portfolio weights and out-of-sample performance for all asset classes (forex, futures, stocks).
Note the job IDs printed by each `sbatch` in steps 1-4, then substitute them below:
```bash
sbatch --dependency=afterok:<ID1>:<ID2>:<ID3>:<ID4> cluster/run_portfolio_eval.slurm
```

## Collecting Results

Forecast results are in `results/covariance/{asset_class}/forecasts/`.
For stock TSFM array jobs, each chunk produces a separate npz file that needs merging
before portfolio evaluation.

---

# Realized Variance Forecasting

## Submission Order

All scripts use `--skip-existing` so re-runs safely skip completed CSVs.

Steps 1-6 are forecast jobs and can run in parallel (no dependencies between them).
Step 7 is evaluation and must wait for all forecast jobs to finish.

### Step 1 — CAPIRe baselines (CPU, array job: 29 tasks)
Runs econometric baselines (HAR, Log-HAR, HAR-J, HAR-RS, HARQ, Realized GARCH, ARFIMA) for all 29 CAPIRe tickers.
```bash
sbatch cluster/run_rv_baselines_capire.slurm
```

### Step 2 — VOLARE stock baselines (CPU, array job: 40 tasks)
Runs the same econometric baselines for all 40 VOLARE stock tickers.
```bash
sbatch cluster/run_rv_baselines_volare_stocks.slurm
```

### Step 3 — VOLARE FX + futures baselines (CPU, single job)
Runs econometric baselines for 5 FX pairs and 5 futures contracts.
```bash
sbatch cluster/run_rv_baselines_volare_small.slurm
```

### Step 4 — CAPIRe foundation models (GPU, array job: 29 tasks)
Runs all TSFMs (Chronos-Bolt, Chronos-2, Moirai, Lag-Llama, Kronos) for all 29 CAPIRe tickers.
```bash
sbatch cluster/run_rv_foundation_capire.slurm
```

### Step 5 — VOLARE stock foundation models (GPU, array job: 40 tasks)
Runs all TSFMs for all 40 VOLARE stock tickers.
```bash
sbatch cluster/run_rv_foundation_volare_stocks.slurm
```

### Step 6 — VOLARE FX + futures foundation models (GPU, single job)
Runs all TSFMs for 5 FX pairs and 5 futures contracts.
```bash
sbatch cluster/run_rv_foundation_volare_small.slurm
```

### Step 7 — Evaluation (after steps 1-6 finish)
Computes metrics, Diebold-Mariano tests, and Model Confidence Sets for all datasets.
Note the job IDs printed by each `sbatch` in steps 1-6, then substitute them below:
```bash
sbatch --dependency=afterok:<ID1>:<ID2>:<ID3>:<ID4>:<ID5>:<ID6> cluster/run_rv_evaluation.slurm
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
