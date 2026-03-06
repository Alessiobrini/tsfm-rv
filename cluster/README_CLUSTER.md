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
