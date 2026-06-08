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

---

# Revision pipeline (post-IJF) — point target, volatility scale

This is the cluster workflow for the **revised** paper. It supersedes the steps
above for the VOLARE results. The four `run_rev_*.slurm` scripts encode every
revision change; the older `run_rv_*` / `run_new_tsfms_*` scripts are kept only
for reference.

## What changed (baked into the code defaults; passed explicitly in the scripts)
- **Target:** point-in-time `RV_{t+h}` (`--target-kind point`), not the h-day average.
- **Scale:** forecast realized **volatility** `sqrt(RV)` (`--scale vol`); QLIKE is
  computed on the variance scale internally (Patton-robust); MSE/MAE on the vol scale.
- **Multi-step:** iterated for pure-RV models (HAR, Log-HAR, ARFIMA, ARMA, MEM);
  direct for the augmented HAR variants (HAR-J/RS/Q — auxiliary regressors can't be
  projected).
- **Point forecast:** conditional **mean** of each TSFM (was the median).
- **Window/context:** 1000-day econometric window, matched 1000 TSFM context.
- **Positivity:** Nelson-Cao-constrained level HAR + Log-HAR/MEM positive by
  construction; residual non-positive forecasts floored at the **min RV in the
  estimation window** (replaces the old 1e-10 floor).
- **New benchmarks:** ARMA(log-RV, IC-selected), MEM (Engle 2002); ARFIMA now uses
  local-Whittle `d` + IC `(p,q)`.

> The old IJF VOLARE results are preserved locally at `results/volare_ijf_archive/`.
> The revised jobs write fresh CSVs into `results/volare/`.

## Environment (one-time)
```bash
conda activate human-x-ai
source cluster/setup_models.sh        # Lag-Llama (from GitHub)
source cluster/setup_new_models.sh    # TimesFM 2.0, Toto, Sundial, Moirai-MoE
# Verify the nine TSFM backends import (TTM ships via granite-tsfm / tsfm_public):
python - <<'PY'
import importlib
for m in ["chronos","timesfm","uni2ts","gluonts","toto","tsfm_public"]:
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: print("MISS", m, "->", type(e).__name__)
import torch; print("CUDA:", torch.cuda.is_available())
PY
```

## Submission order
All scripts use `--skip-existing`, so re-submitting safely resumes. Baselines and
both TSFM groups are independent and run in parallel; evaluation runs last.

```bash
# 1. Econometric baselines (CPU array, 50 tickers, 8 models)
BASE=$(sbatch --parsable cluster/run_rev_baselines.slurm)

# 2. Fast TSFMs (GPU array, 50 tickers): chronos-bolt x2, timesfm-2.5,
#    moirai-2.0-small, ttm  -> preliminary full tables within ~a day.
FAST=$(sbatch --parsable cluster/run_rev_tsfm_fast.slurm)

# 3. Heavy/sampling TSFMs (GPU array, 50 tickers): sundial, toto, lag-llama,
#    moirai-moe-small.
HEAVY=$(sbatch --parsable cluster/run_rev_tsfm_heavy.slurm)

# 4. Evaluation — metrics, DM tests, MCS, LaTeX tables (after 1-3 finish)
sbatch --dependency=afterok:${BASE}:${FAST}:${HEAVY} cluster/run_rev_evaluation.slurm
```

## Expected output
50 assets (40 stocks + 5 FX + 5 futures) x 3 horizons x 17 models
(8 econometric + 9 TSFM) = **2,550** forecast CSVs in
`results/volare/forecasts/`, then metrics/tables in `results/volare/metrics/`
and `results/volare/tables/`.

## Appendix arm (h-day-average target)
To regenerate the legacy h-day-average results for the appendix (routed to
`results/volare_avg/`, leaving the main `results/volare/` untouched), run the
python entry points with `--target-kind avg` — both `run_baselines_volare.py`
and `run_foundation_volare.py` accept it (copy a `run_rev_*` script and change
`--target-kind point` to `--target-kind avg`).
