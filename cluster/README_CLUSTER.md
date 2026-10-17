# Cluster Setup — Realized Covariance Forecasting

> The covariance and portfolio scripts (`run_cov_*.slurm`, `run_portfolio_eval.slurm`) call
> Python files removed from `code/` in commit `2df2f77`. Restore them from git history before use.

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

# Realized Variance Forecasting (first submission, March 2026)

The March 2026 scripts for the first submission (CAPIRe and VOLARE runs, the early context-length
runs, and their evaluation jobs) are in `cluster/_archive/`. Every stored forecast behind the
current paper comes from the June 2026 revision pipeline below, so those scripts are kept for
reference only.

---

# June 2026 pipeline: point target, volatility scale

This is the cluster workflow that produced every stored forecast behind the current paper
(June 2026). The scripts that ran are listed under "Submission order" below.

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

> The old IJF VOLARE results are preserved locally at `results/_archive/volare_ijf/`.
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
All scripts use `--skip-existing`, so re-submitting safely resumes. The jobs are independent and
can run in parallel. Evaluation, tables, and figures are computed afterwards on a workstation from
the stored forecasts, following the pipeline in the top-level `README.md` (steps 3 onward).

```bash
# 1. Econometric baselines, daily re-estimation (CPU array, 50 tickers, 8 models)
sbatch cluster/run_rev_baselines_daily.slurm
#    Two tasks hit the time limit (ARMA and MEM at h = 22 for GS and META):
sbatch cluster/run_gs_meta_h22_fix.slurm

# 2. Fast TSFMs (GPU array, 50 tickers): chronos-bolt x2, timesfm-2.5, moirai-2.0-small, ttm
sbatch cluster/run_rev_tsfm_fast.slurm

# 3. Heavy/sampling TSFMs (GPU array, 50 tickers): sundial, toto, lag-llama, moirai-moe-small
sbatch cluster/run_rev_tsfm_heavy.slurm

# 4. Context-length sensitivity: the nine TSFMs at 128, 256, and 512 days (appendix table)
sbatch cluster/run_rev_context_sensitivity.slurm

# 5. Averaged-target arm (appendix table): the full arm, then the econometric models re-fit daily
sbatch cluster/run_rev_avg_target.slurm
sbatch cluster/run_rev_avg_target_daily.slurm
#    Lag-Llama and Sundial for steps 4 and 5, rerun after a dependency fix:
sbatch cluster/run_rev_lagsundial_fix.slurm
```

The TSFM runs that preceded the upper winsorization cap wrote a few forecasts outside the bounds.
After copying the results back, apply the bounds with
`python code/winsorize_stored_forecasts.py --apply` (a dry run without `--apply` lists what it
would change).

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
