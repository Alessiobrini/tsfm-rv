#!/bin/bash
# =============================================================================
# run_rev_tsfm_fast.slurm — REVISION pipeline, FAST TSFM group (GPU array job)
# =============================================================================
# Post-IJF revision. Zero-shot TSFM forecasts for the FIVE fast models on all
# 50 VOLARE assets (40 stocks + 5 FX + 5 futures) at horizons h = 1, 5, 22.
#
# Fast group (deterministic or quantile-head; no heavy sampling):
#     chronos-bolt-small, chronos-bolt-base, timesfm-2.5, moirai-2.0-small, ttm
# (The slow samplers — sundial, toto, lag-llama, moirai-moe-small — are in
#  run_rev_tsfm_heavy.slurm.)
#
# What changed vs. the original submission (all baked into the python defaults,
# but passed explicitly here for an auditable command line):
#   --target-kind point : forecast point-in-time RV_{t+h} (NOT the h-day average)
#   --scale vol         : forecast realized VOLATILITY = sqrt(RV); QLIKE is
#                         computed on the variance scale internally
#   --context-length 1000 : matched to the 1000-day econometric window
#   point forecast      : conditional MEAN (set in code/models/foundation.py)
#   min-RV floor        : applied in code (replaces the old 1e-10 floor)
#
# One array task per ticker (each task loads its 5 models once and sweeps the
# ticker's 3 horizons). 50 tasks: IDs 0-39 stocks, 40-44 FX, 45-49 futures.
#
# Runtime: each fast model is one forward pass per rolling origin; ~5-15 min per
# ticker for all 5 models on one GPU. 12h wall time is generous headroom for the
# longer FX/futures series. --skip-existing makes the job fully restartable.
#
# Outputs: results/volare/forecasts/<model>_<ticker>_h<h>.csv  (volatility scale)
#
# Submit:
#     sbatch cluster/run_rev_tsfm_fast.slurm
# Resubmit after a partial run (skips finished CSVs):
#     sbatch cluster/run_rev_tsfm_fast.slurm
# Single ticker (debug, e.g. AAPL = task 0):
#     sbatch --array=0 cluster/run_rev_tsfm_fast.slurm
# =============================================================================
#SBATCH --job-name=rev-tsfm-fast
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/rev_tsfm_fast_%A_%a.out
#SBATCH --error=logs/rev_tsfm_fast_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs

# --- Environment ------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface   # shared HF model cache

cd "$SLURM_SUBMIT_DIR"

echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# --- Map array task ID -> (ticker, asset class) -----------------------------
STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)

ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};        CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};    CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi

echo "Ticker: ${TICKER}  (${CLASS})"

# --- Run --------------------------------------------------------------------
python code/run_foundation_volare.py \
    --tickers "${TICKER}" \
    --asset-class "${CLASS}" \
    --horizons 1 5 22 \
    --models chronos-bolt-small chronos-bolt-base timesfm-2.5 moirai-2.0-small ttm \
    --device cuda \
    --context-length 1000 \
    --target-kind point \
    --scale vol \
    --skip-existing

echo "Job finished: $(date)"
