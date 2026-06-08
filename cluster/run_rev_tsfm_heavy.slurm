#!/bin/bash
# =============================================================================
# run_rev_tsfm_heavy.slurm — REVISION pipeline, HEAVY TSFM group (GPU array job)
# =============================================================================
# Post-IJF revision. Zero-shot TSFM forecasts for the FOUR sampling-based models
# on all 50 VOLARE assets at horizons h = 1, 5, 22.
#
# Heavy group (draw num_samples paths per origin -> slower than the fast group):
#     sundial, toto, lag-llama, moirai-moe-small
#
# Same revised settings as the fast script:
#   --target-kind point  --scale vol  --context-length 1000  --device cuda
#   point forecast = conditional MEAN over samples; min-RV floor in code.
#
# Caveats specific to this group:
#   * lag-llama is installed from GitHub by cluster/setup_models.sh (sourced
#     below), matching the original foundation jobs.
#   * moirai-moe has a fixed 512-token context in the wrapper (architectural);
#     it ignores --context-length 1000 and is footnoted as an exception in the
#     paper. Passing 1000 is harmless.
#
# One array task per ticker (loads its 4 models once, sweeps 3 horizons).
# 50 tasks: IDs 0-39 stocks, 40-44 FX, 45-49 futures.
#
# Runtime: sampling makes these ~5-10x slower than the fast group; budget up to
# ~30-60 min per ticker. 24h wall time is generous headroom for long FX/futures
# series. --skip-existing makes the job fully restartable.
#
# Outputs: results/volare/forecasts/<model>_<ticker>_h<h>.csv  (volatility scale)
#
# Submit (after, or in parallel with, the fast group):
#     sbatch cluster/run_rev_tsfm_heavy.slurm
# =============================================================================
#SBATCH --job-name=rev-tsfm-heavy
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/rev_tsfm_heavy_%A_%a.out
#SBATCH --error=logs/rev_tsfm_heavy_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

set -euo pipefail
mkdir -p logs

# --- Environment ------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface

cd "$SLURM_SUBMIT_DIR"

# Ensure Lag-Llama is importable (installs from GitHub on first run).
source cluster/setup_models.sh

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
    --models sundial toto lag-llama moirai-moe-small \
    --device cuda \
    --context-length 1000 \
    --target-kind point \
    --scale vol \
    --skip-existing

echo "Job finished: $(date)"
