#!/bin/bash
# =============================================================================
# run_tsfm_logvol.slurm — zero-shot TSFMs on log volatility
# =============================================================================
# Each model forecasts log volatility, 0.5 * log(RV), from the same context as
# the main run (1,000 days, 512 for TTM), point target, h = 1, 5, 22. The files
# hold the log forecast and the model's 0.1 and 0.9 predictive quantiles,
# unclipped. Writes to results/tsfm_logvol/ so the stored forecasts are left
# untouched. One array task per ticker (0-39 stocks, 40-44 FX, 45-49 futures).
# Submit, choosing the models, device and partition:
#   sbatch --export=ALL,MODELS="ttm toto",DEVICE=cuda -p scavenger-gpu --gres=gpu:1 cluster/run_tsfm_logvol.slurm
#   sbatch --export=ALL,MODELS="moirai-moe-small",DEVICE=cpu -p common cluster/run_tsfm_logvol.slurm
# =============================================================================
#SBATCH --job-name=tsfm-logvol
#SBATCH --requeue
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tsfm_logvol_%A_%a.out
#SBATCH --error=logs/tsfm_logvol_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export HF_HOME=${HF_HOME:-/hpc/group/darec/ab978/.cache/huggingface}
: "${MODELS:?set MODELS}" "${DEVICE:=cuda}"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "Job started: $(date)  ${TICKER} (${CLASS}) on $(hostname)  models: ${MODELS}"

# shellcheck disable=SC2086
python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models ${MODELS} --context-length 1000 --device "${DEVICE}" \
    --target-kind point --scale logvol --skip-existing --results-dir results/tsfm_logvol

echo "Job finished: $(date)"
