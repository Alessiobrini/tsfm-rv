#!/bin/bash
# =============================================================================
# run_tsfm_finetuned.slurm — forecasts of a fine-tuned TTM or Sundial
# =============================================================================
# The fine-tuned checkpoint (code/finetune/train.py) runs through the same
# runner, context and clipping as the zero-shot models, point target, h = 1, 5,
# 22. One array task per ticker (0-39 stocks, 40-44 FX, 45-49 futures):
#   sbatch --export=ALL,MODEL=ttm-ft,CHECKPOINT=<dir> -p scavenger-gpu --gres=gpu:1 cluster/run_tsfm_finetuned.slurm
# Outputs: results/tsfm_ft/forecasts/
# =============================================================================
#SBATCH --job-name=tsfm-ft
#SBATCH --requeue
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/tsfm_ft_%A_%a.out
#SBATCH --error=logs/tsfm_ft_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export HF_HOME=${HF_HOME:-/hpc/group/darec/ab978/.cache/huggingface}
: "${MODEL:?set MODEL}" "${CHECKPOINT:?set CHECKPOINT}" "${DEVICE:=cuda}"

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "Job started: $(date)  ${TICKER} (${CLASS}) on $(hostname)  model: ${MODEL}  checkpoint: ${CHECKPOINT}"

python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models "${MODEL}" --checkpoint "${CHECKPOINT}" --context-length 1000 \
    --device "${DEVICE}" --target-kind point --scale vol --skip-existing --results-dir results/tsfm_ft

echo "Job finished: $(date)"
