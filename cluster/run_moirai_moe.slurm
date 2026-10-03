#!/bin/bash
# =============================================================================
# run_moirai_moe.slurm — Moirai-MoE with a 1,000-day context
# =============================================================================
# Runs Moirai-MoE zero-shot with the same 1,000-day context as the other
# foundation models, point and h-day-average targets. Writes to a separate
# folder so the stored forecasts are left untouched. One GPU array task per
# ticker (0-39 stocks, 40-44 FX, 45-49 futures), on the scavenger partition
# with requeue; finished files are skipped on a restart.
# Submit:  sbatch cluster/run_moirai_moe.slurm
# Outputs: results/moe_ctx1000/forecasts/, results/moe_ctx1000_avg/forecasts/
# =============================================================================
#SBATCH --job-name=moe-ctx1000
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/moe_ctx1000_%A_%a.out
#SBATCH --error=logs/moe_ctx1000_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-/hpc/group/darec/ab978/.cache/huggingface}

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "Job started: $(date)  ${TICKER} (${CLASS}) on $(hostname)"
nvidia-smi -L

python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models moirai-moe-small --context-length 1000 --device cuda \
    --target-kind point --scale vol --skip-existing --results-dir results/moe_ctx1000
python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models moirai-moe-small --context-length 1000 --device cuda \
    --target-kind avg --scale vol --skip-existing --results-dir results/moe_ctx1000_avg

echo "Job finished: $(date)"
