#!/bin/bash
# =============================================================================
# run_context_length.slurm — context-length runs for TTM and Log-HAR
# =============================================================================
# TTM r2 at 512, 1,024 and 1,536 days (the r2 release branches, pinned in
# config.TTM_R2_BRANCHES), and Log-HAR estimated on the 512 most recent days.
# Point target. Writes to separate folders so the stored forecasts are left
# untouched. One CPU array task per ticker (0-39 stocks, 40-44 FX, 45-49 futures).
# Submit:  sbatch cluster/run_context_length.slurm
# Outputs: results/ctx_ttm_r2/forecasts/, results/ctx_loghar512/forecasts/
# =============================================================================
#SBATCH --job-name=ctx-len
#SBATCH --partition=common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctx_len_%A_%a.out
#SBATCH --error=logs/ctx_len_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-/hpc/group/darec/ab978/.cache/huggingface}

STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)
ID=$SLURM_ARRAY_TASK_ID
if   [ "$ID" -lt 40 ]; then TICKER=${STOCK_TICKERS[$ID]};          CLASS=stocks
elif [ "$ID" -lt 45 ]; then TICKER=${FX_TICKERS[$((ID-40))]};      CLASS=fx
else                        TICKER=${FUTURES_TICKERS[$((ID-45))]}; CLASS=futures
fi
echo "Job started: $(date)  ${TICKER} (${CLASS})"

for CTX in 512 1024 1536; do
    python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
        --horizons 1 5 22 --models ttm-r2 --context-length "${CTX}" --device cpu \
        --target-kind point --scale vol --skip-existing --results-dir results/ctx_ttm_r2
done

python code/run_baselines_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models Log-HAR --train-window 512 --target-kind point --scale vol \
    --skip-existing --results-dir results/ctx_loghar512

echo "Job finished: $(date)"
