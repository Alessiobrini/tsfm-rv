#!/bin/bash
# =============================================================================
# run_tsfm_mean_rerun.slurm — Chronos-Bolt and TimesFM 2.5 on their mean
# =============================================================================
# Reruns the two models whose point forecast is now the mean (the average of
# Chronos-Bolt's nine quantiles, TimesFM 2.5's mean output) for every stored
# configuration: point target at contexts 1,000, 128, 256 and 512 days, and the
# averaged target at 1,000 days. Volatility scale, h = 1, 5, 22. Writes to
# results/tsfm_mean_point/ and results/tsfm_mean_avg/. One array task per ticker.
# Submit:  sbatch cluster/run_tsfm_mean_rerun.slurm
# =============================================================================
#SBATCH --job-name=tsfm-mean
#SBATCH --partition=scavenger-gpu
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tsfm_mean_%A_%a.out
#SBATCH --error=logs/tsfm_mean_%A_%a.err

set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
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
MODELS="chronos-bolt-small chronos-bolt-base timesfm-2.5"

# shellcheck disable=SC2086
for CTX in 1000 128 256 512; do
    python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
        --horizons 1 5 22 --models ${MODELS} --context-length "${CTX}" --device cuda \
        --target-kind point --scale vol --skip-existing --results-dir results/tsfm_mean_point
done
python code/run_foundation_volare.py --tickers "${TICKER}" --asset-class "${CLASS}" \
    --horizons 1 5 22 --models ${MODELS} --context-length 1000 --device cuda \
    --target-kind avg --scale vol --skip-existing --results-dir results/tsfm_mean_avg

echo "Job finished: $(date)"
