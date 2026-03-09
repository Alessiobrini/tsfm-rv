#!/bin/bash
#SBATCH --job-name=ctx-sens-stk
#SBATCH --partition=gpu-common
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctx_sens_stk_%A_%a.out
#SBATCH --error=logs/ctx_sens_stk_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

# Ensure Lag-Llama and Kronos are available
source cluster/setup_models.sh

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 40 VOLARE stock tickers
TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
TICKER=${TICKERS[$SLURM_ARRAY_TASK_ID]}

MODELS="chronos-bolt-small chronos-bolt-base moirai-2.0-small lag-llama kronos"

echo "Running context-length sensitivity for ticker: ${TICKER}"

for CTX in 128 256; do
    echo "--- Context length: ${CTX} ---"
    python code/run_foundation_volare.py \
        --tickers ${TICKER} \
        --asset-class stocks \
        --horizons 1 5 22 \
        --models ${MODELS} \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing
done

echo "Job finished: $(date)"
