#!/bin/bash
#SBATCH --job-name=rv-tsfm-capire
#SBATCH --partition=gpu-common
#SBATCH --array=0-28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/rv_tsfm_capire_%A_%a.out
#SBATCH --error=logs/rv_tsfm_capire_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 29 CAPIRe (DJIA) tickers
TICKERS=(AAPL AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GS HD HON IBM INTC JNJ JPM KO MCD MMM MRK MSFT NKE PG TRV UNH V VZ WMT)
TICKER=${TICKERS[$SLURM_ARRAY_TASK_ID]}

echo "Running foundation models for ticker: ${TICKER}"

python code/run_foundation.py \
    --tickers ${TICKER} \
    --horizons 1 5 22 \
    --models chronos-bolt-small chronos-bolt-base moirai-2.0-small lag-llama kronos \
    --device cuda \
    --skip-existing

echo "Job finished: $(date)"
