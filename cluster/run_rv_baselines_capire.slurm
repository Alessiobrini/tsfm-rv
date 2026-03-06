#!/bin/bash
#SBATCH --job-name=rv-base-capire
#SBATCH --partition=gpu-common
#SBATCH --array=0-28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rv_base_capire_%A_%a.out
#SBATCH --error=logs/rv_base_capire_%A_%a.err
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

echo "Running baselines for ticker: ${TICKER}"

python code/run_baselines.py \
    --tickers ${TICKER} \
    --horizons 1 5 22 \
    --skip-existing

echo "Job finished: $(date)"
