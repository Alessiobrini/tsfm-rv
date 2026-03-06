#!/bin/bash
#SBATCH --job-name=rv-base-vol-stk
#SBATCH --partition=gpu-common
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rv_base_vol_stk_%A_%a.out
#SBATCH --error=logs/rv_base_vol_stk_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 40 VOLARE stock tickers
TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
TICKER=${TICKERS[$SLURM_ARRAY_TASK_ID]}

echo "Running VOLARE baselines for ticker: ${TICKER}"

python code/run_baselines_volare.py \
    --tickers ${TICKER} \
    --asset-class stocks \
    --horizons 1 5 22 \
    --skip-existing

echo "Job finished: $(date)"
