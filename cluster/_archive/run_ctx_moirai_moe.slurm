#!/bin/bash
#SBATCH --job-name=ctx-moe
#SBATCH --partition=gpu-common
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctx_moe_%A_%a.out
#SBATCH --error=logs/ctx_moe_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 50 tickers: 40 stocks + 5 FX + 5 futures
STOCK_TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
FX_TICKERS=(AUDUSD EURUSD GBPUSD USDCAD USDJPY)
FUTURES_TICKERS=(C CL ES GC NG)

ID=$SLURM_ARRAY_TASK_ID
if [ $ID -lt 40 ]; then
    TICKER=${STOCK_TICKERS[$ID]}
    CLASS=stocks
elif [ $ID -lt 45 ]; then
    TICKER=${FX_TICKERS[$((ID-40))]}
    CLASS=fx
else
    TICKER=${FUTURES_TICKERS[$((ID-45))]}
    CLASS=futures
fi

echo "Task $ID: $TICKER ($CLASS) — moirai-moe-small ctx=128,256"

for CTX in 128 256; do
    echo "--- Context length: ${CTX} ---"
    python code/run_foundation_volare.py \
        --tickers ${TICKER} \
        --asset-class ${CLASS} \
        --horizons 1 5 22 \
        --models moirai-moe-small \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing
done

echo "Job finished: $(date)"
