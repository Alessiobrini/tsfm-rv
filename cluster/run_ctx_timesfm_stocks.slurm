#!/bin/bash
#SBATCH --job-name=ctx-tfm-stk
#SBATCH --partition=gpu-common
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctx_tfm_stk_%A_%a.out
#SBATCH --error=logs/ctx_tfm_stk_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

# Re-run TimesFM 2.5 context sensitivity after bugfix:
# context_length was not passed to get_foundation_model(), so TimesFM
# always compiled with max_context=512 regardless of --context-length.
# Fixed in run_foundation_volare.py (2026-03-14).

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 40 VOLARE stock tickers
TICKERS=(AAPL ADBE AMD AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GE GOOGL GS HD HON IBM JNJ JPM KO MCD META MMM MRK MSFT NFLX NKE NVDA ORCL PG PM SHW TRV TSLA UNH V VZ WMT XOM)
TICKER=${TICKERS[$SLURM_ARRAY_TASK_ID]}

echo "Running TimesFM 2.5 context-length sensitivity for ticker: ${TICKER}"

for CTX in 128 256; do
    echo "--- Context length: ${CTX} ---"
    python code/run_foundation_volare.py \
        --tickers ${TICKER} \
        --asset-class stocks \
        --horizons 1 5 22 \
        --models timesfm-2.5 \
        --device cuda \
        --context-length ${CTX}
done

echo "Job finished: $(date)"
