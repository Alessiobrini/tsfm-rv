#!/bin/bash
#SBATCH --job-name=ctx-tfm-fxf
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctx_tfm_fxfut_%j.out
#SBATCH --error=logs/ctx_tfm_fxfut_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

# Re-run TimesFM 2.5 context sensitivity (FX + futures) after bugfix.
# See run_ctx_timesfm_stocks.slurm for details.

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

# Remove empty result files so --skip-existing will re-run them
FORECAST_DIR="results/volare/forecasts"
find "$FORECAST_DIR" -name "timesfm_2_5_*_ctx*.csv" -empty -delete 2>/dev/null
echo "Cleaned empty forecast files"

for CTX in 128 256; do
    echo "=== Context length: ${CTX} ==="

    echo "Running TimesFM 2.5 FX (5 tickers), ctx=${CTX}"
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class fx \
        --horizons 1 5 22 \
        --models timesfm-2.5 \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing

    echo "Running TimesFM 2.5 futures (5 tickers), ctx=${CTX}"
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class futures \
        --horizons 1 5 22 \
        --models timesfm-2.5 \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing
done

echo "Job finished: $(date)"
