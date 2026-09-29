#!/bin/bash
#SBATCH --job-name=ttm-all
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ttm_all_%j.out
#SBATCH --error=logs/ttm_all_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

# Run TTM (Tiny Time Mixers) r2.1 zero-shot on all 50 assets.
# TTM is very fast (~5ms/step) so all assets fit in one job.

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai
export HF_HOME=/hpc/group/darec/ab978/.cache/huggingface

echo "Job started: $(date)"
cd "$SLURM_SUBMIT_DIR"

# Default context (512)
for AC in stocks fx futures; do
    echo "=== Asset class: ${AC}, ctx=512 ==="
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class ${AC} \
        --horizons 1 5 22 \
        --models ttm \
        --device cuda \
        --skip-existing
done

# Context sensitivity (128, 256)
for CTX in 128 256; do
    for AC in stocks fx futures; do
        echo "=== Asset class: ${AC}, ctx=${CTX} ==="
        python code/run_foundation_volare.py \
            --all-tickers \
            --asset-class ${AC} \
            --horizons 1 5 22 \
            --models ttm \
            --device cuda \
            --context-length ${CTX} \
            --skip-existing
    done
done

echo "Job finished: $(date)"
