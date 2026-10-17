#!/bin/bash
#SBATCH --job-name=ctx-sens-small
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctx_sens_small_%j.out
#SBATCH --error=logs/ctx_sens_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

# Ensure Lag-Llama is available
source cluster/setup_models.sh

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

MODELS="chronos-bolt-small chronos-bolt-base moirai-2.0-small lag-llama"

for CTX in 128 256; do
    echo "=== Context length: ${CTX} ==="

    echo "Running FX (5 tickers), ctx=${CTX}"
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class fx \
        --horizons 1 5 22 \
        --models ${MODELS} \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing

    echo "Running futures (5 tickers), ctx=${CTX}"
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class futures \
        --horizons 1 5 22 \
        --models ${MODELS} \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing
done

echo "Job finished: $(date)"
