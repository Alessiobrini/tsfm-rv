#!/bin/bash
#SBATCH --job-name=ctx-sens-miss
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/ctx_sens_miss_%j.out
#SBATCH --error=logs/ctx_sens_miss_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

source cluster/setup_models.sh

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

# Models that need gap-filling (Moirai already has full coverage)
MODELS="chronos-bolt-small chronos-bolt-base kronos"

for CTX in 128 256; do
    echo "=== Context length: ${CTX} ==="

    # Missing stocks: CAT, CRM, WMT, XOM
    echo "--- Missing stocks, ctx=${CTX} ---"
    python code/run_foundation_volare.py \
        --tickers CAT CRM WMT XOM \
        --asset-class stocks \
        --horizons 1 5 22 \
        --models ${MODELS} \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing

    # All FX
    echo "--- FX, ctx=${CTX} ---"
    python code/run_foundation_volare.py \
        --all-tickers \
        --asset-class fx \
        --horizons 1 5 22 \
        --models ${MODELS} \
        --device cuda \
        --context-length ${CTX} \
        --skip-existing

    # All futures
    echo "--- Futures, ctx=${CTX} ---"
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
