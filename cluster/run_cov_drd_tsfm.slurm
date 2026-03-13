#!/bin/bash
#SBATCH --job-name=cov-drd-tsfm
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/cov_drd_tsfm_%j.out
#SBATCH --error=logs/cov_drd_tsfm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Running DRD-TSFM covariance forecasting"

cd "$SLURM_SUBMIT_DIR"

MODELS="chronos-bolt-small chronos-bolt-base moirai-2.0-small toto sundial moirai-moe-small"

# Forex (5 assets, 15 pairs)
for MODEL in ${MODELS}; do
    echo "=== Forex DRD-TSFM: ${MODEL} ==="
    python code/run_cov_drd_tsfm.py \
        --asset-class forex \
        --horizons 1 5 22 \
        --models ${MODEL} \
        --device cuda
done

# Futures (5 contracts, 15 pairs)
for MODEL in ${MODELS}; do
    echo "=== Futures DRD-TSFM: ${MODEL} ==="
    python code/run_cov_drd_tsfm.py \
        --asset-class futures \
        --horizons 1 5 22 \
        --models ${MODEL} \
        --device cuda
done

# Stocks (40 assets, 820 pairs — much longer)
for MODEL in ${MODELS}; do
    echo "=== Stocks DRD-TSFM: ${MODEL} ==="
    python code/run_cov_drd_tsfm.py \
        --asset-class stocks \
        --horizons 1 5 22 \
        --models ${MODEL} \
        --device cuda
done

echo "Job finished: $(date)"
