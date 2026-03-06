#!/bin/bash
#SBATCH --job-name=cov-tsfm-small
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/cov_tsfm_small_%j.out
#SBATCH --error=logs/cov_tsfm_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Running TSFM covariance forecasting on forex + futures"

cd "$SLURM_SUBMIT_DIR"

# Forex (15 pairs)
for MODEL in chronos-bolt-small moirai-2.0-small; do
    echo "=== Forex: ${MODEL} ==="
    python code/run_cov_foundation.py \
        --asset-class forex \
        --horizons 1 5 22 \
        --models ${MODEL} \
        --device cuda
done

# Futures (15 pairs)
for MODEL in chronos-bolt-small moirai-2.0-small; do
    echo "=== Futures: ${MODEL} ==="
    python code/run_cov_foundation.py \
        --asset-class futures \
        --horizons 1 5 22 \
        --models ${MODEL} \
        --device cuda
done

echo "Job finished: $(date)"
