#!/bin/bash
#SBATCH --job-name=cov-tsfm-stocks
#SBATCH --partition=gpu-common
#SBATCH --array=0-81
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/cov_tsfm_stocks_%A_%a.out
#SBATCH --error=logs/cov_tsfm_stocks_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

cd "$SLURM_SUBMIT_DIR"

# 820 pairs, 82 array tasks -> 10 pairs per task
PAIRS_PER_TASK=10
PAIR_START=$((SLURM_ARRAY_TASK_ID * PAIRS_PER_TASK))
PAIR_END=$((PAIR_START + PAIRS_PER_TASK))

echo "Processing pairs ${PAIR_START} to ${PAIR_END}"

# Run Chronos-Bolt
python code/run_cov_foundation.py \
    --asset-class stocks \
    --horizons 1 5 22 \
    --models chronos-bolt-small \
    --device cuda \
    --pair-start ${PAIR_START} \
    --pair-end ${PAIR_END}

# Run Moirai
python code/run_cov_foundation.py \
    --asset-class stocks \
    --horizons 1 5 22 \
    --models moirai-2.0-small \
    --device cuda \
    --pair-start ${PAIR_START} \
    --pair-end ${PAIR_END}

echo "Job finished: $(date)"
