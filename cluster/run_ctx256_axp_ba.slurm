#!/bin/bash
#SBATCH --job-name=ctx256-axp-ba
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ctx256_axp_ba_%j.out
#SBATCH --error=logs/ctx256_axp_ba_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

source cluster/setup_models.sh

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

MODELS="chronos-bolt-small chronos-bolt-base lag-llama"

echo "--- ctx256: AXP, BA ---"
python code/run_foundation_volare.py \
    --tickers AXP BA \
    --asset-class stocks \
    --horizons 1 5 22 \
    --models ${MODELS} \
    --device cuda \
    --context-length 256 \
    --skip-existing

echo "Job finished: $(date)"
