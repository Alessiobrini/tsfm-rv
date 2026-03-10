#!/bin/bash
#SBATCH --job-name=rv-chronos2-lora
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/chronos2_lora_%j.out
#SBATCH --error=logs/chronos2_lora_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

echo "Job started: $(date)"
cd "$SLURM_SUBMIT_DIR"

# -------------------------------------------------------------------------
# Chronos-2 LoRA Fine-Tuning on VOLARE Equity RV
# -------------------------------------------------------------------------
# Uses the chronos-forecasting training API with LoRA adapters.
# Fine-tunes chronos-bolt-small on all 40 equity RV series jointly,
# then runs walk-forward evaluation.
#
# References:
#   - Ansari et al. (2025) Chronos-2 paper
#   - https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training
# -------------------------------------------------------------------------

python code/run_finetune_chronos2.py \
    --model-id amazon/chronos-bolt-small \
    --all-tickers \
    --asset-class stocks \
    --horizons 1 5 22 \
    --device cuda \
    --lora-rank 8 \
    --max-epochs 5 \
    --learning-rate 1e-4 \
    --skip-existing

echo "Job finished: $(date)"
