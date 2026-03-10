#!/bin/bash
#SBATCH --job-name=rv-lagllama-ft
#SBATCH --partition=gpu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/lagllama_ft_%j.out
#SBATCH --error=logs/lagllama_ft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessio.brini@duke.edu

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate human-x-ai

# Ensure Lag-Llama is available
source cluster/setup_models.sh

echo "Job started: $(date)"

cd "$SLURM_SUBMIT_DIR"

# Fine-tune Lag-Llama on equity RV (40 stocks, h=1)
# max_epochs=10, retrain every 126 steps, 20 samples
echo "Running Lag-Llama fine-tuning: equities h=1"
python code/run_finetune_lagllama.py \
    --all-tickers \
    --asset-class stocks \
    --horizons 1 \
    --device cuda \
    --max-epochs 10 \
    --retrain-every 126 \
    --num-samples 20 \
    --skip-existing

echo "Running Lag-Llama fine-tuning: equities h=5"
python code/run_finetune_lagllama.py \
    --all-tickers \
    --asset-class stocks \
    --horizons 5 \
    --device cuda \
    --max-epochs 10 \
    --retrain-every 126 \
    --num-samples 20 \
    --skip-existing

echo "Running Lag-Llama fine-tuning: equities h=22"
python code/run_finetune_lagllama.py \
    --all-tickers \
    --asset-class stocks \
    --horizons 22 \
    --device cuda \
    --max-epochs 10 \
    --retrain-every 126 \
    --num-samples 20 \
    --skip-existing

echo "Job finished: $(date)"
