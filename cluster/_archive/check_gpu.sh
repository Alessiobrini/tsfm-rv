#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=check_gpu
#SBATCH --output=gpu_info_%j.out
#SBATCH --mail-user=alessio.brini@duke.edu
#SBATCH --mail-type=END

echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo ""
echo "=== Full nvidia-smi output ==="
nvidia-smi
echo ""
echo "=== CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
echo ""
echo "=== CPU Information ==="
lscpu | grep "Model name"
echo ""
echo "=== Node hostname ==="
hostname
