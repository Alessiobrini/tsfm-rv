#!/bin/bash
# setup_models.sh — Ensure Lag-Llama and Kronos dependencies are available.
# Source this from SLURM scripts before running foundation model code.

set -e

# 1. Install lag-llama if not already present
if ! python -c "import lag_llama" 2>/dev/null; then
    echo "Installing lag-llama..."
    pip install lag-llama --quiet
fi

# 2. Clone Kronos vendor repo if not already present
VENDOR_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vendor/Kronos"
if [ ! -d "$VENDOR_DIR" ]; then
    echo "Cloning Kronos into $VENDOR_DIR..."
    mkdir -p "$(dirname "$VENDOR_DIR")"
    git clone https://github.com/NeoQuasar/Kronos.git "$VENDOR_DIR"
fi

echo "Model dependencies ready."
