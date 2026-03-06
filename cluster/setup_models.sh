#!/bin/bash
# setup_models.sh — Ensure Lag-Llama and Kronos dependencies are available.
# Source this from SLURM scripts before running foundation model code.
# NOTE: no set -e — individual failures should not block the forecasting script
# (which already handles missing models gracefully via try/except).

# 1. Install lag-llama from GitHub if not already present
if ! python -c "import lag_llama" 2>/dev/null; then
    echo "Installing lag-llama from GitHub..."
    pip install "lag-llama @ git+https://github.com/time-series-foundation-models/lag-llama.git" --quiet || \
        echo "WARNING: lag-llama installation failed — model will be skipped"
fi

# 2. Clone Kronos vendor repo if not already present
VENDOR_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vendor/Kronos"
if [ ! -d "$VENDOR_DIR" ]; then
    echo "Cloning Kronos into $VENDOR_DIR..."
    mkdir -p "$(dirname "$VENDOR_DIR")"
    git clone https://github.com/NeoQuasar/Kronos.git "$VENDOR_DIR" || \
        echo "WARNING: Kronos clone failed — model will be skipped"
fi

echo "Model dependencies ready."
