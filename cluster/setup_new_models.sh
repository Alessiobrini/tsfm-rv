#!/bin/bash
# setup_new_models.sh — Install dependencies for Toto, Sundial, and Moirai-MoE.

# Install Toto
if ! python -c "import toto" 2>/dev/null; then
    echo "Installing toto-ts..."
    pip install toto-ts --quiet || echo "WARNING: toto-ts install failed"
fi

# Install Sundial (clone + pip install -e)
SUNDIAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vendor/Sundial"
if [ ! -d "$SUNDIAL_DIR" ]; then
    echo "Cloning Sundial..."
    git clone https://github.com/thuml/Sundial.git "$SUNDIAL_DIR"
    pip install -e "$SUNDIAL_DIR" --quiet || echo "WARNING: Sundial install failed"
fi

# Ensure uni2ts has moirai_moe module
if ! python -c "from uni2ts.model.moirai_moe import MoiraiMoEModule" 2>/dev/null; then
    echo "Upgrading uni2ts for Moirai-MoE support..."
    pip install --upgrade "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git" --quiet || \
        echo "WARNING: uni2ts upgrade for MoE failed"
fi

echo "New model dependencies ready."
