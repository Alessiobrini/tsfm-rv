#!/bin/bash
# setup_new_models.sh — Install dependencies for new TSFMs.

# Upgrade TimesFM to v2.0.0 (v2.5 API, only on GitHub)
TIMESFM_VER=$(python -c "import timesfm; print(timesfm.__version__)" 2>/dev/null)
if [ "$TIMESFM_VER" != "2.0.0" ]; then
    echo "Upgrading timesfm to 2.0.0 from GitHub..."
    pip install "timesfm @ git+https://github.com/google-research/timesfm.git" --quiet || \
        echo "WARNING: timesfm upgrade failed"
fi

# Install Toto
if ! python -c "import toto" 2>/dev/null; then
    echo "Installing toto-ts..."
    pip install toto-ts --quiet || echo "WARNING: toto-ts install failed"
fi

# Sundial: no install needed — loads via HuggingFace trust_remote_code=True.
# Just ensure transformers is installed (already a dependency of other models).
echo "Sundial: uses HuggingFace remote code, no extra install needed."

# Ensure uni2ts has moirai_moe module
if ! python -c "from uni2ts.model.moirai_moe import MoiraiMoEModule" 2>/dev/null; then
    echo "Upgrading uni2ts for Moirai-MoE support..."
    pip install --upgrade "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git" --quiet || \
        echo "WARNING: uni2ts upgrade for MoE failed"
fi

echo "New model dependencies ready."
