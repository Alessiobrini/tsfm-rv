#!/bin/bash
# setup_new_models.sh — Install foundation-model dependencies at versions that
# coexist in ONE env. See requirements.txt for the binding constraints. Model
# packages are installed with --no-deps and pandas/transformers are pinned LAST,
# so a newer model release cannot drag the env to versions that break lag-llama
# (needs pandas < 2.2) or sundial (needs transformers < 4.46). This replaces the
# earlier unpinned `pip install` / `--upgrade` calls, which pulled pandas 3.x and
# transformers 4.57 and silently disabled lag-llama and sundial.

# TimesFM 2.0 (v2.5 API, GitHub only). No pandas dep; --no-deps is safe.
if ! python -c "import timesfm" 2>/dev/null; then
    echo "Installing timesfm 2.0 from GitHub..."
    pip install --no-deps "timesfm @ git+https://github.com/google-research/timesfm.git" --quiet || \
        echo "WARNING: timesfm install failed"
fi

# Toto. Its install pins (transformers==4.52.1, pandas==2.2.3) are conservative;
# it runs on the pinned set at runtime. --no-deps prevents the upgrade.
if ! python -c "import toto" 2>/dev/null; then
    echo "Installing toto-ts==0.2.0 (no-deps)..."
    pip install --no-deps "toto-ts==0.2.0" --quiet || echo "WARNING: toto-ts install failed"
fi

# IBM Granite TTM. 0.3.2+ import transformers>=4.57 symbols (check_torch_load_is_safe)
# and pin pandas>=2.3.3, which break the env; pin 0.3.1.
if ! python -c "import tsfm_public" 2>/dev/null; then
    echo "Installing granite-tsfm==0.3.1 (no-deps) for TTM..."
    pip install --no-deps "granite-tsfm==0.3.1" --quiet || echo "WARNING: granite-tsfm install failed"
fi

# Sundial: loads via HuggingFace trust_remote_code=True, no package needed.
echo "Sundial: uses HuggingFace remote code, no extra install needed."

# Moirai-MoE: ensure the module exists; --no-deps so it cannot bump shared deps.
if ! python -c "from uni2ts.model.moirai_moe import MoiraiMoEModule" 2>/dev/null; then
    echo "Installing uni2ts (no-deps) for Moirai-MoE..."
    pip install --no-deps "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git" --quiet || \
        echo "WARNING: uni2ts install for MoE failed"
fi

# Lock pandas/transformers to the one mutually-compatible point. MUST be last so
# nothing above overrides it. (Not --no-deps: transformers needs its matching
# tokenizers.) gluonts/lag-llama need pandas<2.2; sundial needs transformers<4.44
# (DynamicCache.seen_tokens AND standardize_cache_format); chronos needs >=4.41.
pip install "pandas==2.1.4" "transformers==4.43.4" --quiet || \
    echo "WARNING: pandas/transformers pin failed"

echo "New model dependencies ready."
