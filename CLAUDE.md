# CLAUDE.md — Realized Volatility Forecasting with Time Series Foundation Models

## Project Overview
This paper investigates whether time series foundation models (TSFMs) can outperform established econometric models in forecasting realized volatility. We benchmark multiple state-of-the-art TSFMs — Chronos-2, TimesFM 2.5, Moirai 2.0, and Chronos-Bolt — against standard econometric models (HAR, Realized GARCH, Realized EGARCH, ARFIMA) using the VOLARE open-source dataset. Unlike prior work (e.g., arxiv 2505.11163) which tested only TimesFM, this study provides a comprehensive multi-model horse race across the leading foundation models, evaluating both zero-shot and fine-tuned performance.

## Research Question
Can time series foundation models, applied zero-shot or with minimal fine-tuning, match or exceed traditional econometric benchmarks (HAR, Realized GARCH, ARFIMA) in forecasting daily realized volatility across multiple asset classes?

## Key Contribution
- First comprehensive multi-TSFM comparison for realized volatility forecasting (Chronos-2, TimesFM 2.5, Moirai 2.0, Chronos-Bolt)
- Uses VOLARE, a new open-source realized volatility dataset with methodological consistency across assets
- Evaluates zero-shot vs. fine-tuned TSFM performance against econometric benchmarks
- Tests across multiple forecast horizons (1-day, 5-day, 22-day) and multiple assets
- Provides practical guidance on when TSFMs add value over econometric models for practitioners

## Methodology

### Econometric Benchmarks
1. **HAR** (Heterogeneous Autoregressive Model) — Corsi (2009): RV_t = β₀ + β₁·RV_{t-1} + β₂·RV_{t-5,t-1} + β₃·RV_{t-22,t-1} + ε_t
2. **Realized GARCH** — Hansen, Huang & Shek (2012): Incorporates high-frequency data into GARCH framework
3. **Realized EGARCH** — Hansen & Huang (2016): Adds leverage effects via log transformation
4. **ARFIMA** — Captures long memory in volatility series

### Time Series Foundation Models
1. **Chronos-2** (Amazon, Oct 2025): Encoder-only, direct quantile, supports univariate/multivariate/covariates
2. **TimesFM 2.5** (Google): Decoder-only, 200M params, autoregressive quantile prediction
3. **Moirai 2.0** (Salesforce): Decoder-only, quantile forecasting, multi-token prediction
4. **Chronos-Bolt** (Amazon): Distilled variant, 250x faster, competitive accuracy

### Evaluation Strategy
- **Expanding window** estimation with rolling one-step-ahead forecasts
- **Forecast horizons**: h = 1, 5, 22 days
- **Loss functions**: MSE, MAE, QLIKE
- **Statistical tests**: Diebold-Mariano test for pairwise forecast comparison, Model Confidence Set (MCS) by Hansen et al. (2011)
- **Zero-shot evaluation**: TSFMs applied directly without any training on the target series
- **Fine-tuned evaluation**: TSFMs with incremental learning on realized volatility data

## Data Source
- **VOLARE** (VOLatility Archive for Realized Estimates)
  - Paper: arxiv 2602.19732
  - Provides: realized variance, bipower variation, semivariances, realized quarticity, realized kernels
  - Multi-asset coverage with methodological consistency
  - Built from ultra high-frequency data (Kibot)
  - Download the bulk dataset; focus on major liquid assets (e.g., S&P 500 constituents, major indices, FX)

## Timeline
- **Deadline**: March 18, 2026 (submissions open Feb 18)
- Work fast. Claude Code handles the execution. Focus on research design and interpretation.

## Output Format
- **LaTeX paper** compiled to PDF
- Academic finance style: Introduction, Literature Review, Data, Methodology, Results, Robustness Checks, Conclusion
- No length constraint (per conference rules)
- Include well-labeled tables and figures throughout
- BibTeX for references

## Style Guidelines
- Formal academic tone, precise and direct
- **AVOID** AI-typical language: "delve", "crucial", "landscape", "in the realm of", "it is important to note", "comprehensive", "leveraging", "robust" (unless statistically precise), "novel" (use "new" or be specific)
- **PREFER** direct statements, specific claims backed by data, varied sentence structure
- Every claim must reference data, a citation, or a table/figure
- Write for an audience of AI review agents: prioritize logical structure, explicit argumentation, clear section transitions, and self-contained abstracts
- Abstract should contain: motivation (1 sentence), gap (1 sentence), what we do (2 sentences), key findings (2-3 sentences), implication (1 sentence)

## Tools & Packages
- **Python**: pandas, numpy, statsmodels, arch (for GARCH models), scikit-learn, matplotlib, seaborn
- **Foundation models**: chronos-forecasting (Amazon), timesfm (Google), uni2ts (Salesforce Moirai)
- **LaTeX**: biblatex, booktabs, graphicx, amsmath
- **Statistical testing**: Custom implementation or existing packages for DM test and MCS

## Rules
1. Always update WORKFLOW.md when completing a significant task
2. Commit to git after each phase
3. Keep all code reproducible with clear comments and random seeds
4. Save all intermediate results (forecasts, metrics) to CSV for reproducibility
5. When in doubt about methodology, prefer the approach most common in top finance journals (JF, JFE, RFS)
6. Log every AI tool/model interaction in the workflow description
7. Always gather BibTeX entries from Google Scholar — do not use any other source for reference metadata

## Key References to Cite
- Corsi (2009) — HAR model
- Hansen, Huang & Shek (2012) — Realized GARCH
- Andersen, Bollerslev, Diebold & Labys (2003) — Realized volatility foundations
- Ansari et al. (2024, 2025) — Chronos, Chronos-2
- Das et al. (2024) — TimesFM
- Woo et al. (2024) — Moirai
- The VOLARE paper (arxiv 2602.19732)
- The TimesFM volatility paper (arxiv 2505.11163) — key paper to differentiate from
