"""Add \caption{} to all table files that are included in main.tex."""
import re
import os

TABLE_DIR = r'G:/Other computers/Dell Duke/Workfiles/Postdoc_file/human_x_AI_finance/paper/tables'

# Map label -> caption text
CAPTIONS = {
    'tab:main_results': 'Forecast accuracy for 40 U.S.\\ equities',
    'tab:main_results_median': 'Forecast accuracy for 40 U.S.\\ equities (medians)',
    'tab:mcs': 'Model Confidence Set inclusion rates for 40 equities',
    'tab:dm_summary': 'Diebold--Mariano pairwise win rates',
    'tab:fx_results': 'Forecast accuracy for 5 FX pairs',
    'tab:futures_results': 'Forecast accuracy for 5 futures contracts',
    'tab:portfolio': 'Global minimum variance portfolio performance',
    'tab:cov_accuracy': 'Element-wise covariance forecast accuracy',
    'tab:subsample': 'Sub-sample stability: pre- and post-COVID',
    'tab:window_512': 'Estimation window sensitivity: 252 vs.\\ 512 days',
    'tab:qlike_floor_sensitivity': 'QLIKE floor sensitivity analysis',
    'tab:mz_bias_corrected': 'Mincer--Zarnowitz bias-corrected evaluation',
    'tab:computational_cost': 'Computational cost per forecast step',
    'tab:mz': 'Mincer--Zarnowitz regression ($h = 1$)',
    'tab:mz_h5': 'Mincer--Zarnowitz regression ($h = 5$)',
    'tab:mz_h22': 'Mincer--Zarnowitz regression ($h = 22$)',
    'tab:per_asset_qlike': 'Per-asset QLIKE loss ($h = 1$)',
    'tab:per_asset_qlike_h5': 'Per-asset QLIKE loss ($h = 5$)',
    'tab:per_asset_qlike_h22': 'Per-asset QLIKE loss ($h = 22$)',
}

for fname in os.listdir(TABLE_DIR):
    if not fname.endswith('.tex'):
        continue
    fpath = os.path.join(TABLE_DIR, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already has \caption
    if r'\caption{' in content:
        print(f'SKIP (has caption): {fname}')
        continue

    # Find \label{...}
    label_match = re.search(r'\\label\{([^}]+)\}', content)
    if not label_match:
        print(f'SKIP (no label): {fname}')
        continue

    label = label_match.group(1)
    if label not in CAPTIONS:
        print(f'SKIP (no caption mapping): {fname} ({label})')
        continue

    caption_text = CAPTIONS[label]

    # Insert \caption before \label
    old = f'\\label{{{label}}}'
    new = f'\\caption{{{caption_text}}}\n\\label{{{label}}}'
    content = content.replace(old, new)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'ADDED caption to {fname}: "{caption_text}"')
