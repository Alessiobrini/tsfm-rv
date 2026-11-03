"""Shared style for the paper's figures: printed size, font, model names and model colors.

Every figure script imports this module, calls apply_style() before drawing, and sizes its
figure with figsize(), so that each PDF is drawn at the width it prints at and the text in
every figure has the same font and size as the document's captions.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

# set_size() lives in code/utils/figsize.py. It is loaded by path because code/utils.py, a
# module with the same name, would shadow a package import.
_spec = importlib.util.spec_from_file_location(
    "figsize", Path(__file__).resolve().parent / "utils" / "figsize.py")
_figsize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figsize)
set_size = _figsize.set_size

# \textwidth of paper/main.tex in points (article class, a4paper geometry), read from the log.
TEXTWIDTH_PT = 418.25368

# \small in a 10 pt article, the size of the captions.
FONT_SIZE = 9

# \footnotesize, only for the rotated model names under the box plot and the heatmap, where
# 15 names share about 125 pt and touch at FONT_SIZE.
DENSE_TICK_SIZE = 8

RC_PARAMS = {
    # Computer Modern from the fonts bundled with matplotlib, no LaTeX install needed.
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def apply_style():
    """Set the shared rcParams."""
    plt.rcParams.update(RC_PARAMS)


def figsize(fraction=1.0, subplots=(1, 1)):
    """Figure size in inches for a figure printed at `fraction` of the text width."""
    return set_size(TEXTWIDTH_PT, fraction, subplots)


def savefig(fig, path):
    """Save at the drawn size (no tight bounding box, which would change the width), without
    creation dates, so that two runs on the same inputs give the same file."""
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})


# Model names as they appear in the paper's tables (filename prefix -> paper name).
MODEL_DISPLAY = {
    "HAR": "HAR",
    "HAR_J": "HAR-J",
    "HAR_RS": "HAR-RS",
    "HARQ": "HARQ",
    "Log_HAR": "Log-HAR",
    "ARFIMA": "ARFIMA",
    "ARMA": "ARMA",
    "MEM": "MEM",
    "chronos_bolt_small": "Chronos-Bolt-S",
    "chronos_bolt_base": "Chronos-Bolt-B",
    "moirai_2_0_small": "Moirai-2.0-S",
    "lag_llama": "Lag-Llama",
    "timesfm_2_5": "TimesFM-2.5",
    "toto": "Toto",
    "sundial": "Sundial",
    "moirai_moe_small": "Moirai-MoE-S",
    "ttm": "TTM",
}

# Foundation models in the order of the paper's tables.
TSFM_ORDER = [
    "chronos_bolt_small", "chronos_bolt_base", "moirai_2_0_small", "moirai_moe_small",
    "lag_llama", "timesfm_2_5", "toto", "sundial", "ttm",
]

# One color per model, used by every figure. Econometric models in blues and greens, the
# foundation models in light warm tones and purples, and TTM in a strong red. Log-HAR, TTM and
# Sundial (the three lines of Fig. 1) pass the categorical palette checks for color-vision
# deficiency and contrast.
MODEL_COLORS = {
    "HAR": "#6baed6",
    "HAR_J": "#9ecae1",
    "HAR_RS": "#c6dbef",
    "HARQ": "#4292c6",
    "Log_HAR": "#08519c",
    "ARFIMA": "#74c476",
    "ARMA": "#a1d99b",
    "MEM": "#c7e9c0",
    "chronos_bolt_small": "#fdd0a2",
    "chronos_bolt_base": "#fdae6b",
    "moirai_2_0_small": "#dadaeb",
    "moirai_moe_small": "#bcbddc",
    "lag_llama": "#d9d9d9",
    "timesfm_2_5": "#e7cb94",
    "toto": "#f4cae4",
    "sundial": "#6a51a3",
    "ttm": "#d62728",
}

# Realized volatility drawn as data rather than as a model.
ACTUAL_COLOR = "0.6"
