"""
visualization/plots.py — All paper figures and diagnostic plots.

Functions for:
    - Time series plots of RV
    - Forecast comparison plots (actual vs predicted)
    - ACF plots for model residuals
    - Loss function comparison bar charts
    - DM test heatmaps
    - MCS result visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path

from config import FIGURES_DIR, PLOTS_DIR


def plot_actual_vs_forecast(
    actual: pd.Series,
    forecasts: Dict[str, pd.Series],
    ticker: str,
    horizon: int,
    save_path: Optional[Path] = None,
) -> None:
    """Plot actual RV against model forecasts.

    Parameters
    ----------
    actual : pd.Series
        Realized values.
    forecasts : Dict[str, pd.Series]
        Model name -> forecast series.
    ticker : str
        Asset ticker for title.
    horizon : int
        Forecast horizon.
    save_path : Path, optional
        File path to save figure. If None, saves to FIGURES_DIR.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual.index, actual.values, 'k-', linewidth=0.8, label='Actual RV', alpha=0.7)

    colors = plt.cm.Set2(np.linspace(0, 1, len(forecasts)))
    for (name, fcast), color in zip(forecasts.items(), colors):
        ax.plot(fcast.index, fcast.values, linewidth=0.6, label=name, color=color, alpha=0.8)

    ax.set_title(f'{ticker} — Actual vs Forecast RV (h={horizon})', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Realized Variance')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURES_DIR / f'forecast_{ticker}_h{horizon}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_comparison(
    results: pd.DataFrame,
    loss_type: str = "QLIKE",
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing model losses.

    Parameters
    ----------
    results : pd.DataFrame
        Rows = models, columns include loss_type.
    loss_type : str
        Which loss to plot.
    save_path : Path, optional
        File path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    results_sorted = results.sort_values(loss_type)
    ax.barh(results_sorted.index, results_sorted[loss_type], color='steelblue', alpha=0.8)
    ax.set_xlabel(loss_type)
    ax.set_title(f'Model Comparison — {loss_type}')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURES_DIR / f'loss_comparison_{loss_type}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_dm_heatmap(
    dm_pvalues: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Heatmap of pairwise DM test p-values.

    Parameters
    ----------
    dm_pvalues : pd.DataFrame
        Square matrix of p-values from dm_test_matrix().
    save_path : Path, optional
        File path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        dm_pvalues, annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0, vmax=0.1, ax=ax, square=True,
    )
    ax.set_title('Diebold-Mariano Test p-values (pairwise)')

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURES_DIR / 'dm_test_heatmap.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
