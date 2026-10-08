"""Overview scatter plot of SPL against every input feature."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from functions.data_loader import FEATURES, TARGET


def plot_spl_vs_features(df: pd.DataFrame) -> Figure:
    """Plot SPL against every input feature on a 2x3 grid.

    Frequency spans several decades, so its axis is log-scaled. The unused
    sixth panel is hidden.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data`.

    Returns:
        The created matplotlib figure (not yet shown).
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, feature in zip(axes.flat, FEATURES):
        ax.scatter(df[feature], df[TARGET], s=8, alpha=0.4)
        ax.set_xlabel(feature)
        ax.set_ylabel(TARGET)
        if feature == "frequency_hz":
            ax.set_xscale("log")

    # 5 features on a 6-panel grid: blank out the leftover panel.
    axes.flat[-1].axis("off")
    fig.tight_layout()
    return fig
