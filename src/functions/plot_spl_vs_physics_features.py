"""Scatter plot of SPL against each physics-derived feature."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from functions.data_loader import TARGET
from functions.physics_features import PHYSICS_FEATURES


def plot_spl_vs_physics_features(df: pd.DataFrame) -> Figure:
    """Plot SPL against log-frequency, Strouhal number and Reynolds number.

    Strouhal and Reynolds numbers each span two to three decades, so their
    axes are log-scaled. Log-frequency is already logarithmic and is plotted
    on a linear axis.

    Args:
        df: Dataset with physics features, as returned by
            :func:`functions.physics_features.add_physics_features`.

    Returns:
        The created matplotlib figure (not yet shown).
    """
    fig, axes = plt.subplots(1, len(PHYSICS_FEATURES), figsize=(14, 4.5), sharey=True)

    for ax, feature in zip(axes, PHYSICS_FEATURES):
        ax.scatter(df[feature], df[TARGET], s=8, alpha=0.4)
        ax.set_xlabel(feature)
        if feature in ("strouhal", "reynolds"):
            ax.set_xscale("log")

    # y-axis is shared, so only the left panel needs a label.
    axes[0].set_ylabel(TARGET)
    fig.tight_layout()
    return fig
