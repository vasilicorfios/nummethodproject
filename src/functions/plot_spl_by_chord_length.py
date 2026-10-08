"""Scatter plots of SPL against each feature, split by chord length."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from functions.data_loader import FEATURES, TARGET


def plot_spl_by_chord_length(df: pd.DataFrame) -> Figure:
    """Plot SPL against each non-chord feature, one row per chord length.

    Chord length takes only a handful of discrete values, so splitting by it
    reveals trends that are blurred in the pooled overview plot. The y-axis
    is shared so noise levels are comparable across rows.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data`.

    Returns:
        The created matplotlib figure (not yet shown).
    """
    chord_lengths = sorted(df["chord_length_m"].unique())
    other_features = [f for f in FEATURES if f != "chord_length_m"]

    fig, axes = plt.subplots(
        len(chord_lengths),
        len(other_features),
        figsize=(16, 3 * len(chord_lengths)),
        sharey=True,
    )

    for row, chord in enumerate(chord_lengths):
        subset = df[df["chord_length_m"] == chord]
        for col, feature in enumerate(other_features):
            ax = axes[row, col]
            ax.scatter(subset[feature], subset[TARGET], s=8, alpha=0.4)
            if feature == "frequency_hz":
                ax.set_xscale("log")
            # Label x only on the bottom row and y only on the left column
            # to keep the grid uncluttered.
            if row == len(chord_lengths) - 1:
                ax.set_xlabel(feature)
            if col == 0:
                ax.set_ylabel(f"chord={chord} m\n{TARGET}")

    fig.tight_layout()
    return fig
