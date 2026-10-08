"""Exploratory scatter plots for the airfoil self-noise dataset.

Produces two figures:

1. Sound pressure level (SPL) against each input feature.
2. SPL against each remaining feature, with one row per chord length, to
   separate the effect of chord length from the other variables.

Run directly to display both figures::

    python src/explore_plots.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from data_loader import load_airfoil_data

#: Name of the target column (scaled sound pressure level, dB).
TARGET = "spl_db"

#: Input features plotted against the target in the overview figure.
FEATURES = [
    "frequency_hz",
    "attack_angle_deg",
    "chord_length_m",
    "velocity_mps",
    "disp_thickness_m",
]


def plot_spl_vs_features(df: pd.DataFrame) -> Figure:
    """Plot SPL against every input feature on a 2x3 grid.

    Frequency spans several decades, so its axis is log-scaled. The unused
    sixth panel is hidden.

    Args:
        df: Dataset as returned by
            :func:`data_loader.load_airfoil_data`.

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


def plot_spl_by_chord_length(df: pd.DataFrame) -> Figure:
    """Plot SPL against each non-chord feature, one row per chord length.

    Chord length takes only a handful of discrete values, so splitting by it
    reveals trends that are blurred in the pooled overview plot. The y-axis
    is shared so noise levels are comparable across rows.

    Args:
        df: Dataset as returned by
            :func:`data_loader.load_airfoil_data`.

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


def main() -> None:
    """Load the dataset and display both exploratory figures."""
    df = load_airfoil_data()
    plot_spl_vs_features(df)
    plot_spl_by_chord_length(df)
    plt.show()


if __name__ == "__main__":
    main()
