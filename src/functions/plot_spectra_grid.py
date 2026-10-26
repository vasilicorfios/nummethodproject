"""Grid of measured spectra: one panel per chord and velocity, one line per angle."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure

from functions.data_loader import TARGET
from functions.make_folds import SETUP_KEYS
from functions.plot_results import AXIS, GRID, INK, INK_SECONDARY, SURFACE

#: Sequential colour map for angle of attack (light = small angle).
ANGLE_CMAP = "Blues"

#: Start of the colour map actually used (0 = lightest).
CMAP_START = 0.25


def plot_spectra_grid(df: pd.DataFrame) -> Figure:
    """Plot every setup's measured spectrum, arranged by chord and velocity.

    Rows are chord lengths, columns are free-stream velocities, and each line
    is one setup (one angle of attack), coloured by angle. Axes are shared so
    panels can be compared directly.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data`.

    Returns:
        The created matplotlib figure (not yet shown).
    """
    chords = sorted(df["chord_length_m"].unique())
    velocities = sorted(df["velocity_mps"].unique())
    n_setups = df.groupby(SETUP_KEYS).ngroups

    # Drop the lightest part of the colour map so 0 deg stays visible; lines
    # and colour bar use this same trimmed map.
    cmap = ListedColormap(plt.get_cmap(ANGLE_CMAP)(np.linspace(CMAP_START, 1, 256)))
    norm = Normalize(df["attack_angle_deg"].min(), df["attack_angle_deg"].max())

    fig, axes = plt.subplots(len(chords), len(velocities), figsize=(12, 14),
                             sharex=True, sharey=True, facecolor=SURFACE)

    for row, chord in enumerate(chords):
        for col, velocity in enumerate(velocities):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            panel = df[(df["chord_length_m"] == chord) & (df["velocity_mps"] == velocity)]
            # groupby sorts by angle, so the darker, high-angle lines sit on top.
            for angle, spectrum in panel.groupby("attack_angle_deg"):
                spectrum = spectrum.sort_values("frequency_hz")
                ax.plot(spectrum["frequency_hz"], spectrum[TARGET], color=cmap(norm(angle)),
                        lw=1.2, marker="o", ms=2.5)
            ax.set_xscale("log")
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(AXIS)
            if row == 0:
                ax.set_title(f"U = {velocity:g} m/s", fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel(f"chord {chord * 1000:g} mm\nSPL (dB)", fontsize=9, color=INK)
            if row == len(chords) - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=9, color=INK)

    fig.suptitle(f"Scaled SPL against frequency for all {n_setups} setups "
                 "(one line = one angle of attack)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))

    bar_ax = fig.add_axes((0.92, 0.3, 0.015, 0.4))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_ax)
    bar.set_label("Angle of attack (deg)", fontsize=9, color=INK)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    bar.outline.set_edgecolor(AXIS)
    return fig

