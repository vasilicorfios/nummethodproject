"""Figures for the error analysis (:mod:`functions.error_analysis`).

* :func:`plot_error_map`: every setup placed by its test conditions and
  coloured by its RMSE, so clusters of poorly predicted conditions stand out.
* :func:`plot_tonal_by_angle` and :func:`plot_tonal_by_error`: peak
  prominence against chord Reynolds number for every setup.

The style matches :mod:`functions.plot_results`; save with
:func:`functions.plot_tuning.save_png_pdf`.
"""

# Postponed annotations: the docs build mocks matplotlib.
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from functions.evaluate import ANGLE_BAND_LABELS
from functions.make_folds import TEST_FOLD, assign_folds
from functions.plot_results import INK, INK_SECONDARY, MUTED, SURFACE, _style
from functions.plot_tuning import RMSE_CMAP

def plot_error_map(tables: list[tuple[str, pd.DataFrame]], data: pd.DataFrame,
                   label_ranks: pd.DataFrame | None = None) -> Figure:
    """Per-setup RMSE placed on the test matrix, one row of panels per model.

    Each panel is one free-stream velocity: angle of attack across, chord
    up the side. Filled markers are cross-validation setups coloured by
    RMSE (one shared scale); hollow grey markers are held-out test setups,
    shown only so gaps in the test matrix are visible (they are not scored).

    Args:
        tables: ``(label, per_setup_rmse table)`` pairs, one row of panels each.
        data: The dataset (for the test matrix).
        label_ranks: Optional table with ``setup_id`` and ``rank``; those
            setups are labelled with their rank on the last row of panels.

    Returns:
        The figure (not shown or saved).
    """
    folded = assign_folds(data).drop_duplicates("setup_id")
    chords = sorted(folded["chord_length_m"].unique())
    velocities = sorted(folded["velocity_mps"].unique())
    test = folded[folded["fold"] == TEST_FOLD]
    vmin = min(t["rmse"].min() for _, t in tables)
    vmax = max(t["rmse"].max() for _, t in tables)

    fig, axes = plt.subplots(len(tables), len(velocities), figsize=(3.4 * len(velocities), 3.0 * len(tables)),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    fig.set_facecolor(SURFACE)
    for r, (label, table) in enumerate(tables):
        for c, velocity in enumerate(velocities):
            ax = axes[r, c]
            _style(ax)
            here = table[table["velocity_mps"] == velocity]
            y = here["chord_length_m"].map(chords.index)
            points = ax.scatter(here["attack_angle_deg"], y, c=here["rmse"], cmap=RMSE_CMAP,
                                vmin=vmin, vmax=vmax, s=70, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            t = test[test["velocity_mps"] == velocity]
            ax.scatter(t["attack_angle_deg"], t["chord_length_m"].map(chords.index), s=70,
                       facecolors="none", edgecolors=MUTED, linewidths=1, zorder=2)
            if label_ranks is not None and r == len(tables) - 1:
                ranked = label_ranks.merge(here[["setup_id", "attack_angle_deg", "chord_length_m"]], on="setup_id")
                for _, p in ranked.iterrows():
                    ax.annotate(str(int(p["rank"])), (p["attack_angle_deg"], chords.index(p["chord_length_m"])),
                                xytext=(5, 5), textcoords="offset points", fontsize=7.5, color=INK)
            ax.set_title(f"{label}, U = {velocity:g} m/s", fontsize=9, loc="left")
            ax.set_yticks(range(len(chords)), [f"{ch:g}" for ch in chords])
            ax.set_ylim(-0.6, len(chords) - 0.4)
    for ax in axes[-1]:
        ax.set_xlabel("Angle of attack (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Chord (m)")
    fig.colorbar(points, ax=list(axes.flat), label="Setup RMSE (dB)", shrink=0.8)
    return fig


#: Ordinal blue ramp for the angle bands (light = small angle, dark = large),
#: with well-separated steps. The lightest still stands out from white.
ANGLE_BAND_COLOURS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: A different marker per angle band, so bands can be told apart without
#: relying on shade alone (e.g. in greyscale print).
ANGLE_BAND_MARKERS = ["o", "s", "^", "D", "v"]


def _tonal_axes(ax, table: pd.DataFrame) -> None:
    """Shared look of the tonal plots: axes, labels and limits."""
    _style(ax)
    # Reynolds number spans more than a decade, so it goes on a log axis, with
    # the actual Reynolds numbers written on the ticks so they can be read off.
    ax.set_xscale("log")
    lo, hi = table["reynolds"].min() / 1.15, table["reynolds"].max() * 1.15
    ax.set_xlim(lo, hi)
    ticks = [t for t in (5e4, 1e5, 2e5, 5e5, 1e6, 2e6) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:,.0f}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Chord Reynolds number (log scale)")
    ax.set_ylabel("Peak prominence (dB)")
    ax.set_ylim(0, table["prominence_db"].max() * 1.08)


def plot_tonal_by_angle(table: pd.DataFrame) -> Figure:
    """Peak prominence against chord Reynolds number (log axis), coloured by angle band.

    Each dot is one setup. Prominence is how far the spectrum's biggest peak
    stands above its surroundings. Angles are grouped into the bands of
    :data:`functions.evaluate.ANGLE_BAND_LABELS` (0° on its own), because 27
    separate angles would be too many colours to tell apart.

    Args:
        table: Output of :func:`functions.error_analysis.tonal_table`.

    Returns:
        The figure (not shown or saved).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    fig.set_facecolor(SURFACE)
    _tonal_axes(ax, table)
    for label, colour, marker in zip(ANGLE_BAND_LABELS, ANGLE_BAND_COLOURS, ANGLE_BAND_MARKERS):
        part = table[table["angle_band"] == label]
        ax.scatter(part["reynolds"], part["prominence_db"], s=48, color=colour, marker=marker,
                   edgecolors=SURFACE, linewidths=1, label=f"{label}° (n={len(part)})", zorder=3)
    # Upper right is empty; the most tonal setups are at the upper left.
    ax.legend(title="Angle of attack", frameon=False, fontsize=8, title_fontsize=8,
              labelcolor=INK_SECONDARY, loc="upper right")
    return fig


def plot_tonal_by_error(table: pd.DataFrame, label: str) -> Figure:
    """Peak prominence against chord Reynolds number (log axis), coloured by model error.

    Same axes as :func:`plot_tonal_by_angle`, with each setup coloured by
    the model's out-of-fold RMSE for that setup (darker = larger error).

    Args:
        table: :func:`functions.error_analysis.tonal_table` output merged
            with a per-setup ``rmse`` column.
        label: Model name for the colour-bar label.

    Returns:
        The figure (not shown or saved).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    fig.set_facecolor(SURFACE)
    _tonal_axes(ax, table)
    # Draw the worst setups last so they are not hidden under others.
    ordered = table.sort_values("rmse")
    points = ax.scatter(ordered["reynolds"], ordered["prominence_db"], c=ordered["rmse"],
                        cmap=RMSE_CMAP, s=45, edgecolors=SURFACE, linewidths=1, zorder=3)
    fig.colorbar(points, ax=ax, label=f"{label} setup RMSE (dB)")
    return fig
