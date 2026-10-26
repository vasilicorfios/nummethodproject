"""Figures for the error analysis (:mod:`functions.error_analysis`).

* :func:`plot_tonal_by_angle`: peak prominence against chord Reynolds number
  for every setup.
* :func:`plot_tonal_by_strouhal_and_frequency`: peak prominence against the
  peak's Strouhal number and log10 frequency.

The style matches :mod:`functions.plot_results`; save with
:func:`functions.plot_tuning.save_png_pdf`.
"""

# Postponed annotations: the docs build mocks matplotlib.
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from functions.evaluate import ANGLE_BAND_LABELS
from functions.plot_results import INK_SECONDARY, SURFACE, _style

#: Ordinal blue ramp for the angle bands (light = small angle, dark = large),
#: with well-separated steps. The lightest still stands out from white.
ANGLE_BAND_COLOURS = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: A different marker per angle band, so bands can be told apart without
#: relying on shade alone (e.g. in greyscale print).
ANGLE_BAND_MARKERS = ["o", "s", "^", "D", "v"]


def _scatter_by_angle_band(ax, table: pd.DataFrame, x: str) -> None:
    """Plot ``prominence_db`` against column ``x``, one colour and marker per angle band."""
    for label, colour, marker in zip(ANGLE_BAND_LABELS, ANGLE_BAND_COLOURS, ANGLE_BAND_MARKERS):
        part = table[table["angle_band"] == label]
        ax.scatter(part[x], part["prominence_db"], s=48, color=colour, marker=marker,
                   edgecolors=SURFACE, linewidths=1, label=f"{label}° (n={len(part)})", zorder=3)


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
    _scatter_by_angle_band(ax, table, "reynolds")
    # Upper right is empty; the most tonal setups are at the upper left.
    ax.legend(title="Angle of attack", frameon=False, fontsize=8, title_fontsize=8,
              labelcolor=INK_SECONDARY, loc="upper right")
    return fig


def plot_tonal_by_strouhal_and_frequency(table: pd.DataFrame) -> Figure:
    """Peak prominence against the peak's Strouhal number and log10 frequency.

    Left: prominence against ``St = f_peak δ* / U`` (log axis). Right:
    prominence against ``log10(f_peak)``. Each dot is one setup, coloured and
    shaped by angle band as in :func:`plot_tonal_by_angle`. Setups with no
    interior peak have no peak frequency and are left out; the count is in
    the title.

    Args:
        table: Output of :func:`functions.error_analysis.tonal_table`.

    Returns:
        The figure (not shown or saved).
    """
    peaked = table[table["peak_hz"].notna()]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    fig.set_facecolor(SURFACE)
    for ax, x, xlabel in ((left, "strouhal_peak", "Strouhal number at the peak, f δ* / U (log scale)"),
                          (right, "log10_peak_hz", "log10(peak frequency / Hz)")):
        _style(ax)
        _scatter_by_angle_band(ax, peaked, x)
        ax.set_xlabel(xlabel)
    # Strouhal number spans about two decades, so it goes on a log axis, with
    # plain numbers on the ticks so values can be read off.
    left.set_xscale("log")
    lo, hi = peaked["strouhal_peak"].min() / 1.15, peaked["strouhal_peak"].max() * 1.15
    left.set_xlim(lo, hi)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if lo <= t <= hi]
    left.set_xticks(ticks, [f"{t:g}" for t in ticks])
    left.xaxis.set_minor_formatter(plt.NullFormatter())
    left.set_ylabel("Peak prominence (dB)")
    left.set_ylim(0, peaked["prominence_db"].max() * 1.08)
    right.legend(title="Angle of attack", frameon=False, fontsize=8, title_fontsize=8,
                 labelcolor=INK_SECONDARY, loc="upper right")
    left_out = len(table) - len(peaked)
    fig.suptitle(f"Peak prominence, {len(peaked)} setups"
                 + (f" ({left_out} without an interior peak left out)" if left_out else ""),
                 fontsize=10)
    return fig
