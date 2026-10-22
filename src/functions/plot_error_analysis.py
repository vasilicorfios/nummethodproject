"""Figures for the error analysis (:mod:`functions.error_analysis`).

* :func:`plot_error_map`: every setup placed by its test conditions and
  coloured by its RMSE, so clusters of poorly predicted conditions stand out.
* :func:`plot_peak_scaling`: the Strouhal-scaling check on measured peaks,
  and the model's predicted peak frequency against the measured one.
* :func:`plot_residual_heatmap`: residual (measured − predicted) for every
  setup and frequency, with the measured peaks marked.

The style matches :mod:`functions.plot_results`; save with
:func:`functions.plot_tuning.save_png_pdf`.
"""

# Postponed annotations: the docs build mocks matplotlib.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from functions.make_folds import TEST_FOLD, assign_folds
from functions.plot_results import INK, INK_SECONDARY, MUTED, SERIES_COLOURS, SURFACE, _style
from functions.plot_tuning import RMSE_CMAP

#: Diverging map for residuals: blue (model too high) -> neutral -> red
#: (model too low). The neutral midpoint means "no error".
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residual", ["#2a78d6", "#f0efec", "#e34948"])


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


def plot_peak_scaling(peaks: pd.DataFrame, fit: dict, model_label: str) -> Figure:
    """Strouhal-scaling check (left) and predicted vs measured peak frequency (right).

    Left: measured peak frequency against ``U / δ*`` on log axes for setups
    with a peak. Constant-Strouhal scaling would put the points on a line of
    slope 1 (dashed, drawn through the median Strouhal number); the solid
    line is the fitted slope. Right: the frequency of the model's predicted
    peak against the measured one; marker size counts setups (frequencies
    are discrete 1/3-octave values, so many coincide). Dotted lines mark one
    1/3-octave step either side of perfect agreement.

    Args:
        peaks: :func:`functions.error_analysis.spectral_peaks` output (with
            predicted peaks).
        fit: :func:`functions.error_analysis.peak_scaling_fit` output for
            ``disp_thickness_m``.
        model_label: Name of the model in the right panel.

    Returns:
        The figure (not shown or saved).
    """
    p = peaks[peaks["has_peak"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    fig.set_facecolor(SURFACE)
    for ax in (left, right):
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")

    x = p["velocity_mps"] / p["disp_thickness_m"]
    left.scatter(x, p["peak_hz"], s=28, color=INK, alpha=0.7, edgecolors="none")
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    left.plot(xs, 10 ** (fit["intercept"] + fit["slope"] * np.log10(xs)), color=SERIES_COLOURS[0],
              linewidth=2, label=f"fit: slope {fit['slope']:.2f} ± {fit['slope_ci95']:.2f}")
    left.plot(xs, fit["median_st"] * xs, color=MUTED, linestyle="--", linewidth=1.5,
              label=f"constant St = {fit['median_st']:.3f} (slope 1)")
    left.set_xlabel("U / δ*  (1/s)")
    left.set_ylabel("Measured peak frequency (Hz)")
    left.set_title(f"Strouhal scaling, {fit['n']} setups with a peak", fontsize=10, loc="left")
    left.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    counts = p.groupby(["peak_hz", "pred_peak_hz"]).size().reset_index(name="n")
    right.scatter(counts["peak_hz"], counts["pred_peak_hz"], s=25 * counts["n"],
                  color=SERIES_COLOURS[1], alpha=0.8, edgecolors=SURFACE, linewidths=1)
    lo, hi = 150, 25000
    right.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    step = 2 ** (1 / 3)  # one 1/3-octave step
    for factor in (step, 1 / step):
        right.plot([lo, hi], [lo * factor, hi * factor], color=MUTED, linestyle=":", linewidth=1)
    right.set_xlim(lo, hi)
    right.set_ylim(lo, hi)
    within = (p["pred_peak_steps_off"].abs() <= 1).mean()
    right.set_title(f"{model_label}: predicted peak within one step in {within:.0%}",
                    fontsize=10, loc="left")
    right.set_xlabel("Measured peak frequency (Hz)")
    right.set_ylabel("Predicted peak frequency (Hz)")
    return fig


def plot_residual_heatmap(matrices: list[tuple[str, pd.DataFrame]], data: pd.DataFrame,
                          peaks: pd.DataFrame) -> Figure:
    """Residual for every setup (rows) and frequency (columns), per model.

    Red means the model predicts too low, blue too high. Rows are grouped by
    chord (lines separate the groups) and ordered by angle then velocity
    within each chord. Black dots mark each setup's measured peak.

    Args:
        matrices: ``(label, residual_matrix)`` pairs, one panel each, with
            identical row order.
        data: The dataset (for the chord of each setup).
        peaks: :func:`functions.error_analysis.spectral_peaks` output.

    Returns:
        The figure (not shown or saved).
    """
    first = matrices[0][1]
    limit = float(np.nanpercentile(np.abs(np.concatenate([m.to_numpy().ravel() for _, m in matrices])), 95))
    chord_of = assign_folds(data).drop_duplicates("setup_id").set_index("setup_id")["chord_length_m"]
    row_chords = chord_of.reindex(first.index).to_numpy()
    freqs = list(first.columns)
    peak_at = peaks[peaks["has_peak"]].set_index("setup_id")["peak_hz"]

    fig, axes = plt.subplots(1, len(matrices), figsize=(4.6 * len(matrices) + 1.2, 11),
                             sharey=True, squeeze=False, layout="constrained")
    fig.set_facecolor(SURFACE)
    for ax, (label, matrix) in zip(axes[0], matrices):
        image = ax.imshow(matrix.to_numpy(), cmap=RESIDUAL_CMAP, vmin=-limit, vmax=limit,
                          aspect="auto", interpolation="nearest")
        ax.set_facecolor("#e1e0d9")  # unmeasured frequencies show as light grey
        rows = [i for i, s in enumerate(matrix.index) if s in peak_at.index]
        cols = [freqs.index(peak_at[matrix.index[i]]) for i in rows]
        ax.scatter(cols, rows, s=6, color=INK, zorder=3)
        # Lines between chord groups, and the chord written at each group's middle.
        changes = np.flatnonzero(np.diff(row_chords)) + 0.5
        for y in changes:
            ax.axhline(y, color=INK, linewidth=0.8)
        starts = np.concatenate([[-0.5], changes])
        ends = np.concatenate([changes, [len(row_chords) - 0.5]])
        ax.set_yticks((starts + ends) / 2, [f"c = {c:g} m" for c in dict.fromkeys(row_chords)])
        ticks = [i for i, f in enumerate(freqs) if f in (200, 500, 1000, 2000, 5000, 10000, 20000)]
        ax.set_xticks(ticks, [f"{freqs[i] / 1000:g}k" if freqs[i] >= 1000 else f"{freqs[i]:g}" for i in ticks])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(label, fontsize=10, loc="left", color=INK)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.colorbar(image, ax=list(axes[0]), shrink=0.6,
                 label="Residual, measured − predicted (dB)\nred: model too low, blue: too high")
    return fig
