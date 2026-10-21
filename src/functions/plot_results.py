"""Figures of cross-validation results for the HTML results page.

Every function takes one or more :class:`functions.evaluate.CVResult`
objects and returns a matplotlib figure; :func:`save_result_figures` draws
them all and writes PNGs. Each result keeps the same colour in every figure
(first result blue, second orange, ...), so a model/feature set is
recognisable across the page. Measured data are drawn in neutral ink, never
a series colour.

Figures that show individual predictions use one CV seed
(:data:`functions.make_folds.CV_SEEDS` ``[0]`` by default) so that every row
appears exactly once.

Example:
    >>> from pathlib import Path
    >>> from functions.evaluate import cross_validate
    >>> from functions.plot_results import save_result_figures
    >>> results = [cross_validate(data, "linear", fs) for fs in ("A", "B")]
    >>> save_result_figures(data, results, Path("doc/generated"))  # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks matplotlib, and evaluating
# ``Figure | None`` on the mock at import time would fail.
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from functions.evaluate import (
    BREAKDOWN_COLUMNS,
    CVResult,
    coefficient_summary,
    oof_rmse_breakdown,
)
from functions.make_folds import CV_SEEDS, assign_folds

#: Series colours in fixed order (validated categorical palette, light mode).
#: Results beyond the third should be split across figures rather than
#: added to the same scatter, where more hues stop being distinguishable.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

#: Neutral chart colours: measured data, reference lines, grid and text.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

#: Readable labels for the breakdown columns.
BREAKDOWN_TITLES = {
    "chord_length_m": "Chord (m)",
    "velocity_mps": "Velocity (m/s)",
    "angle_band": "Angle of attack (deg)",
    "frequency_band": "Frequency band",
}


def _label(result: CVResult) -> str:
    """Legend label for a result, e.g. ``"linear / B"``."""
    return f"{result.model} / {result.feature_set}"


def _style(ax: Axes) -> None:
    """Apply the recessive axis style: hairline grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _frequency_axis(ax: Axes) -> None:
    """Log frequency axis labelled in plain Hz (200, 500, 1k, ...) not powers of ten."""
    ax.set_xscale("log")
    ticks = [200, 500, 1000, 2000, 5000, 10000, 20000]
    lo, hi = ax.get_xlim()
    shown = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(shown, [f"{t // 1000}k" if t >= 1000 else str(t) for t in shown])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def _new_figure(n_panels: int, width_per_panel: float, height: float, **kwargs):
    """Create a row of styled panels on the chart surface colour.

    Constrained layout keeps titles and labels inside the figure without
    per-figure margin tuning.
    """
    fig, axes = plt.subplots(
        1, n_panels, figsize=(width_per_panel * n_panels, height), squeeze=False,
        layout="constrained", **kwargs,
    )
    fig.set_facecolor(SURFACE)
    axes = axes[0]
    for ax in axes:
        _style(ax)
    return fig, axes


def _one_seed(result: CVResult, seed: int) -> pd.DataFrame:
    """OOF predictions for one seed, with residual = measured - predicted."""
    oof = result.oof[result.oof["seed"] == seed]
    return oof.assign(residual=oof["y_true"] - oof["y_pred"])


def plot_predicted_vs_measured(results: list[CVResult], seed: int = CV_SEEDS[0]) -> Figure:
    """Scatter OOF predicted SPL against measured SPL, one panel per result.

    Points on the dashed y = x line are perfect predictions. A cloud flatter
    than the line means the model shrinks extremes towards the mean.

    Args:
        results: Outputs of :func:`functions.evaluate.cross_validate`.
        seed: CV seed whose OOF predictions are shown.

    Returns:
        The figure (not shown or saved).
    """
    fig, axes = _new_figure(len(results), 4.6, 4.6, sharex=True, sharey=True)
    oofs = [_one_seed(r, seed) for r in results]
    lo = min(min(o["y_true"].min(), o["y_pred"].min()) for o in oofs) - 2
    hi = max(max(o["y_true"].max(), o["y_pred"].max()) for o in oofs) + 2

    for i, (ax, result, oof) in enumerate(zip(axes, results, oofs)):
        ax.plot([lo, hi], [lo, hi], color=MUTED, linestyle="--", linewidth=1)
        ax.scatter(oof["y_true"], oof["y_pred"], s=8, alpha=0.4,
                   color=SERIES_COLOURS[i], edgecolors="none")
        rmse = np.sqrt(np.mean(oof["residual"] ** 2))
        ax.set_title(f"{_label(result)}   RMSE {rmse:.2f} dB (seed {seed})", fontsize=10, loc="left")
        ax.set_xlabel("Measured SPL (dB)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted SPL (dB)")
    return fig


def plot_residual_by_frequency(results: list[CVResult], data: pd.DataFrame,
                               seed: int = CV_SEEDS[0]) -> Figure:
    """Plot the mean OOF residual at each measured frequency.

    An unbiased model would sit on the zero line at every frequency; a
    systematic curve shows spectral shape the model cannot represent.

    Args:
        results: Outputs of :func:`functions.evaluate.cross_validate`.
        data: The dataset the results were computed on (for frequencies).
        seed: CV seed whose OOF predictions are used.

    Returns:
        The figure (not shown or saved).
    """
    fig, (ax,) = _new_figure(1, 8, 4)
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, result in enumerate(results):
        oof = _one_seed(result, seed)
        freq = data.loc[oof["row"], "frequency_hz"].to_numpy()
        bias = oof.groupby(freq)["residual"].mean()
        ax.plot(bias.index, bias.to_numpy(), color=SERIES_COLOURS[i], linewidth=2,
                marker="o", markersize=5, label=_label(result))
    _frequency_axis(ax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean residual, measured - predicted (dB)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_rmse_breakdown(results: list[CVResult]) -> Figure:
    """Grouped bars of OOF RMSE for each subgroup, one panel per breakdown.

    Args:
        results: Outputs of :func:`functions.evaluate.cross_validate`.

    Returns:
        The figure (not shown or saved).
    """
    breakdowns = [oof_rmse_breakdown(r) for r in results]
    fig, axes = _new_figure(len(BREAKDOWN_COLUMNS), 3.4, 3.8, sharey=True)
    width = 0.8 / len(results)

    for ax, column in zip(axes, BREAKDOWN_COLUMNS):
        for i, (result, bd) in enumerate(zip(results, breakdowns)):
            rows = bd[bd["breakdown"] == column]
            x = np.arange(len(rows)) + (i - (len(results) - 1) / 2) * width
            # White edges give the 2px gap between adjacent bars.
            ax.bar(x, rows["rmse"], width=width, color=SERIES_COLOURS[i],
                   edgecolor=SURFACE, linewidth=1.5, label=_label(result))
        labels = rows["group"].astype(str).str.replace(" Hz", "").str.replace(" kHz", "k")
        ax.set_xticks(np.arange(len(rows)), labels, rotation=35, ha="right")
        ax.set_title(BREAKDOWN_TITLES[column], fontsize=10, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("OOF RMSE (dB)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_coefficients(results: list[CVResult]) -> Figure | None:
    """Dot-and-whisker plot of linear coefficients (mean ± std over fits).

    A whisker crossing zero means the coefficient's sign is not reliably
    estimated.

    Args:
        results: Outputs of :func:`functions.evaluate.cross_validate`.
            Results without linear coefficients are skipped.

    Returns:
        The figure, or ``None`` if no result has coefficients.
    """
    summaries = [(i, r, coefficient_summary(r)) for i, r in enumerate(results)]
    summaries = [s for s in summaries if not s[2].empty]
    if not summaries:
        return None

    # Union of features in first-seen order (raw inputs first).
    features = list(dict.fromkeys(f for _, _, s in summaries for f in s.index))
    fig, (ax,) = _new_figure(1, 7.5, 0.45 * len(features) + 1.2)
    ax.axvline(0, color=MUTED, linewidth=1)
    offset = 0.32 / max(len(summaries) - 1, 1)

    for k, (i, result, summary) in enumerate(summaries):
        y = np.array([features.index(f) for f in summary.index]) + (k - (len(summaries) - 1) / 2) * offset
        ax.errorbar(summary["effect_db_per_sd"], y, xerr=summary["std"], fmt="o",
                    color=SERIES_COLOURS[i], markersize=6, capsize=3, linewidth=1.5,
                    label=_label(result))
    ax.set_yticks(range(len(features)), features)
    ax.invert_yaxis()
    ax.set_xlabel("Effect on SPL (dB per input standard deviation)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def plot_example_spectra(results: list[CVResult], data: pd.DataFrame,
                         seed: int = CV_SEEDS[0]) -> Figure:
    """Measured and predicted spectra for the best, median and worst setups.

    Setups are ranked by the first result's per-setup OOF RMSE, so the
    three panels show what a good, typical and poor prediction looks like.

    Args:
        results: Outputs of :func:`functions.evaluate.cross_validate`.
        data: The dataset the results were computed on.
        seed: CV seed whose OOF predictions are shown.

    Returns:
        The figure (not shown or saved).
    """
    setup_of_row = assign_folds(data, cv_seed=seed)["setup_id"]
    first = _one_seed(results[0], seed)
    per_setup = (
        first.assign(setup=setup_of_row.loc[first["row"]].to_numpy())
        .groupby("setup")["residual"]
        .apply(lambda r: np.sqrt(np.mean(r ** 2)))
        .sort_values()
    )
    picks = {
        "Best": per_setup.index[0],
        "Median": per_setup.index[len(per_setup) // 2],
        "Worst": per_setup.index[-1],
    }

    fig, axes = _new_figure(3, 4.2, 3.8, sharey=True)
    for ax, (name, setup) in zip(axes, picks.items()):
        rows = setup_of_row.index[setup_of_row == setup]
        conditions = data.loc[rows].sort_values("frequency_hz")
        ax.plot(conditions["frequency_hz"], conditions["spl_db"], color=INK, linewidth=1.5,
                marker="o", markersize=4, label="Measured")
        for i, result in enumerate(results):
            oof = _one_seed(result, seed).set_index("row").loc[conditions.index]
            ax.plot(conditions["frequency_hz"], oof["y_pred"], color=SERIES_COLOURS[i],
                    linewidth=2, label=_label(result))
        c = conditions.iloc[0]
        ax.set_title(
            f"{name} ({_label(results[0])})\n"
            f"c = {c['chord_length_m']:.4f} m, U = {c['velocity_mps']:.1f} m/s, "
            f"α = {c['attack_angle_deg']:.1f}°",
            fontsize=9, loc="left",
        )
        _frequency_axis(ax)
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("SPL (dB)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return fig


def save_result_figures(data: pd.DataFrame, results: list[CVResult], out_dir: Path) -> list[Path]:
    """Draw every results figure and save it as a PNG.

    Args:
        data: The dataset the results were computed on.
        results: Outputs of :func:`functions.evaluate.cross_validate`, in the
            order their colours should be assigned.
        out_dir: Directory for the PNG files (created if missing).

    Returns:
        Paths of the files written. ``cv_coefficients.png`` is skipped when
        no result has linear coefficients.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "cv_predicted_vs_measured.png": plot_predicted_vs_measured(results),
        "cv_residual_by_frequency.png": plot_residual_by_frequency(results, data),
        "cv_rmse_breakdown.png": plot_rmse_breakdown(results),
        "cv_coefficients.png": plot_coefficients(results),
        "cv_example_spectra.png": plot_example_spectra(results, data),
    }
    written = []
    for name, fig in figures.items():
        if fig is None:
            continue
        path = out_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        # Close each figure so batch runs do not accumulate open windows.
        plt.close(fig)
        written.append(path)
    return written
