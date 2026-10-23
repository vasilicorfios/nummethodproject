"""Figure comparing each model's cross-validation estimate with its hold-out RMSE.

The other hold-out figures reuse :mod:`functions.plot_results`. Colours
follow the rest of the project: linear blue, final MLP orange, and the
runner-up MLP the third series colour (:data:`HOLDOUT_COLOURS`).
"""

# Postponed annotations: the docs build mocks matplotlib.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from functions.models import FINAL_MODEL_NAME, RUNNER_UP_MODEL_NAME
from functions.plot_results import INK, INK_SECONDARY, SERIES_COLOURS, SURFACE, _style

#: Colour of each model in the hold-out figures.
HOLDOUT_COLOURS = {
    "linear": SERIES_COLOURS[0],
    FINAL_MODEL_NAME: SERIES_COLOURS[1],
    RUNNER_UP_MODEL_NAME: SERIES_COLOURS[2],
}


def plot_cv_vs_holdout(estimates: pd.DataFrame, metrics: pd.DataFrame) -> Figure:
    """CV mean ± fold SD (error bar) and hold-out RMSE (point) for each model.

    If a model generalised as cross-validation predicted, its hold-out point
    lies within (or near) its CV error bar.

    Args:
        estimates: Output of :func:`functions.final.cv_estimates`.
        metrics: Output of :func:`functions.final.holdout_metrics`.

    Returns:
        The figure (not shown or saved).
    """
    merged = estimates.merge(metrics[["model", "label", "rmse"]], on="model")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    fig.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(merged))
    for i, row in merged.iterrows():
        colour = HOLDOUT_COLOURS[row["model"]]
        # The CV estimate: a hollow marker with a ±1 SD bar, drawn just above the line.
        ax.errorbar(row["cv_rmse"], y[i] + 0.12, xerr=row["cv_sd"], fmt="o", mfc=SURFACE,
                    mec=colour, color=colour, markersize=7, capsize=4, linewidth=1.5,
                    label="CV estimate ± fold SD" if i == 0 else None)
        # The hold-out result: a filled marker just below.
        ax.plot(row["rmse"], y[i] - 0.12, "o", color=colour, markersize=8,
                label="hold-out RMSE" if i == 0 else None)
        ax.annotate(f"{row['rmse']:.2f}", (row["rmse"], y[i] - 0.12), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{label}\n({estimate})" for label, estimate in zip(merged["label"], merged["estimate"])])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (dB)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower right")
    return fig


def plot_holdout_residuals(predictions: pd.DataFrame, labels: dict[str, str]) -> Figure:
    """Residual (measured − predicted) against measured SPL on the hold-out.

    One panel per model, all on the same axes. Points above zero are rows
    where the model predicted too low; below zero, too high. A model with no
    systematic error would scatter evenly around zero at every SPL; a tilt in
    the cloud shows a trend, such as over-predicting quiet rows and
    under-predicting loud ones.

    Args:
        predictions: Contents of ``results/holdout_predictions.csv``: a
            ``measured_spl_db`` column and one ``pred_<model>`` column per model.
        labels: Model name -> panel title, in panel order.

    Returns:
        The figure (not shown or saved).
    """
    measured = predictions["measured_spl_db"]
    residuals = {m: measured - predictions[f"pred_{m}"] for m in labels}
    limit = max(r.abs().max() for r in residuals.values()) * 1.05

    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.2), sharex=True,
                             sharey=True, squeeze=False, layout="constrained")
    fig.set_facecolor(SURFACE)
    for ax, (model, label) in zip(axes[0], labels.items()):
        _style(ax)
        r = residuals[model]
        ax.axhline(0, color=INK_SECONDARY, linewidth=1)
        ax.scatter(measured, r, s=12, alpha=0.6, color=HOLDOUT_COLOURS[model], edgecolors="none")
        rmse = float(np.sqrt(np.mean(r ** 2)))
        ax.set_title(f"{label}   RMSE {rmse:.2f} dB, bias {r.mean():+.2f} dB",
                     fontsize=10, loc="left")
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("Measured SPL (dB)")
    axes[0, 0].set_ylabel("Residual, measured − predicted (dB)")
    return fig
