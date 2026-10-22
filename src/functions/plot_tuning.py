"""Figures for the nested-CV tuning of the MLP.

* :func:`plot_inner_grid_heatmap`: mean inner-CV RMSE for every setting in
  the grid, showing which layer sizes and alphas work and whether the best
  one sits at the edge of the grid.
* :func:`plot_loss_curves`: training loss and internal validation score per
  epoch for the chosen model in a few outer folds, showing where early
  stopping ended training.

Predicted-vs-measured and RMSE-breakdown comparisons reuse
:mod:`functions.plot_results` directly. :func:`save_png_pdf` writes every
figure as PNG (for the HTML docs) and PDF (vector, for the report). The
visual style matches :mod:`functions.plot_results`.
"""

# Postponed annotations: the docs build mocks matplotlib, and evaluating
# figure type hints on the mock at import time would fail.
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from functions.plot_results import INK, MUTED, SERIES_COLOURS, SURFACE, _new_figure, _style
from functions.tuning import ALPHA_KEY, LAYERS_KEY, TuningResult

#: Single-hue sequential ramp (light = low RMSE, dark = high RMSE).
RMSE_CMAP = LinearSegmentedColormap.from_list("rmse_blues", ["#cde2fb", "#2a78d6", "#0d366b"])

#: Colour of the tuned MLP when it is drawn on its own.
MLP_COLOUR = SERIES_COLOURS[1]


def save_png_pdf(fig: Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save a figure as ``<stem>.png`` and ``<stem>.pdf`` and close it.

    Args:
        fig: Figure to save.
        out_dir: Destination directory (created if missing).
        stem: File name without extension.

    Returns:
        The two paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    # Close so batch runs do not keep every figure in memory.
    plt.close(fig)
    return paths


def plot_inner_grid_heatmap(results: list[TuningResult]) -> Figure:
    """Heatmap of mean inner-CV RMSE over the grid, one panel per feature set.

    Each cell is the inner-CV RMSE of one setting, averaged over every outer
    fold and seed. The outlined cell has the lowest average. Rows are
    network sizes (smallest at the top) and columns are alpha values, so a
    best cell on the border suggests the grid should be extended.

    Args:
        results: Outputs of :func:`functions.tuning.nested_cv`.

    Returns:
        The figure (not shown or saved).
    """
    tables = []
    for r in results:
        layers = [str(h) for h in r.param_grid[LAYERS_KEY]]
        alphas = list(r.param_grid[ALPHA_KEY])
        table = (
            r.inner_grid.groupby(["hidden_layer_sizes", "alpha"])["mean_inner_rmse"].mean()
            .unstack("alpha").reindex(index=layers, columns=alphas)
        )
        tables.append(table)
    vmin = min(t.min().min() for t in tables)
    vmax = max(t.max().max() for t in tables)

    fig, axes = _new_figure(len(results), 4.6, 3.6)
    for ax, r, table in zip(axes, results, tables):
        values = table.to_numpy()
        image = ax.imshow(values, cmap=RMSE_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(values.shape[1]), [f"{a:g}" for a in table.columns])
        ax.set_yticks(range(values.shape[0]), table.index)
        ax.set_xlabel("alpha (L2 penalty)")
        ax.set_title(f"Feature set {r.feature_set}", fontsize=10, loc="left")
        # Write each value in the cell; white text on the darker cells.
        midpoint = (vmin + vmax) / 2
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > midpoint else INK)
        best_i, best_j = np.unravel_index(np.nanargmin(values), values.shape)
        ax.add_patch(Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                               fill=False, edgecolor=INK, linewidth=2))
    axes[0].set_ylabel("Hidden layer sizes")
    fig.colorbar(image, ax=list(axes), label="Mean inner-CV RMSE (dB)", shrink=0.9)
    return fig


def plot_loss_curves(result: TuningResult, keys: list[tuple[int, int]]) -> Figure:
    """Training loss and early-stopping score per epoch for chosen models.

    Top row: training loss (half the mean squared error on the
    *standardised* target, plus the L2 penalty). Bottom row: R² on the
    random 10% of rows that ``MLPRegressor`` holds back for early stopping.
    The dashed line marks the epoch with the highest validation score.
    Training stops once the score has failed to beat its best by more than
    ``tol`` for 20 epochs in a row; because tiny gains below ``tol`` do not
    count, the stop can come fewer than 20 epochs after the highest point.
    The final model keeps the weights from the best-scoring epoch.

    Args:
        result: Output of :func:`functions.tuning.nested_cv`.
        keys: ``(seed, fold)`` pairs to show (one column each).

    Returns:
        The figure (not shown or saved).
    """
    fig, axes = plt.subplots(2, len(keys), figsize=(4.0 * len(keys), 5.2),
                             sharex="col", squeeze=False, layout="constrained")
    fig.set_facecolor(SURFACE)
    for col, (seed, fold) in enumerate(keys):
        mlp = result.selected_models[(seed, fold)].regressor_.named_steps["regressor"]
        epochs = np.arange(1, len(mlp.loss_curve_) + 1)
        best_epoch = int(np.argmax(mlp.validation_scores_)) + 1
        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            _style(ax)
            ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=1)
        top.plot(epochs, mlp.loss_curve_, color=MLP_COLOUR, linewidth=2)
        bottom.plot(epochs, mlp.validation_scores_, color=MLP_COLOUR, linewidth=2)
        top.set_yscale("log")
        top.set_title(
            f"Seed {seed}, fold {fold}: {mlp.hidden_layer_sizes}, alpha={mlp.alpha:g}\n"
            f"best epoch {best_epoch}, stopped at {mlp.n_iter_}",
            fontsize=9, loc="left",
        )
        bottom.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training loss (scaled target)")
    axes[1, 0].set_ylabel("Early-stopping R²\n(random 10% of rows)")
    return fig
