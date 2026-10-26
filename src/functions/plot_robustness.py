"""Figure for the robustness shift tests.

:func:`plot_shift_tests`: RMSE for each held-out chord and velocity, linear
vs MLP, with each model's in-distribution (nested-CV) RMSE dashed.

Linear is always blue and the MLP orange, as in every other comparison.
Save with :func:`functions.plot_tuning.save_png_pdf`.
"""

# Postponed annotations: the docs build mocks matplotlib.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from functions.plot_results import INK, INK_SECONDARY, SERIES_COLOURS, SURFACE, _style

#: Colour and legend label of each model.
MODEL_STYLE = {
    "linear": (SERIES_COLOURS[0], "linear / B"),
    "mlp": (SERIES_COLOURS[1], "final MLP / B"),
}

#: Readable panel titles for the shift tests.
SHIFT_TITLES = {
    "leave-one-chord-out": "Leave one chord out (held-out chord, m)",
    "leave-one-velocity-out": "Leave one velocity out (held-out velocity, m/s)",
}


def plot_shift_tests(table: pd.DataFrame, reference: dict[str, float]) -> Figure:
    """Grouped bars of RMSE per held-out value, one panel per shift test.

    Error bars on the MLP bars are the SD over its random seeds. Dashed lines
    are each model's nested-CV RMSE, i.e. its accuracy when the chord and
    velocity were seen in training. "extrap." marks held-out values outside
    the training range.

    Args:
        table: Output of :func:`functions.shift_tests.run_shift_tests`.
        reference: Output of
            :func:`functions.shift_tests.in_distribution_reference`.

    Returns:
        The figure (not shown or saved).
    """
    shifts = list(dict.fromkeys(table["shift"]))
    fig, axes = plt.subplots(1, len(shifts), figsize=(6.2 * len(shifts), 4.2), sharey=True,
                             squeeze=False, layout="constrained")
    fig.set_facecolor(SURFACE)
    models = [m for m in MODEL_STYLE if m in set(table["model"])]
    width = 0.8 / len(models)

    for ax, shift in zip(axes[0], shifts):
        _style(ax)
        part = table[table["shift"] == shift]
        values = sorted(part["held_out"].unique())
        x = np.arange(len(values))
        for i, model in enumerate(models):
            colour, label = MODEL_STYLE[model]
            rows = part[part["model"] == model].set_index("held_out").loc[values]
            err = rows["rmse_sd_seeds"].fillna(0).to_numpy()
            ax.bar(x + (i - (len(models) - 1) / 2) * width, rows["rmse"], width=width, color=colour,
                   edgecolor=SURFACE, linewidth=1.5, label=label,
                   yerr=err if err.any() else None, capsize=3, error_kw={"ecolor": INK, "elinewidth": 1})
            ax.axhline(reference[model], color=colour, linestyle="--", linewidth=1.5,
                       label=f"{label}, nested CV ({reference[model]:.2f} dB)")
        regimes = part.drop_duplicates("held_out").set_index("held_out")["regime"]
        ax.set_xticks(x, [f"{v:g}\n(extrap.)" if regimes[v] == "extrapolation" else f"{v:g}"
                          for v in values])
        ax.grid(axis="x", visible=False)
        ax.set_title(SHIFT_TITLES.get(shift, shift), fontsize=10, loc="left")
    axes[0, 0].set_ylabel("RMSE on held-out rows (dB)")
    # The last panel has free space at the top (the tallest bar is on the left).
    axes[0, -1].legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper right")
    return fig
