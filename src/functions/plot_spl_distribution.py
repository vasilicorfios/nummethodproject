"""Histogram of the target, scaled sound pressure level (SPL)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from functions.data_loader import TARGET


def plot_spl_distribution(df: pd.DataFrame, bin_width_db: float = 1.0) -> Figure:
    """Plot the number of observations in each SPL bin.

    Bin edges are whole multiples of ``bin_width_db`` so bars line up with
    round dB values. Dashed and dotted lines mark the mean and median, so
    any skew in the target is visible at a glance.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data`.
        bin_width_db: Width of each histogram bin in dB. Must be positive.

    Returns:
        The created matplotlib figure (not yet shown).

    Raises:
        ValueError: If ``bin_width_db`` is not positive.
    """
    if bin_width_db <= 0:
        raise ValueError(f"bin_width_db must be positive, got {bin_width_db}")

    spl = df[TARGET]
    # Snap the outer edges outward to the bin grid so no value falls outside.
    lo = np.floor(spl.min() / bin_width_db) * bin_width_db
    hi = np.ceil(spl.max() / bin_width_db) * bin_width_db
    bins = np.arange(lo, hi + bin_width_db, bin_width_db)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(spl, bins=bins, edgecolor="white", linewidth=0.8)
    ax.axvline(spl.mean(), color="black", linestyle="--", linewidth=1.2,
               label=f"Mean {spl.mean():.1f} dB")
    ax.axvline(spl.median(), color="black", linestyle=":", linewidth=1.2,
               label=f"Median {spl.median():.1f} dB")
    ax.set_xlabel("Scaled sound pressure level (dB)")
    ax.set_ylabel("Number of observations")
    ax.set_title(f"Distribution of SPL (n = {len(spl)}, {bin_width_db:g} dB bins)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
