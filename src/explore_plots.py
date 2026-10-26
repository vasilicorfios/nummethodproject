"""Entry point that displays the exploratory plots for the airfoil dataset.

The plotting functions live in their own modules
(:mod:`functions.plot_spl_distribution`, :mod:`functions.plot_spl_vs_features`,
:mod:`functions.plot_spl_by_chord_length`,
:mod:`functions.plot_spl_vs_physics_features` and
:mod:`functions.plot_spectra_grid`); this script loads the data,
adds the physics features, saves the figures used by the HTML docs to
``doc/generated/`` and then shows every figure::

    python src/explore_plots.py
    python src/explore_plots.py --no-show   # only write the PNGs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.error_analysis import tonal_table
from functions.physics_features import add_physics_features
from functions.plot_spl_by_chord_length import plot_spl_by_chord_length
from functions.plot_spl_distribution import plot_spl_distribution
from functions.plot_spl_vs_features import plot_spl_vs_features
from functions.plot_error_analysis import plot_tonal_by_angle, plot_tonal_by_strouhal_and_frequency
from functions.plot_spectra_grid import plot_spectra_grid
from functions.plot_spl_vs_physics_features import plot_spl_vs_physics_features
from functions.plot_tuning import save_png_pdf

#: Directory for figures included by the Sphinx data page
#: (``doc/exploration.rst``): ``<project root>/doc/generated``.
DOC_FIGURES_DIR = Path(__file__).resolve().parent.parent / "doc" / "generated"


def save_explore_figures(df: pd.DataFrame, out_dir: Path = DOC_FIGURES_DIR) -> list[Path]:
    """Draw the exploratory figures used in the docs and save them as PNGs.

    The per-chord-length grid is left out: it is very tall and is meant for
    interactive inspection rather than the HTML page.

    Args:
        df: Dataset with physics features, as returned by
            :func:`functions.physics_features.add_physics_features`.
        out_dir: Directory for the PNG files (created if missing).

    Returns:
        Paths of the files written: ``explore_spl_distribution.png``,
        ``explore_spl_vs_features.png``,
        ``explore_spl_vs_physics_features.png``, and
        ``explore_tonal_peaks.png`` / ``.pdf`` (peak prominence against
        Reynolds number by angle band, for every setup),
        ``explore_tonal_peaks_strouhal_frequency.png`` / ``.pdf`` (peak
        prominence against the peak's Strouhal number and log frequency) and
        ``explore_spectra_grid.png`` / ``.pdf`` (every measured spectrum, by
        chord and velocity).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "explore_spl_distribution.png": plot_spl_distribution(df),
        "explore_spl_vs_features.png": plot_spl_vs_features(df),
        "explore_spl_vs_physics_features.png": plot_spl_vs_physics_features(df),
    }
    written = []
    for name, fig in figures.items():
        path = out_dir / name
        fig.savefig(path, dpi=150)
        # Close each figure so it is not shown twice when main() re-draws.
        plt.close(fig)
        written.append(path)
    # Tonal peaks, one dot per setup. Saved as PDF too, for the report.
    tonal = tonal_table(df, include_holdout=True)
    written += save_png_pdf(plot_tonal_by_angle(tonal), out_dir, "explore_tonal_peaks")
    written += save_png_pdf(plot_tonal_by_strouhal_and_frequency(tonal), out_dir,
                            "explore_tonal_peaks_strouhal_frequency")
    written += save_png_pdf(plot_spectra_grid(df), out_dir, "explore_spectra_grid")
    return written


def main() -> None:
    """Save the docs figures, then display all exploratory figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true",
                        help="write the PNGs without opening plot windows")
    args = parser.parse_args()

    df = add_physics_features(load_airfoil_data())
    for path in save_explore_figures(df):
        print(f"Wrote {path}")
    if args.no_show:
        return

    plot_spl_distribution(df)
    plot_spl_vs_features(df)
    plot_spl_by_chord_length(df)
    plot_spl_vs_physics_features(df)
    plot_spectra_grid(df)
    plt.show()


if __name__ == "__main__":
    main()
