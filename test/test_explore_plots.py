"""Tests for :mod:`functions.plot_spl_distribution` and :mod:`explore_plots`."""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from explore_plots import save_explore_figures  # noqa: E402
from functions.plot_spl_distribution import plot_spl_distribution  # noqa: E402


def test_histogram_counts_every_row(raw_data):
    """The bar heights add up to the number of rows, so no value is dropped."""
    fig = plot_spl_distribution(raw_data)
    total = sum(patch.get_height() for patch in fig.axes[0].patches)
    plt.close(fig)
    assert total == len(raw_data)


def test_rejects_non_positive_bin_width(raw_data):
    """A zero bin width would give an empty or infinite bin grid."""
    with pytest.raises(ValueError):
        plot_spl_distribution(raw_data, bin_width_db=0)


def test_explore_figures_written_and_closed(data, tmp_path):
    """Every docs figure is saved (the tonal plot also as PDF) and no figure is left open."""
    before = len(plt.get_fignums())
    paths = save_explore_figures(data, tmp_path)
    assert {p.name for p in paths} == {
        "explore_spl_distribution.png",
        "explore_spl_vs_features.png",
        "explore_spl_vs_physics_features.png",
        "explore_tonal_peaks.png",
        "explore_tonal_peaks.pdf",
        "explore_tonal_peaks_strouhal_frequency.png",
        "explore_tonal_peaks_strouhal_frequency.pdf",
        "explore_spectra_grid.png",
        "explore_spectra_grid.pdf",
    }
    for path in paths:
        expected = b"\x89PNG\r\n\x1a\n" if path.suffix == ".png" else b"%PDF"
        assert path.read_bytes()[:len(expected)] == expected
    assert len(plt.get_fignums()) == before
