"""Tests for :mod:`functions.plot_results`."""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import pytest  # noqa: E402

from functions.evaluate import cross_validate  # noqa: E402
from functions.plot_results import save_result_figures  # noqa: E402


@pytest.fixture(scope="module")
def results(data):
    """Linear baseline results on both feature sets."""
    return [cross_validate(data, "linear", fs) for fs in ("A", "B")]


def test_all_figures_written(data, results, tmp_path):
    """Every figure is saved as a non-empty PNG."""
    paths = save_result_figures(data, results, tmp_path)
    assert {p.name for p in paths} == {
        "cv_predicted_vs_measured.png",
        "cv_residual_by_frequency.png",
        "cv_rmse_breakdown.png",
        "cv_coefficients.png",
        "cv_example_spectra.png",
    }
    for path in paths:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_figures_left_open(data, results, tmp_path):
    """Figures are closed after saving so batch runs do not leak memory."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    save_result_figures(data, results, tmp_path)
    assert len(plt.get_fignums()) == before
