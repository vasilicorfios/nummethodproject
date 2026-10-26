"""Tests for :mod:`functions.error_analysis` and its figures.

Most tests use made-up "models" whose errors are known exactly: a perfect
model (prediction = measurement) and one that is wrong only at each
setup's peak. The answers can then be worked out by hand.
"""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from functions.error_analysis import (  # noqa: E402
    peak_error_share,
    peak_error_summary,
    spectral_peaks,
)
from functions.evaluate import CVResult, per_setup_rmse  # noqa: E402
from functions.make_folds import TEST_FOLD, assign_folds  # noqa: E402
from functions.plot_error_analysis import (  # noqa: E402
    plot_error_map,
)
from functions.plot_tuning import save_png_pdf  # noqa: E402


def _fake_result(data: pd.DataFrame, offsets: pd.Series | None = None) -> CVResult:
    """A 'model' predicting every CV row as measured, plus optional offsets (dB)."""
    folded = assign_folds(data)
    cv = folded[folded["fold"] != TEST_FOLD]
    y_pred = cv["spl_db"] + (offsets.reindex(cv.index).fillna(0.0) if offsets is not None else 0.0)
    oof = pd.DataFrame({"seed": 42, "fold": cv["fold"], "row": cv.index,
                        "y_true": cv["spl_db"].to_numpy(), "y_pred": y_pred.to_numpy(),
                        "chord_length_m": cv["chord_length_m"], "velocity_mps": cv["velocity_mps"]})
    return CVResult("fake", "B", pd.DataFrame(), oof.reset_index(drop=True), pd.DataFrame())


@pytest.fixture(scope="module")
def perfect(data):
    """A perfect model and the peak table it produces."""
    result = _fake_result(data)
    return result, spectral_peaks(data, result)


def test_peaks_only_for_cv_setups_and_interior(data, perfect):
    """One row per CV setup; peaks are interior points with the required prominence."""
    _, peaks = perfect
    folded = assign_folds(data)
    assert set(peaks["setup_id"]) == set(folded.loc[folded["fold"] != TEST_FOLD, "setup_id"])
    found = peaks[peaks["has_peak"]]
    assert (found["prominence_db"] >= 5.0).all()
    assert (found["peak_index"] > 0).all()  # never the first frequency
    assert found["st_peak"].to_numpy() == pytest.approx(
        (found["peak_hz"] * found["disp_thickness_m"] / found["velocity_mps"]).to_numpy())


def test_perfect_model_finds_every_peak(perfect):
    """A model equal to the measurements puts its peak exactly on the measured one."""
    _, peaks = perfect
    assert (peaks.loc[peaks["has_peak"], "pred_peak_steps_off"] == 0).all()


def test_peak_error_share_when_error_only_at_peak(data, perfect):
    """If the only error is at the peak frequency, all squared error is in the window."""
    _, peaks = perfect
    folded = assign_folds(data)
    offsets = {}
    for _, p in peaks[peaks["has_peak"]].iterrows():
        rows = folded[folded["setup_id"] == p["setup_id"]].sort_values("frequency_hz")
        offsets[rows.index[p["peak_index"]]] = 3.0
    shares = peak_error_share(_fake_result(data, pd.Series(offsets)), data, peaks)
    assert shares["error_share"].to_numpy() == pytest.approx(1.0)
    # Window = peak +/- 1 step = 3 frequencies (fewer only if the peak is one from the end).
    n_freq = folded[folded["fold"] != TEST_FOLD].groupby("setup_id").size()
    expected = 3 / n_freq.loc[shares["setup_id"]].to_numpy()
    assert (shares["frequency_share"].to_numpy() <= expected + 1e-12).all()
    # One 3 dB error over n frequencies gives RMSE 3 / sqrt(n).
    assert shares["rmse"].to_numpy() == pytest.approx(3 / np.sqrt(n_freq.loc[shares["setup_id"]].to_numpy()))


def test_peak_error_summary_hand_calculated():
    """Medians are taken over all peaked setups and over the strongly tonal ones."""
    shares = pd.DataFrame({"setup_id": [1, 2, 3], "error_share": [0.1, 0.2, 0.9],
                           "frequency_share": [0.2, 0.2, 0.25]})
    peaks = pd.DataFrame({"setup_id": [1, 2, 3], "prominence_db": [6.0, 8.0, 20.0]})
    out = peak_error_summary(shares, peaks)
    assert out["n"].tolist() == [3, 1]
    assert out["median_error_share"].tolist() == pytest.approx([0.2, 0.9])
    assert out["median_frequency_share"].tolist() == pytest.approx([0.2, 0.25])


def test_error_analysis_figures_saved(data, linear_b, tmp_path):
    """The error map saves as PNG and PDF."""
    setups = per_setup_rmse(linear_b, data)
    paths = save_png_pdf(plot_error_map([("linear", setups)], data,
                                        setups[["setup_id"]].head(3).assign(rank=[1, 2, 3])),
                         tmp_path, "map")
    assert len(paths) == 2 and all(p.stat().st_size > 0 for p in paths)


@pytest.fixture(scope="module")
def linear_b(data):
    """Linear baseline on feature set B (fast)."""
    from functions.evaluate import cross_validate
    return cross_validate(data, "linear", "B")


def test_tonal_table_counts_and_values(data):
    """One row per setup (106 with the hold-out, 90 without); valid prominences and bands."""
    from functions.error_analysis import tonal_table
    full = tonal_table(data, include_holdout=True)
    cv_only = tonal_table(data)
    assert len(full) == 106 and len(cv_only) == 90
    folded = assign_folds(data)
    holdout_setups = set(folded.loc[folded["fold"] == TEST_FOLD, "setup_id"])
    assert set(cv_only["setup_id"]).isdisjoint(holdout_setups)
    assert (full["prominence_db"] >= 0).all()
    assert full["angle_band"].notna().all()
    np.testing.assert_allclose(full["log10_reynolds"], np.log10(full["reynolds"]))


def test_tonal_table_hand_made_spectrum(data):
    """A flat spectrum with one 10 dB spike has a 10 dB-prominent peak at that frequency."""
    from functions.error_analysis import tonal_table
    folded = assign_folds(data)
    setup_id = folded["setup_id"].iat[0]
    setup_rows = folded.index[folded["setup_id"] == setup_id]
    fake = data.copy()
    ordered = fake.loc[setup_rows].sort_values("frequency_hz").index
    fake.loc[ordered, "spl_db"] = 120.0
    fake.loc[ordered[3], "spl_db"] = 130.0
    row = tonal_table(fake, include_holdout=True).set_index("setup_id").loc[setup_id]
    assert row["prominence_db"] == pytest.approx(10.0)
    assert row["peak_hz"] == fake.loc[ordered[3], "frequency_hz"]


def test_tonal_figures_saved(data, tmp_path):
    """The angle-coloured and error-coloured tonal plots save as PNG and PDF."""
    from functions.error_analysis import tonal_table
    from functions.plot_error_analysis import plot_tonal_by_angle, plot_tonal_by_error
    table = tonal_table(data, include_holdout=True)
    paths = save_png_pdf(plot_tonal_by_angle(table), tmp_path, "by_angle")
    paths += save_png_pdf(plot_tonal_by_error(table.assign(rmse=table["prominence_db"] / 5), "test"),
                          tmp_path, "by_error")
    assert len(paths) == 4 and all(p.stat().st_size > 0 for p in paths)
