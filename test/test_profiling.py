"""Tests for :mod:`functions.profiling`."""

import pytest

from functions.make_folds import TEST_FOLD, assign_folds
from functions.profiling import linear_flop_estimate, profile_model


def test_flop_estimate_hand_calculated():
    """n=100, p=2: fit 2*100*4 + 11*8 + 7*200 + 5*100; predict 4*200 + 200."""
    est = linear_flop_estimate(100, 2)
    assert est["fit_flops"] == pytest.approx(2788)
    assert est["predict_flops"] == pytest.approx(1000)


def test_flop_estimate_scales_with_n_p_squared():
    """For n >> p the 2np^2 term dominates, so doubling n roughly doubles cost."""
    small = linear_flop_estimate(10_000, 8)["fit_flops"]
    large = linear_flop_estimate(20_000, 8)["fit_flops"]
    assert large / small == pytest.approx(2.0, rel=0.01)


def test_profile_model_uses_cv_rows_only(data):
    """Profiling trains on the CV rows only and reports positive measurements."""
    result = profile_model(data, "linear", "B", n_repeats=2)
    folded = assign_folds(data)
    assert result.n_rows == (folded["fold"] != TEST_FOLD).sum()
    assert result.n_features == 8
    assert result.fit_time_s > 0 and result.predict_time_s > 0
    assert result.peak_fit_bytes > 0
    assert "fit" in result.profile_text
    summary = result.summary()
    assert "profile_text" not in summary
    assert summary["fit_mflops_per_s"] > 0
