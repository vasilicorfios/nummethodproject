"""Tests for :mod:`functions.profiling`."""

import pytest
from sklearn.compose import TransformedTargetRegressor

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


def test_mlp_weight_count_128_128_on_8_inputs():
    """(128, 128) on 8 inputs: 8x128 + 128x128 + 128x1 weights plus 128 + 128 + 1 biases."""
    from functions.profiling import mlp_weight_count
    expected = (8 * 128 + 128 * 128 + 128 * 1) + (128 + 128 + 1)
    assert expected == 17793
    assert mlp_weight_count(8, (128, 128)) == expected
    assert mlp_weight_count(8, (64, 64)) == (8 * 64 + 64 * 64 + 64) + (64 + 64 + 1)


def test_mlp_flop_estimate_hand_calculated():
    """n=100 with 10% held back: 90 training rows (6 FLOPs/weight), 10 validation rows (2)."""
    from functions.profiling import mlp_flop_estimate
    est = mlp_flop_estimate(n=100, n_weights=10, n_iter=3, validation_fraction=0.1)
    assert est["fit_flops"] == pytest.approx((6 * 90 + 2 * 10) * 10 * 3)
    assert est["predict_flops"] == pytest.approx(2 * 100 * 10)


class _RecordingModel(TransformedTargetRegressor):
    """A model that remembers which rows every ``fit`` call received.

    Defined at module level so it can be pickled (the profiler pickles the
    fitted model to measure its size).
    """

    seen: list = []

    def fit(self, X, y, **kwargs):
        _RecordingModel.seen.append(set(X.index))
        return super().fit(X, y, **kwargs)


def test_profiler_never_sees_holdout_rows(data, monkeypatch):
    """Every fit inside profile_model receives exactly the CV rows, no test rows."""
    import functions.profiling as profiling

    real_build = profiling.build_model

    def recording_build(name):
        model = real_build(name)
        return _RecordingModel(regressor=model.regressor, transformer=model.transformer)

    _RecordingModel.seen = []
    monkeypatch.setattr(profiling, "build_model", recording_build)
    profiling.profile_model(data, "linear", "B", n_repeats=2)

    folded = assign_folds(data)
    test_rows = set(folded.index[folded["fold"] == TEST_FOLD])
    cv_rows = set(folded.index[folded["fold"] != TEST_FOLD])
    assert len(_RecordingModel.seen) == 4  # 2 timed fits + memory fit + cProfile fit
    assert all(rows == cv_rows and rows.isdisjoint(test_rows) for rows in _RecordingModel.seen)


def test_profile_final_mlp_details(data):
    """The final MLP's profile reports its real weight count, epochs, size and FLOPs."""
    from functions.models import FINAL_FEATURE_SET, FINAL_MODEL_NAME
    from functions.profiling import mlp_flop_estimate
    result = profile_model(data, FINAL_MODEL_NAME, FINAL_FEATURE_SET, n_repeats=1)
    assert result.n_weights == 17793
    assert 0 < result.n_iter < 2000  # stopped early, not at the cap
    assert result.model_bytes > 8 * result.n_weights  # weights plus optimiser state
    assert result.fit_flops == pytest.approx(
        mlp_flop_estimate(result.n_rows, 17793, result.n_iter)["fit_flops"])
    assert result.fit_time_min_s <= result.fit_time_s
    assert result.summary()["weight_bytes"] == 8 * 17793


def test_machine_info_reports_threads():
    """Machine info includes versions and the thread settings used."""
    from functions.profiling import machine_info
    info = machine_info()
    for key in ("cpu", "logical_cpus", "python", "numpy", "scikit_learn",
                "OMP_NUM_THREADS", "blas"):
        assert key in info and info[key]


def test_profile_plan_and_comparison():
    """Default plan is the report's four models; ratios divide the right rows."""
    import pandas as pd
    import run_profile
    plan = run_profile.profile_plan(run_profile.parse_args([]))
    assert plan == [("linear", "A"), ("linear", "B"), ("mlp_final", "B"), ("mlp_runner_up", "B")]
    custom = run_profile.profile_plan(run_profile.parse_args(["--models", "linear", "--feature-sets", "B"]))
    assert custom == [("linear", "B")]

    cols = {c: [10.0, 2.0] for c in run_profile.COMPARED}
    summary = pd.DataFrame({"model": ["a", "b"], "feature_set": ["B", "B"], **cols})
    out = run_profile.compare(summary, ("a", "B"), ("b", "B"))
    assert (out["ratio"] == 5.0).all()
    assert len(out) == len(run_profile.COMPARED)
