"""Tests for :mod:`functions.shift_tests` (leave one chord / velocity out)."""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import functions.shift_tests as shift_tests  # noqa: E402
from functions.make_folds import TEST_FOLD, assign_folds  # noqa: E402
from functions.plot_robustness import plot_shift_tests  # noqa: E402
from functions.plot_tuning import save_png_pdf  # noqa: E402
from functions.shift_tests import (  # noqa: E402
    SHIFT_COLUMNS,
    SHIFTS,
    cv_rows,
    in_distribution_reference,
    run_shift_tests,
    shift_splits,
)

#: A tiny, fast MLP for tests (the real one takes about a second per fit).
TINY_MLP = {"regressor__regressor__hidden_layer_sizes": (4,), "regressor__regressor__max_iter": 5}


@pytest.fixture(scope="module")
def tiny_table(data):
    """Shift tests with linear and a tiny 2-seed MLP."""
    with pytest.warns(Warning):  # ConvergenceWarning from max_iter=5
        return run_shift_tests(data, mlp_seeds=[0, 1], mlp_params=TINY_MLP, progress=None)


def _test_rows(data):
    folded = assign_folds(data)
    return set(folded.index[folded["fold"] == TEST_FOLD])


@pytest.mark.parametrize("column", list(SHIFTS))
def test_held_out_value_never_in_training(data, column):
    """Each split's training rows never contain the held-out chord / velocity."""
    cv = cv_rows(data)
    n_splits = 0
    for value, regime, train, test in shift_splits(cv, column):
        assert value not in set(train[column])
        assert set(test[column]) == {value}
        assert len(train) + len(test) == len(cv)
        n_splits += 1
    assert n_splits == cv[column].nunique()


@pytest.mark.parametrize("column", list(SHIFTS))
def test_no_holdout_rows_in_shift_splits(data, column):
    """Neither training nor test rows of any split come from the test hold-out."""
    holdout = _test_rows(data)
    for _, _, train, test in shift_splits(cv_rows(data), column):
        assert holdout.isdisjoint(train.index) and holdout.isdisjoint(test.index)


def test_no_holdout_rows_reach_any_fit(data, monkeypatch):
    """Every model fitted by run_shift_tests sees only cross-validation rows."""
    seen = []
    real_build = shift_tests.build_model

    def recording_build(name):
        model = real_build(name)
        real_fit = model.fit

        def fit(X, y, **kwargs):
            seen.append(set(X.index))
            return real_fit(X, y, **kwargs)

        model.fit = fit
        return model

    monkeypatch.setattr(shift_tests, "build_model", recording_build)
    run_shift_tests(data, mlp_seeds=[], progress=None)
    assert len(seen) == 6 + 4  # one linear fit per held-out chord and velocity
    holdout = _test_rows(data)
    assert all(rows.isdisjoint(holdout) for rows in seen)


def test_extrapolation_labels():
    """Smallest and largest held-out values are extrapolation, the rest interpolation."""
    cv = pd.DataFrame({"velocity_mps": [31.7, 39.6, 55.5, 71.3]})
    regimes = {v: r for v, r, _, _ in shift_splits(cv, "velocity_mps")}
    assert regimes == {31.7: "extrapolation", 39.6: "interpolation",
                       55.5: "interpolation", 71.3: "extrapolation"}


def test_table_shape_and_coverage(tiny_table):
    """One row per held-out value per model; every CV setup is tested once per shift."""
    assert list(tiny_table.columns) == SHIFT_COLUMNS
    assert len(tiny_table) == (6 + 4) * 2
    linear = tiny_table[tiny_table["model"] == "linear"]
    mlp = tiny_table[tiny_table["model"] == "mlp"]
    assert (linear["n_runs"] == 1).all() and linear["rmse_sd_seeds"].isna().all()
    assert (mlp["n_runs"] == 2).all() and mlp["rmse_sd_seeds"].notna().all()
    for shift in SHIFTS.values():
        part = linear[linear["shift"] == shift]
        assert part["n_setups"].sum() == 90 and part["n_rows"].sum() == 1276


def test_in_distribution_reference_reads_saved_summary():
    """The reference is read from tuning_summary.csv, not recomputed."""
    summary = pd.DataFrame({"feature_set": ["A", "B"], "rmse_mean": [2.6, 2.4],
                            "linear_rmse_mean": [4.9, 4.6]})
    assert in_distribution_reference(summary) == {"mlp": 2.4, "linear": 4.6}


def test_shift_figure_saved(tiny_table, tmp_path):
    """The shift-test bar chart saves as PNG and PDF."""
    paths = save_png_pdf(plot_shift_tests(tiny_table, {"mlp": 2.4, "linear": 4.6}), tmp_path, "shift")
    assert all(p.stat().st_size > 0 for p in paths)
    assert np.isfinite(tiny_table["rmse"]).all()
