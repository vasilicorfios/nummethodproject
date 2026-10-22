"""Tests for :mod:`functions.evaluate`.

Metrics, bands and breakdowns are checked against hand-calculated values.
The cross-validation tests check the guarantees the results depend on:
the test hold-out is never used and every CV row is predicted exactly once
per seed by a model that did not see it.
"""

import numpy as np
import pandas as pd
import pytest

from functions.evaluate import (
    ANGLE_BAND_LABELS,
    BREAKDOWN_COLUMNS,
    FREQUENCY_BAND_LABELS,
    CVResult,
    angle_band,
    coefficient_summary,
    compute_metrics,
    cross_validate,
    frequency_band,
    oof_rmse_breakdown,
    summarise,
)
from functions.make_folds import CV_SEEDS, N_FOLDS, TEST_FOLD, assign_folds


@pytest.fixture(scope="module")
def linear_a(data):
    """Cross-validation result for the linear baseline on feature set A."""
    return cross_validate(data, "linear", "A")


def test_compute_metrics_hand_calculated():
    """Errors (0, 0, 0, 2): RMSE 1, MAE 0.5, R² = 1 - 4/5."""
    m = compute_metrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(0.2)


def test_perfect_predictions():
    """Perfect predictions score RMSE 0 and R² 1."""
    m = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}


@pytest.mark.parametrize("angle, band", [
    (0.0, "0"), (0.1, "0-5"), (5.0, "0-5"), (5.01, "5-10"),
    (10.0, "5-10"), (15.0, "10-15"), (15.4, ">15"), (22.2, ">15"),
])
def test_angle_band_edges(angle, band):
    """Zero is its own band; upper edges are inclusive."""
    assert angle_band(pd.Series([angle])).iat[0] == band


@pytest.mark.parametrize("freq, band", [
    (200, "200-500 Hz"), (500, "200-500 Hz"), (630, "630-1600 Hz"),
    (1600, "630-1600 Hz"), (2000, "2-5 kHz"), (5000, "2-5 kHz"),
    (6300, "6.3-20 kHz"), (20000, "6.3-20 kHz"),
])
def test_frequency_band_edges(freq, band):
    """Each edge frequency starts the band above it."""
    assert frequency_band(pd.Series([float(freq)])).iat[0] == band


def test_every_measurement_gets_a_band(data):
    """No real row falls outside the defined bands."""
    assert angle_band(data["attack_angle_deg"]).notna().all()
    assert frequency_band(data["frequency_hz"]).notna().all()


def test_test_set_never_used(data, linear_a):
    """No held-out test row is ever predicted (and so never trained on)."""
    folded = assign_folds(data)
    test_rows = set(folded.index[folded["fold"] == TEST_FOLD])
    assert test_rows.isdisjoint(set(linear_a.oof["row"]))
    # Training sizes add up to the CV rows only, never including test rows.
    n_cv = len(data) - len(test_rows)
    assert ((linear_a.fold_scores["n_train"] + linear_a.fold_scores["n_val"]) == n_cv).all()


def test_each_cv_row_predicted_once_per_seed(data, linear_a):
    """OOF predictions cover every CV row exactly once in every repeat."""
    for seed in CV_SEEDS:
        folded = assign_folds(data, cv_seed=seed)
        cv_rows = folded.index[folded["fold"] != TEST_FOLD]
        rows = linear_a.oof.loc[linear_a.oof["seed"] == seed, "row"]
        assert rows.is_unique
        assert set(rows) == set(cv_rows)


def test_fold_scores_shape_and_timing(linear_a):
    """One score row per seed and fold, with positive measured times."""
    assert len(linear_a.fold_scores) == len(CV_SEEDS) * N_FOLDS
    assert (linear_a.fold_scores[["fit_time_s", "predict_time_s"]] > 0).all().all()


def test_baseline_result_regression(linear_a):
    """The linear baseline reproduces its recorded RMSE (detects unintended changes)."""
    assert summarise(linear_a)["rmse"] == pytest.approx(4.894, abs=1e-3)


def test_summarise_matches_manual_calculation(linear_a):
    """Headline RMSE is the mean ± sample SD of all 15 per-fold RMSEs."""
    per_fold = [
        np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))
        for _, g in linear_a.oof.groupby(["seed", "fold"])
    ]
    assert len(per_fold) == len(CV_SEEDS) * N_FOLDS
    summary = summarise(linear_a)
    assert summary["n_folds"] == len(per_fold)
    assert summary["rmse"] == pytest.approx(np.mean(per_fold))
    assert summary["rmse_std"] == pytest.approx(np.std(per_fold, ddof=1))


def _hand_made_result() -> CVResult:
    """A tiny CVResult with known errors for breakdown tests."""
    oof = pd.DataFrame({
        "seed": [1, 1, 1],
        "fold": [0, 0, 1],
        "row": [0, 1, 2],
        "y_true": [100.0, 100.0, 100.0],
        "y_pred": [103.0, 96.0, 100.0],  # errors 3, -4, 0
        "chord_length_m": [0.1, 0.1, 0.2],
        "velocity_mps": [40.0, 40.0, 40.0],
        "angle_band": pd.Categorical(["0", "0", "0-5"], ANGLE_BAND_LABELS, ordered=True),
        "frequency_band": pd.Categorical(["2-5 kHz"] * 3, FREQUENCY_BAND_LABELS, ordered=True),
    })
    coefs = pd.DataFrame({
        "seed": [1, 1, 1],
        "fold": [0, 1, 2],
        "feature": ["x", "x", "x"],
        "effect_db_per_sd": [1.0, 1.0, -1.0],
    })
    return CVResult("linear", "A", pd.DataFrame(), oof, coefs)


def test_rmse_breakdown_hand_calculated():
    """Chord 0.1 has errors 3 and -4: RMSE = sqrt(12.5); chord 0.2 has none."""
    out = oof_rmse_breakdown(_hand_made_result())
    chord = out[out["breakdown"] == "chord_length_m"].set_index("group")
    assert chord.loc[0.1, "rmse"] == pytest.approx(np.sqrt(12.5))
    assert chord.loc[0.1, "n"] == 2
    assert chord.loc[0.2, "rmse"] == pytest.approx(0.0)
    # Empty band categories are left out rather than reported as NaN.
    angle = out[out["breakdown"] == "angle_band"]
    assert set(angle["group"]) == {"0", "0-5"}
    assert set(out["breakdown"]) == set(BREAKDOWN_COLUMNS)


def test_coefficient_summary_hand_calculated():
    """Effects (1, 1, -1): mean 1/3 and two of three fits agree in sign."""
    out = coefficient_summary(_hand_made_result())
    assert out.loc["x", "effect_db_per_sd"] == pytest.approx(1 / 3)
    assert out.loc["x", "sign_agreement"] == pytest.approx(2 / 3)


def test_coefficients_recorded_for_linear(linear_a):
    """One coefficient per feature for every one of the 15 fits."""
    assert len(linear_a.coefficients) == len(CV_SEEDS) * N_FOLDS * 5


def test_per_setup_rmse_hand_calculated(data):
    """One setup's errors (3, -4) in seed 1 and (0, 0) in seed 2: mean RMSE sqrt(12.5)/2."""
    from functions.evaluate import per_setup_rmse
    setups = assign_folds(data)
    rows = setups.index[setups["setup_id"] == 0][:2]
    oof = pd.DataFrame({
        "seed": [1, 1, 2, 2],
        "row": list(rows) * 2,
        "y_true": [100.0, 100.0, 100.0, 100.0],
        "y_pred": [97.0, 104.0, 100.0, 100.0],  # residuals 3, -4, 0, 0
    })
    out = per_setup_rmse(CVResult("m", "A", pd.DataFrame(), oof, pd.DataFrame()), data)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["setup_id"] == 0 and row["n_rows"] == 2
    assert row["rmse"] == pytest.approx(np.sqrt(12.5) / 2)
    assert row["bias"] == pytest.approx(-0.25)  # mean of seed biases (-0.5, 0)
    assert row["chord_length_m"] == setups.loc[rows[0], "chord_length_m"]


def test_per_setup_rmse_covers_every_cv_setup(data, linear_a):
    """Every cross-validation setup appears once; no test-set setup appears."""
    from functions.evaluate import per_setup_rmse
    out = per_setup_rmse(linear_a, data)
    folded = assign_folds(data)
    cv_setups = set(folded.loc[folded["fold"] != TEST_FOLD, "setup_id"])
    assert set(out["setup_id"]) == cv_setups
    assert out["setup_id"].is_unique
    assert out["rmse"].is_monotonic_decreasing  # worst first
