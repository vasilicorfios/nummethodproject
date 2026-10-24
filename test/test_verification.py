"""Manufactured-solution tests: are the linear solver and the MLP implemented correctly?

See :mod:`functions.verification` for the method and the pass marks.
"""

import numpy as np
import pytest

from functions.models import FINAL_MODEL_NAME
from functions.verification import (
    CONTROL_MIN_RMSE,
    LINEAR_COEFS,
    LINEAR_MAX_REL_ERROR,
    N_POINTS,
    SINE_MAX_RMSE,
    SINE_MIN_R2,
    check_linear,
    check_sine,
    manufactured_sine,
)


@pytest.fixture(scope="module")
def mlp_sine():
    """The final MLP pipeline fitted to the manufactured sine (about 1 s)."""
    return check_sine(FINAL_MODEL_NAME)


def test_linear_solver_recovers_manufactured_law():
    """Least squares recovers y = 3 + 2 x1 - 5 x2 exactly (to round-off)."""
    result = check_linear()
    assert result["passed"]
    assert result["max_coef_rel_error"] <= LINEAR_MAX_REL_ERROR
    np.testing.assert_allclose(result["recovered_coefs"], LINEAR_COEFS, rtol=LINEAR_MAX_REL_ERROR)
    assert result["max_abs_pred_error"] < 1e-6


def test_sine_data_scores_between_training_points():
    """200 training points on [0, 2pi]; the scoring points never coincide with them."""
    X, y, X_between, y_between = manufactured_sine()
    assert len(X) == N_POINTS
    assert X["x"].min() == 0 and X["x"].max() == pytest.approx(2 * np.pi)
    assert not np.isclose(X_between["x"].to_numpy()[:, None], X["x"].to_numpy()[None, :]).any()
    np.testing.assert_allclose(y_between, np.sin(X_between["x"]))


def test_final_mlp_learns_a_smooth_nonlinear_function(mlp_sine):
    """The final MLP learns sin(x) to RMSE <= 0.1 and R2 >= 0.98 between the points."""
    assert mlp_sine["rmse"] <= SINE_MAX_RMSE
    assert mlp_sine["r2"] >= SINE_MIN_R2
    assert mlp_sine["passed"]
    # It stopped by early stopping, not by running out of epochs.
    assert 0 < mlp_sine["n_iter"] < 2000


def test_linear_negative_control_fails_on_sine():
    """A straight line cannot fit a sine, so the test must be able to fail."""
    control = check_sine("linear")
    assert control["rmse"] >= CONTROL_MIN_RMSE
    assert not control["passed"]
