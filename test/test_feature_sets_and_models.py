"""Tests for :mod:`functions.feature_sets` and :mod:`functions.models`.

The model tests use a manufactured solution: data generated from a known
linear formula, which the linear pipeline must recover exactly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from functions.data_loader import FEATURES
from functions.feature_sets import FEATURE_SETS, get_feature_columns
from functions.models import MODELS, build_model, coefficient_effects
from functions.physics_features import PHYSICS_FEATURES

#: Coefficients of the manufactured linear relationship y = 3 + 2*x1 - 5*x2.
TRUE_INTERCEPT, TRUE_COEFS = 3.0, np.array([2.0, -5.0])


@pytest.fixture
def linear_data():
    """Noise-free data from a known linear formula, inputs on very different scales."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "x1": rng.uniform(0, 1e6, 200),   # Reynolds-number-like scale
        "x2": rng.uniform(0, 1e-2, 200),  # Strouhal-number-like scale
    })
    y = pd.Series(TRUE_INTERCEPT + X.to_numpy() @ TRUE_COEFS)
    return X, y


def test_feature_sets_contents():
    """Set A is the raw inputs; set B adds the physics features."""
    assert get_feature_columns("A") == FEATURES
    assert get_feature_columns("B") == FEATURES + PHYSICS_FEATURES


def test_feature_set_returns_copy():
    """Changing the returned list must not change the registry."""
    cols = get_feature_columns("A")
    cols.append("junk")
    assert "junk" not in FEATURE_SETS["A"]


def test_unknown_names_raise():
    """Typos in feature set or model names fail with KeyError."""
    with pytest.raises(KeyError):
        get_feature_columns("Z")
    with pytest.raises(KeyError):
        build_model("not_a_model")


def test_build_model_structure():
    """Each model is a fresh, unfitted target-scaled pipeline."""
    for name, spec in MODELS.items():
        a, b = build_model(name), build_model(name)
        assert a is not b
        assert isinstance(a, TransformedTargetRegressor)
        assert isinstance(a.transformer, StandardScaler)
        assert ("scaler" in a.regressor.named_steps) == spec.scale_inputs
        assert not hasattr(a, "regressor_")  # not fitted


def test_mlp_registered_with_early_stopping():
    """The MLP stops early after 20 epochs without improvement, with a fixed seed."""
    mlp = build_model("mlp").regressor.named_steps["regressor"]
    assert mlp.early_stopping is True
    assert mlp.n_iter_no_change == 20
    assert mlp.max_iter == 2000
    assert mlp.random_state == 0
    assert MODELS["mlp"].scale_inputs


def test_linear_recovers_manufactured_solution(linear_data):
    """Scaling must not stop the linear model reproducing an exact linear law."""
    X, y = linear_data
    model = build_model("linear").fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, rtol=1e-9)


def test_scalers_fitted_on_training_rows_only(linear_data):
    """Scaler statistics come from the rows passed to fit, nothing else."""
    X, y = linear_data
    train_X, train_y = X.iloc[:100], y.iloc[:100]
    model = build_model("linear").fit(train_X, train_y)
    np.testing.assert_allclose(
        model.regressor_.named_steps["scaler"].mean_, train_X.mean().to_numpy()
    )
    np.testing.assert_allclose(model.transformer_.mean_, [train_y.mean()])
    assert not np.allclose(model.regressor_.named_steps["scaler"].mean_, X.mean())


def test_coefficient_effects_match_known_coefficients(linear_data):
    """Effect per std equals the true raw coefficient times the input's std."""
    X, y = linear_data
    model = build_model("linear").fit(X, y)
    effects = coefficient_effects(model, X.columns.tolist())
    # StandardScaler uses the population std (ddof=0).
    expected = TRUE_COEFS * X.std(ddof=0).to_numpy()
    # Inputs span ~1e8 in scale, so scaling and unscaling leave round-off at
    # the ~1e-9 relative level; 1e-6 is still far tighter than any real error.
    np.testing.assert_allclose(effects.to_numpy(), expected, rtol=1e-6)
    assert effects.index.tolist() == ["x1", "x2"]


def test_coefficient_effects_none_without_coefficients(linear_data):
    """Models without linear coefficients (e.g. trees) return None."""
    X, y = linear_data
    tree = TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", DecisionTreeRegressor(random_state=0)),
        ]),
        transformer=StandardScaler(),
    ).fit(X, y)
    assert coefficient_effects(tree, X.columns.tolist()) is None


def test_final_and_runner_up_mlp_settings():
    """The frozen models have the chosen sizes and alphas; everything else is the default."""
    from functions.models import (
        FINAL_FEATURE_SET,
        FINAL_MLP,
        FINAL_MODEL_NAME,
        RUNNER_UP_MLP,
        RUNNER_UP_MODEL_NAME,
    )
    assert FINAL_FEATURE_SET == "B"
    assert FINAL_MLP == {"hidden_layer_sizes": (128, 128), "alpha": 1.0}
    assert RUNNER_UP_MLP == {"hidden_layer_sizes": (64, 64), "alpha": 0.1}
    for name, settings in ((FINAL_MODEL_NAME, FINAL_MLP), (RUNNER_UP_MODEL_NAME, RUNNER_UP_MLP)):
        model = build_model(name)
        mlp = model.regressor.named_steps["regressor"]
        assert mlp.hidden_layer_sizes == settings["hidden_layer_sizes"]
        assert mlp.alpha == settings["alpha"]
        assert mlp.early_stopping is True
        assert mlp.n_iter_no_change == 20
        assert mlp.max_iter == 2000
        assert mlp.random_state == 0
        assert mlp.validation_fraction == 0.1
        assert "scaler" in model.regressor.named_steps
