"""Model registry: every model is built as the same scikit-learn pipeline.

Each model returned by :func:`build_model` has the structure::

    TransformedTargetRegressor(
        regressor=Pipeline([("scaler", StandardScaler()),   # if scale_inputs
                            ("regressor", <estimator>)]),
        transformer=StandardScaler(),                       # scales spl_db
    )

Because the scalers sit inside the estimator, calling ``fit`` on the training
rows fits them on those rows only; validation and test rows are transformed
with the training mean and standard deviation and never influence them.

Scaling the target to zero mean and unit variance matters for SVR (its
``epsilon`` tube and ``C`` are in target units) and for neural networks
(output scale and loss magnitude). For the linear baseline neither scaling
step changes the predictions, but it is applied anyway so every model runs
through the same pipeline and the linear coefficients are comparable in size.

Registered models:

* ``"linear"``: ordinary least squares, the baseline.
* ``"mlp"``: a multi-layer perceptron (``MLPRegressor``) with early
  stopping. The layer sizes and ``alpha`` set here are only defaults; the
  values actually reported come from nested cross-validation in
  :mod:`functions.tuning`.
* ``"mlp_final"``: the MLP with the frozen final settings
  :data:`FINAL_MLP`, chosen from the nested-CV results, for use on
  :data:`FINAL_FEATURE_SET`.
* ``"mlp_runner_up"``: the cheaper alternative :data:`RUNNER_UP_MLP`.

Later scripts (profiling, shift tests, the final hold-out run) must use
:data:`FINAL_MODEL_NAME`, :data:`FINAL_FEATURE_SET` and these constants
rather than retyping the values.

Tree-based models and SVR are planned; each is added as one
:class:`ModelSpec` entry in :data:`MODELS` (trees do not need input
scaling, so they would use ``scale_inputs=False``).

Example:
    >>> from functions.models import build_model
    >>> model = build_model("linear")
    >>> model.fit(X_train, y_train).predict(X_val)  # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas, and evaluating
# ``pd.Series | None`` on the mock at import time would fail.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Default MLP settings. Hidden layer sizes and ``alpha`` are tuned by
#: :mod:`functions.tuning`; the rest are fixed for every fit.
MLP_DEFAULTS = {
    "hidden_layer_sizes": (64, 64),
    "alpha": 1e-3,  # L2 penalty: larger values give smoother, simpler fits
    # Hold back 10% of the training rows and stop once their score has not
    # improved by more than tol for 20 epochs in a row. This stops
    # training before the network starts memorising noise.
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "tol": 1e-4,
    # A high cap, so early stopping (not the cap) decides when to stop.
    "max_iter": 2000,
    # Fixed so the random starting weights, and therefore results, repeat.
    "random_state": 0,
}


#: Feature set of the final model: the 5 raw inputs plus log-frequency,
#: Strouhal and Reynolds numbers. The tuned MLP scored better on B than on A
#: (outer RMSE 2.37 vs 2.56 dB over the same 15 folds).
FINAL_FEATURE_SET = "B"

#: Frozen final MLP settings, chosen from the nested-CV results on set B
#: (cross-validation data only; the test hold-out was not used). This
#: setting won 7 of the 15 outer folds, more than any other, and had the
#: lowest mean inner-CV RMSE of all 35 settings (2.50 dB). Only these two
#: values differ from :data:`MLP_DEFAULTS`; early stopping, ``max_iter``,
#: ``random_state`` and ``validation_fraction`` stay the same.
FINAL_MLP = {"hidden_layer_sizes": (128, 128), "alpha": 1.0}

#: A cheaper alternative to profile against the final model: about 4x
#: fewer weights, with a mean inner-CV RMSE of 2.54 dB, about 0.04 dB worse.
#: It differs from :data:`FINAL_MLP` in both network size and ``alpha``.
#: (The same network with ``alpha=1.0`` scored 2.51 dB; ``alpha=0.1`` was
#: chosen deliberately.)
RUNNER_UP_MLP = {"hidden_layer_sizes": (64, 64), "alpha": 0.1}

#: Registry names of the frozen models.
FINAL_MODEL_NAME = "mlp_final"
RUNNER_UP_MODEL_NAME = "mlp_runner_up"


def _make_mlp(**overrides) -> MLPRegressor:
    """Return a new MLP with :data:`MLP_DEFAULTS`, optionally overridden.

    Arguments are passed by keyword: ``MLPRegressor``'s first positional
    parameter is ``loss``, not the layer sizes.

    Args:
        **overrides: Settings that replace the defaults, e.g.
            ``**FINAL_MLP``.
    """
    return MLPRegressor(**{**MLP_DEFAULTS, **overrides})


def _make_final_mlp() -> MLPRegressor:
    """Return a new MLP with the frozen :data:`FINAL_MLP` settings."""
    return _make_mlp(**FINAL_MLP)


def _make_runner_up_mlp() -> MLPRegressor:
    """Return a new MLP with the :data:`RUNNER_UP_MLP` settings."""
    return _make_mlp(**RUNNER_UP_MLP)


@dataclass(frozen=True)
class ModelSpec:
    """How to build one registered model.

    Attributes:
        make: Zero-argument callable returning a new, unfitted estimator.
            A fresh estimator is needed for every fold so no fitted state is
            shared between folds.
        scale_inputs: Whether to standardise the inputs before the
            estimator. Needed for distance- and gradient-based models (SVR,
            MLP), unnecessary for trees.
    """

    make: Callable[[], RegressorMixin]
    scale_inputs: bool = True


#: Model name -> :class:`ModelSpec`. Add planned models (tree-based, SVR)
#: here; anything with randomness must be given a fixed ``random_state`` so
#: results are reproducible.
MODELS = {
    "linear": ModelSpec(make=LinearRegression, scale_inputs=True),
    # Neural networks train by gradient descent, which needs inputs on
    # similar scales, so the inputs are standardised.
    "mlp": ModelSpec(make=_make_mlp, scale_inputs=True),
    # Frozen settings from tuning; built by name so no script retypes them.
    FINAL_MODEL_NAME: ModelSpec(make=_make_final_mlp, scale_inputs=True),
    RUNNER_UP_MODEL_NAME: ModelSpec(make=_make_runner_up_mlp, scale_inputs=True),
}


def build_model(name: str) -> TransformedTargetRegressor:
    """Build a new, unfitted model wrapped in the standard scaling pipeline.

    Args:
        name: Key of :data:`MODELS`, e.g. ``"linear"``.

    Returns:
        A :class:`~sklearn.compose.TransformedTargetRegressor` whose inner
        regressor is a :class:`~sklearn.pipeline.Pipeline` of an optional
        input :class:`~sklearn.preprocessing.StandardScaler` and the
        estimator. The target is standardised by a second ``StandardScaler``
        and predictions are converted back to dB automatically.

    Raises:
        KeyError: If ``name`` is not a registered model.
    """
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]

    steps = []
    if spec.scale_inputs:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", spec.make()))

    return TransformedTargetRegressor(
        regressor=Pipeline(steps),
        transformer=StandardScaler(),
    )


def coefficient_effects(model: TransformedTargetRegressor, feature_names: list[str]) -> pd.Series | None:
    """Return a fitted linear model's coefficients as dB per standard deviation.

    Inside the pipeline both inputs and target are standardised, so each
    fitted coefficient is the change in SPL, in target standard deviations,
    for a one-standard-deviation change in that input with all other inputs
    held fixed. Multiplying by the target's standard deviation turns this
    into dB per input standard deviation, which can be compared directly
    across features with very different units (Hz, m, m/s, Reynolds number).

    Correlated inputs (e.g. ``frequency_hz`` and ``log_frequency`` in set B)
    share their effect between them, so individual coefficients can be large,
    of opposite sign and unstable between folds even when predictions are
    good. Look at the spread over folds before interpreting one.

    Args:
        model: A model from :func:`build_model` that has been fitted.
        feature_names: Input columns in the order the model was fitted on.

    Returns:
        A Series indexed by feature name, in dB per input standard deviation,
        or ``None`` if the model has no linear coefficients or does not scale
        its inputs (raw-unit coefficients are not comparable across features).
    """
    steps = model.regressor_.named_steps
    estimator = steps["regressor"]
    if not hasattr(estimator, "coef_") or "scaler" not in steps:
        return None
    # transformer_ is the target scaler fitted on this model's training rows.
    target_sd = model.transformer_.scale_[0]
    return pd.Series(estimator.coef_.ravel() * target_sd, index=feature_names)
