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

Only the linear baseline is registered so far. Tree-based models, SVR and an
MLP are planned; each is added as one :class:`ModelSpec` entry in
:data:`MODELS` (trees do not need input scaling, so they would use
``scale_inputs=False``).

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


#: Model name -> :class:`ModelSpec`. Add planned models (tree-based, SVR,
#: MLP) here; anything with randomness must be given a fixed
#: ``random_state`` so results are reproducible.
MODELS = {
    "linear": ModelSpec(make=LinearRegression, scale_inputs=True),
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
