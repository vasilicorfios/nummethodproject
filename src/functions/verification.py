"""Manufactured-solution checks: is each model implemented correctly?

A **manufactured solution** is made-up data generated from a formula whose
answer is known exactly. If the code is implemented correctly, the model must
recover that answer. This checks the *implementation* (verification), as
opposed to how well the model predicts real noise data (validation).

Both checks use the real model pipelines from :func:`functions.models.build_model`,
including the input and target scaling, so the whole chain is verified, not just
the estimator inside it.

* **Linear least squares** (:func:`check_linear`): data from
  ``y = 3 + 2 x1 − 5 x2`` with no noise, with the two inputs about 10⁸ apart
  in scale (like Reynolds and Strouhal numbers). An exact solver must recover
  the coefficients and reproduce ``y`` to round-off.
* **MLP** (:func:`check_sine`): 200 evenly spaced points of ``y = sin(x)``
  on [0, 2π]. A correctly implemented network should learn this smooth
  non-linear curve to a small error, and it is scored on points **between**
  the training points, so it must have learned the curve, not memorised the
  points. The linear model is run on the same data as a **negative
  control**: it cannot represent a sine, so it must fail. That shows the test
  is able to fail.

The MLP pass marks (:data:`SINE_MAX_RMSE`, :data:`SINE_MIN_R2`) were set
after a first measurement, with a safety margin. They test that the network
learns a smooth non-linear function, not that it reproduces it exactly: the
final MLP's strong L2 penalty (``alpha = 1``) and early stopping deliberately
smooth the fit. The largest error occurs at the ends of the range, where a
network has data on one side only, so no pass mark is set on the maximum
error.
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

import numpy as np
import pandas as pd

from functions.models import FINAL_MODEL_NAME, build_model, coefficient_effects

#: Manufactured linear law: intercept and coefficients of ``y = 3 + 2 x1 − 5 x2``.
LINEAR_INTERCEPT = 3.0
LINEAR_COEFS = np.array([2.0, -5.0])

#: Number of points in each manufactured dataset.
N_POINTS = 200

#: Linear check: largest allowed relative error in the recovered coefficients.
#: Round-off with inputs 10⁸ apart in scale is about 1e-9, so 1e-6 is strict.
LINEAR_MAX_REL_ERROR = 1e-6

#: MLP check: largest allowed RMSE between training points (10% of the
#: sine's amplitude of 1).
SINE_MAX_RMSE = 0.10

#: MLP check: smallest allowed R² between training points.
SINE_MIN_R2 = 0.98

#: Negative control: the linear model must have at least this RMSE on the
#: sine, i.e. clearly fail.
CONTROL_MIN_RMSE = 0.30


def manufactured_linear(n: int = N_POINTS, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Noise-free data from the known linear law, inputs on very different scales.

    Args:
        n: Number of points.
        seed: Seed for drawing the inputs.

    Returns:
        ``(X, y)``: inputs ``x1`` in [0, 10⁶] and ``x2`` in [0, 10⁻²], and exact ``y``.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.uniform(0, 1e6, n), "x2": rng.uniform(0, 1e-2, n)})
    return X, LINEAR_INTERCEPT + X.to_numpy() @ LINEAR_COEFS


def check_linear(n: int = N_POINTS) -> dict:
    """Fit the linear pipeline to the manufactured law and measure the recovery.

    The raw coefficients are recovered from the scaled model with
    ``beta_j = coef_effect_j / sigma_j`` (see
    :func:`functions.models.coefficient_effects`).

    Args:
        n: Number of points.

    Returns:
        ``max_coef_rel_error``, ``max_abs_pred_error`` (in units of
        ``y``), the recovered coefficients and ``passed``.
    """
    X, y = manufactured_linear(n)
    model = build_model("linear").fit(X, y)
    effects = coefficient_effects(model, list(X.columns)).to_numpy()
    recovered = effects / X.std(ddof=0).to_numpy()  # undo "per standard deviation"
    rel_error = np.abs(recovered - LINEAR_COEFS) / np.abs(LINEAR_COEFS)
    pred_error = np.abs(model.predict(X) - y)
    return {
        "recovered_coefs": recovered,
        "max_coef_rel_error": float(rel_error.max()),
        "max_abs_pred_error": float(pred_error.max()),
        "passed": bool(rel_error.max() <= LINEAR_MAX_REL_ERROR),
    }


def manufactured_sine(n: int = N_POINTS) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Training points and in-between test points of ``y = sin(x)`` on [0, 2π].

    Args:
        n: Number of evenly spaced training points.

    Returns:
        ``(X_train, y_train, X_between, y_between)``. The "between" points
        are a fine grid on the same range with the training points removed,
        so the model is scored where it never saw data.
    """
    x = np.linspace(0, 2 * np.pi, n)
    fine = np.linspace(0, 2 * np.pi, 5 * n + 1)
    between = fine[~np.isclose(fine[:, None], x[None, :]).any(axis=1)]
    return pd.DataFrame({"x": x}), np.sin(x), pd.DataFrame({"x": between}), np.sin(between)


def check_sine(model_name: str = FINAL_MODEL_NAME, n: int = N_POINTS) -> dict:
    """Fit a model to the manufactured sine and score it between the points.

    Args:
        model_name: Key of :data:`functions.models.MODELS`.
        n: Number of training points.

    Returns:
        ``train_rmse``, ``rmse`` and ``r2`` (between the training points),
        ``max_error`` and the ``x`` where it occurs, ``n_iter`` (epochs, for
        MLPs), ``passed`` (MLP pass marks met), and the ``x``/``prediction``
        arrays for plotting.
    """
    X, y, X_between, y_between = manufactured_sine(n)
    model = build_model(model_name).fit(X, y)
    pred = model.predict(X_between)
    error = pred - y_between
    rmse = float(np.sqrt(np.mean(error ** 2)))
    r2 = float(1 - np.sum(error ** 2) / np.sum((y_between - y_between.mean()) ** 2))
    estimator = model.regressor_.named_steps["regressor"]
    k = int(np.argmax(np.abs(error)))
    return {
        "model": model_name,
        "train_rmse": float(np.sqrt(np.mean((model.predict(X) - y) ** 2))),
        "rmse": rmse,
        "r2": r2,
        "max_error": float(np.abs(error[k])),
        "max_error_x": float(X_between["x"].iat[k]),
        "n_iter": getattr(estimator, "n_iter_", None),
        "passed": rmse <= SINE_MAX_RMSE and r2 >= SINE_MIN_R2,
        "x": X_between["x"].to_numpy(),
        "prediction": pred,
    }
