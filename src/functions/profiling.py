"""Runtime, memory, profiler and FLOP analysis of model training and prediction.

:func:`profile_model` fits a model on all cross-validation rows (the test
hold-out is dropped first, exactly as in :mod:`functions.evaluate`) and
measures:

* **Time:** fit and predict are each repeated :data:`N_TIMING_REPEATS`
  times and timed with :func:`time.perf_counter`; the median and the
  minimum are reported. The median is the typical time; the minimum is the
  least disturbed by other programs running on the computer.
* **Peak memory:** with :mod:`tracemalloc`, which sees Python and NumPy
  allocations. Memory allocated directly inside compiled BLAS/LAPACK code is
  not seen, so the figure is a lower bound. Memory is measured in its own
  fit because tracemalloc slows code down and would distort the timings.
* **Model size:** the number of fitted weights, their size as 8-byte
  numbers, and the size of the fitted model when saved with :mod:`pickle`.
  For an MLP the pickled file is about 3x the weights alone, because
  scikit-learn also saves the Adam optimiser's two running averages per
  weight, which are only needed to continue training.
* **Epochs:** for an MLP, ``n_iter_``, the number of passes over the data
  before early stopping ended training.
* **Profile:** a :mod:`cProfile` run of one fit, listing where the time goes.
* **FLOPs:** **estimates from formulas**, not measurements
  (:func:`linear_flop_estimate`, :func:`mlp_flop_estimate`). Dividing them
  by the measured time gives the achieved FLOP rate, which shows whether a
  fit is limited by arithmetic or by Python and library overhead.

:func:`machine_info` records the computer and library versions, including
how many threads the maths libraries were allowed to use, so the timings can
be reproduced.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> from functions.physics_features import add_physics_features
    >>> from functions.profiling import profile_model
    >>> data = add_physics_features(load_airfoil_data())
    >>> profile_model(data, "linear", "B").summary()   # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

import cProfile
import io
import math
import os
import pickle
import platform
import pstats
import sys
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor

from functions.data_loader import TARGET
from functions.feature_sets import get_feature_columns
from functions.make_folds import CV_SEEDS, TEST_FOLD, assign_folds
from functions.models import build_model

#: Number of timed fit/predict repeats; the median and minimum are reported.
N_TIMING_REPEATS = 5

#: Number of functions listed in the cProfile report.
N_PROFILE_LINES = 15


def linear_flop_estimate(n: int, p: int) -> dict[str, float]:
    """Estimate floating-point operations for the linear pipeline.

    Counts for an ``n`` x ``p`` training matrix:

    * Input scaler: mean and variance (~3np) plus transform (~2np).
    * Target scaler: the same for one column (~5n).
    * ``LinearRegression``: centring (~2np), then a least-squares solve with
      LAPACK ``gelsd`` (SVD based). For ``n >> p`` this costs about
      ``2np^2 + 11p^3`` FLOPs (R-SVD least squares, Golub & Van Loan,
      *Matrix Computations*). The ``2np^2`` term dominates.
    * Predict: input transform (2np), matrix-vector product (2np) and
      inverse target transform (2n).

    These are leading-order counts for comparison with measured times, not
    exact operation counts.

    Args:
        n: Number of rows.
        p: Number of input features.

    Returns:
        A dict with ``fit_flops`` and ``predict_flops``.
    """
    fit = 2 * n * p**2 + 11 * p**3 + 7 * n * p + 5 * n
    predict = 4 * n * p + 2 * n
    return {"fit_flops": float(fit), "predict_flops": float(predict)}


def mlp_weight_count(n_inputs: int, hidden_layer_sizes: tuple[int, ...]) -> int:
    """Number of weights and biases in a fully connected MLP with one output.

    Each layer has (inputs x outputs) weights plus one bias per output.
    For example 8 inputs and hidden layers (128, 128) give
    ``8*128 + 128*128 + 128*1`` weights and ``128 + 128 + 1`` biases.

    Args:
        n_inputs: Number of input features.
        hidden_layer_sizes: Neurons in each hidden layer.

    Returns:
        Total number of trainable parameters.
    """
    sizes = [n_inputs, *hidden_layer_sizes, 1]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def mlp_flop_estimate(n: int, n_weights: int, n_iter: int,
                      validation_fraction: float = 0.1) -> dict[str, float]:
    """Estimate floating-point operations for training and using an MLP.

    Rule of thumb: one forward pass costs about 2 FLOPs per weight per row
    (a multiply and an add), and the backward pass (computing gradients)
    about 4, so training costs about 6 per weight per row per epoch.

    With early stopping, ``MLPRegressor`` holds back
    ``ceil(validation_fraction * n)`` rows. Only the rest are trained on
    (6 FLOPs per weight), but the held-back rows are scored with a forward
    pass every epoch (2 FLOPs per weight)::

        fit     = (6 * n_train + 2 * n_val) * n_weights * n_iter
        predict = 2 * n * n_weights

    This ignores the scalers, activation functions and the optimiser update
    (all much smaller), so it is an estimate, not a measurement.

    Args:
        n: Rows passed to ``fit`` (and to ``predict``).
        n_weights: Weights plus biases (:func:`mlp_weight_count`).
        n_iter: Epochs run (``n_iter_`` of the fitted model).
        validation_fraction: Share of rows held back for early stopping.

    Returns:
        A dict with ``fit_flops`` and ``predict_flops``.
    """
    n_val = math.ceil(validation_fraction * n)
    n_train = n - n_val
    fit = (6 * n_train + 2 * n_val) * n_weights * n_iter
    predict = 2 * n * n_weights
    return {"fit_flops": float(fit), "predict_flops": float(predict)}


def _inner_estimator(model: TransformedTargetRegressor):
    """The fitted estimator inside the scaling pipeline."""
    return model.regressor_.named_steps["regressor"]


def _model_details(model: TransformedTargetRegressor, model_name: str,
                   n: int, p: int) -> dict:
    """Weights, epochs and FLOP estimates of a fitted model.

    Returns a dict with ``n_weights``, ``n_iter`` (``None`` for models that
    do not train in epochs), ``fit_flops`` and ``predict_flops`` (``None``
    when no formula exists for the model).
    """
    estimator = _inner_estimator(model)
    if isinstance(estimator, MLPRegressor):
        n_weights = (sum(w.size for w in estimator.coefs_)
                     + sum(b.size for b in estimator.intercepts_))
        flops = mlp_flop_estimate(n, n_weights, estimator.n_iter_, estimator.validation_fraction)
        return {"n_weights": n_weights, "n_iter": estimator.n_iter_, **flops}
    if model_name == "linear":
        # p coefficients plus one intercept.
        return {"n_weights": p + 1, "n_iter": None, **linear_flop_estimate(n, p)}
    return {"n_weights": None, "n_iter": None, "fit_flops": None, "predict_flops": None}


def machine_info() -> dict[str, str]:
    """Describe the computer, software and thread settings used for timing.

    Thread counts matter: NumPy's maths library (BLAS) may spread a matrix
    product over several CPU cores, which changes timings. The values come
    from :mod:`threadpoolctl` (installed with scikit-learn) and the
    ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` / ``MKL_NUM_THREADS``
    environment variables ("unset" means the library chose for itself).

    Returns:
        A dict of text values: ``cpu``, ``logical_cpus``, ``platform``,
        ``python``, ``numpy``, ``scikit_learn``, the three thread environment
        variables, and ``blas`` (library name and threads in use).
    """
    info = {
        "cpu": _cpu_name(),
        "logical_cpus": str(os.cpu_count()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        info[var] = os.environ.get(var, "unset")
    try:
        from threadpoolctl import threadpool_info
        pools = [f"{p.get('internal_api')} ({p.get('num_threads')} threads)" for p in threadpool_info()]
        info["blas"] = "; ".join(pools) or "none detected"
    except ImportError:  # threadpoolctl missing: report rather than fail
        info["blas"] = "unknown (threadpoolctl not installed)"
    return info


def _cpu_name() -> str:
    """Human-readable CPU name, falling back to :func:`platform.processor`."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


@dataclass
class ProfileResult:
    """Output of :func:`profile_model`.

    Attributes:
        model: Name of the profiled model.
        feature_set: Name of the feature set.
        n_rows: Training rows (all cross-validation rows, no test rows).
        n_features: Number of input features.
        n_weights: Fitted weights plus biases (``None`` if unknown).
        n_iter: Epochs run before early stopping (``None`` for non-MLPs).
        fit_time_s: Median fit time over the timing repeats.
        fit_time_min_s: Fastest fit time over the repeats.
        predict_time_s: Median time to predict all ``n_rows`` rows.
        predict_time_min_s: Fastest predict time over the repeats.
        input_bytes: Size of the training input matrix as float64.
        peak_fit_bytes: Peak memory traced by tracemalloc during one fit.
        model_bytes: Size of the fitted model when pickled.
        fit_flops: Estimated fit FLOPs, or ``None`` if no formula exists.
        predict_flops: Estimated predict FLOPs, or ``None``.
        profile_text: The cProfile report of one fit, sorted by cumulative
            time.
    """

    model: str
    feature_set: str
    n_rows: int
    n_features: int
    n_weights: int | None
    n_iter: int | None
    fit_time_s: float
    fit_time_min_s: float
    predict_time_s: float
    predict_time_min_s: float
    input_bytes: int
    peak_fit_bytes: int
    model_bytes: int
    fit_flops: float | None
    predict_flops: float | None
    profile_text: str

    def summary(self) -> dict[str, float | str | None]:
        """Return the headline numbers as one flat row.

        Returns:
            A dict of every attribute except ``profile_text``, plus
            ``weight_bytes`` (weights stored as 8-byte floats; ``None`` if
            unknown), and
            ``fit_mflops_per_s`` and ``predict_mflops_per_s`` (achieved rate:
            estimated FLOPs / median measured time, in millions per second;
            ``None`` without a FLOP estimate).
        """
        row = {k: v for k, v in vars(self).items() if k != "profile_text"}
        row["weight_bytes"] = None if self.n_weights is None else 8 * self.n_weights
        for stage in ["fit", "predict"]:
            flops = row[f"{stage}_flops"]
            row[f"{stage}_mflops_per_s"] = (
                None if flops is None else flops / row[f"{stage}_time_s"] / 1e6
            )
        return row


def profile_model(
    data: pd.DataFrame,
    model_name: str,
    feature_set: str,
    n_repeats: int = N_TIMING_REPEATS,
) -> ProfileResult:
    """Measure time, peak memory, size and the call profile of one model.

    The model is trained on all cross-validation rows for the first seed in
    :data:`~functions.make_folds.CV_SEEDS` (the test hold-out is the same for
    every seed and is dropped first). This is the largest training set used
    during model selection, so it gives an upper bound on per-fold costs.

    Args:
        data: Dataset with every column the feature set needs (see
            :func:`functions.evaluate.cross_validate`).
        model_name: Key of :data:`functions.models.MODELS`.
        feature_set: Key of :data:`functions.feature_sets.FEATURE_SETS`.
        n_repeats: Number of timed fit and predict calls.

    Returns:
        A :class:`ProfileResult`.

    Raises:
        KeyError: If the model or feature set name is unknown.
    """
    columns = get_feature_columns(feature_set)
    folded = assign_folds(data, cv_seed=CV_SEEDS[0])
    # Drop the test hold-out before anything else: it is never profiled on.
    cv = folded[folded["fold"] != TEST_FOLD]
    X, y = cv[columns], cv[TARGET]

    fit_times, predict_times = [], []
    for _ in range(n_repeats):
        # A fresh model each time, so every repeat does the full training.
        model = build_model(model_name)
        start = time.perf_counter()
        model.fit(X, y)
        fit_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        model.predict(X)
        predict_times.append(time.perf_counter() - start)

    # Separate untimed fit: tracemalloc's bookkeeping would inflate times.
    tracemalloc.start()
    build_model(model_name).fit(X, y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profiler = cProfile.Profile()
    profiled = build_model(model_name)
    profiler.enable()
    profiled.fit(X, y)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(N_PROFILE_LINES)

    n, p = X.shape
    details = _model_details(model, model_name, n, p)

    return ProfileResult(
        model=model_name,
        feature_set=feature_set,
        n_rows=n,
        n_features=p,
        n_weights=details["n_weights"],
        n_iter=details["n_iter"],
        fit_time_s=float(np.median(fit_times)),
        fit_time_min_s=float(np.min(fit_times)),
        predict_time_s=float(np.median(predict_times)),
        predict_time_min_s=float(np.min(predict_times)),
        input_bytes=n * p * 8,
        peak_fit_bytes=peak,
        # Size of the saved model, including the scalers in the pipeline.
        model_bytes=len(pickle.dumps(model)),
        fit_flops=details["fit_flops"],
        predict_flops=details["predict_flops"],
        profile_text=buffer.getvalue(),
    )
