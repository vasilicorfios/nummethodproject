"""Runtime, memory, profiler and FLOP analysis of model training and prediction.

:func:`profile_model` fits a model once on all cross-validation rows (the
test hold-out is dropped first, exactly as in :mod:`functions.evaluate`) and
measures:

* **Time:** median of repeated fit and predict calls, timed with
  :func:`time.perf_counter`. Repeats smooth out timer noise on fits that
  take only milliseconds.
* **Peak memory:** with :mod:`tracemalloc`, which sees Python and NumPy
  allocations. Memory allocated directly inside compiled BLAS/LAPACK code is
  not seen, so the figure is a lower bound. Memory is measured in its own
  fit because tracemalloc slows code down and would distort the timings.
* **Profile:** a :mod:`cProfile` run of one fit, listing where the time goes.
* **FLOPs:** an analytical estimate (:func:`linear_flop_estimate`) for
  models that have one. Dividing it by the measured time gives the achieved
  FLOP rate, which shows whether a fit is limited by arithmetic or by
  Python and library overhead.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> from functions.physics_features import add_physics_features
    >>> from functions.profiling import profile_model
    >>> data = add_physics_features(load_airfoil_data())
    >>> profile_model(data, "linear", "B").summary()   # doctest: +SKIP
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from functions.data_loader import TARGET
from functions.feature_sets import get_feature_columns
from functions.make_folds import CV_SEEDS, TEST_FOLD, assign_folds
from functions.models import build_model

#: Number of timed fit/predict repeats; the median is reported.
N_TIMING_REPEATS = 20

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


#: Model name -> FLOP estimate function ``(n, p) -> dict``. Models without an
#: entry are profiled without a FLOP estimate.
FLOP_ESTIMATES: dict[str, Callable[[int, int], dict[str, float]]] = {
    "linear": linear_flop_estimate,
}


@dataclass
class ProfileResult:
    """Output of :func:`profile_model`.

    Attributes:
        model: Name of the profiled model.
        feature_set: Name of the feature set.
        n_rows: Training rows (all cross-validation rows, no test rows).
        n_features: Number of input features.
        fit_time_s: Median fit time over the timing repeats.
        predict_time_s: Median time to predict all ``n_rows`` rows.
        input_bytes: Size of the training input matrix as float64.
        peak_fit_bytes: Peak memory traced by tracemalloc during one fit.
        fit_flops: Estimated fit FLOPs, or ``None`` if no estimate exists.
        predict_flops: Estimated predict FLOPs, or ``None``.
        profile_text: The cProfile report of one fit, sorted by cumulative
            time.
    """

    model: str
    feature_set: str
    n_rows: int
    n_features: int
    fit_time_s: float
    predict_time_s: float
    input_bytes: int
    peak_fit_bytes: int
    fit_flops: float | None
    predict_flops: float | None
    profile_text: str

    def summary(self) -> dict[str, float | str | None]:
        """Return the headline numbers as one flat row.

        Returns:
            A dict of every attribute except ``profile_text``, plus
            ``fit_mflops_per_s`` and ``predict_mflops_per_s`` (achieved rate,
            estimated FLOPs / measured time, in millions per second; ``None``
            without a FLOP estimate).
        """
        row = {k: v for k, v in vars(self).items() if k != "profile_text"}
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
    """Measure time, peak memory and the call profile of one model.

    The model is trained on all cross-validation rows for the first seed in
    :data:`~functions.make_folds.CV_SEEDS` (the test hold-out is the same for
    every seed and is dropped). This is the largest training set used during
    model selection, so it gives an upper bound on per-fold costs.

    Args:
        data: Dataset with every column the feature set needs (see
            :func:`functions.evaluate.cross_validate`).
        model_name: Key of :data:`functions.models.MODELS`.
        feature_set: Key of :data:`functions.feature_sets.FEATURE_SETS`.
        n_repeats: Number of timed fit and predict calls; the median is
            reported. Lower it for slow models.

    Returns:
        A :class:`ProfileResult`.

    Raises:
        KeyError: If the model or feature set name is unknown.
    """
    columns = get_feature_columns(feature_set)
    folded = assign_folds(data, cv_seed=CV_SEEDS[0])
    cv = folded[folded["fold"] != TEST_FOLD]
    X, y = cv[columns], cv[TARGET]

    fit_times, predict_times = [], []
    for _ in range(n_repeats):
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
    model = build_model(model_name)
    profiler.enable()
    model.fit(X, y)
    profiler.disable()
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(N_PROFILE_LINES)

    n, p = X.shape
    estimate = FLOP_ESTIMATES.get(model_name)
    flops = estimate(n, p) if estimate else {"fit_flops": None, "predict_flops": None}

    return ProfileResult(
        model=model_name,
        feature_set=feature_set,
        n_rows=n,
        n_features=p,
        fit_time_s=float(np.median(fit_times)),
        predict_time_s=float(np.median(predict_times)),
        input_bytes=n * p * 8,
        peak_fit_bytes=peak,
        fit_flops=flops["fit_flops"],
        predict_flops=flops["predict_flops"],
        profile_text=buffer.getvalue(),
    )
