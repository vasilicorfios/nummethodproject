"""Robustness shift tests: do the models work on conditions they never saw?

Cross-validation holds out *setups*, but every held-out setup still has
neighbours in training with the same chord and the same velocity. A shift
test is harder: it holds out **every** setup with one chord (or one
velocity), so the model must predict a value of that input it was never
trained on.

* **Leave-one-chord-out:** for each of the 6 chords, train on the other 5,
  test on the held-out chord.
* **Leave-one-velocity-out:** for each of the 4 velocities, train on the
  other 3, test on the held-out one.

A held-out value between the training values is **interpolation**; the
smallest or largest value is **extrapolation**, outside the training range,
which is usually much harder.

Models: the final MLP (:data:`functions.models.FINAL_MLP`) and the linear
baseline, both on :data:`functions.models.FINAL_FEATURE_SET`, built with
:func:`functions.models.build_model` so all scaling stays inside the
pipeline. Nothing is re-tuned. The MLP is repeated with
``random_state`` 0, 1 and 2 (:data:`MLP_SEEDS`) to show the result is not
down to one lucky weight initialisation; linear is deterministic, so it runs
once.

Only cross-validation rows are used: the test hold-out is dropped first.

**Chord and angle are confounded.** High angles of attack were only tested
on the small chords, so leaving out the 0.0254 m chord also removes most of
the high-angle setups. A chord shift result therefore partly reflects angle,
too. An angle-extrapolation test was deliberately not run for the same
reason: holding out the highest angles would mostly hold out the smallest
chord.

Example:
    >>> from functions.shift_tests import run_shift_tests
    >>> table = run_shift_tests(data)   # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from functions.data_loader import TARGET
from functions.evaluate import compute_metrics
from functions.feature_sets import get_feature_columns
from functions.make_folds import TEST_FOLD, assign_folds
from functions.models import FINAL_FEATURE_SET, FINAL_MODEL_NAME, build_model

#: The two inputs that are held out one value at a time, and the test names.
SHIFTS = {
    "chord_length_m": "leave-one-chord-out",
    "velocity_mps": "leave-one-velocity-out",
}

#: ``random_state`` values for the repeated MLP runs.
MLP_SEEDS = [0, 1, 2]

#: Grid key for the MLP's random_state inside the scaling pipeline.
MLP_SEED_KEY = "regressor__regressor__random_state"

#: Columns of the table returned by :func:`run_shift_tests`.
SHIFT_COLUMNS = [
    "shift", "held_out", "regime", "model", "n_runs", "n_setups", "n_rows",
    "rmse", "rmse_sd_seeds", "mae", "r2", "bias",
]


def cv_rows(data: pd.DataFrame) -> pd.DataFrame:
    """The cross-validation rows, with ``setup_id``; the test hold-out removed.

    Args:
        data: Dataset with physics features.

    Returns:
        The rows that are not in the test hold-out.
    """
    folded = assign_folds(data)
    return folded[folded["fold"] != TEST_FOLD]


def shift_splits(cv: pd.DataFrame, column: str):
    """Yield one train/test split per value of ``column``.

    Args:
        cv: Cross-validation rows (see :func:`cv_rows`).
        column: ``"chord_length_m"`` or ``"velocity_mps"``.

    Yields:
        ``(held_out_value, regime, train_rows, test_rows)``. ``regime`` is
        ``"extrapolation"`` for the smallest and largest values (outside the
        remaining training range) and ``"interpolation"`` otherwise.
    """
    values = sorted(cv[column].unique())
    for value in values:
        regime = "extrapolation" if value in (values[0], values[-1]) else "interpolation"
        yield value, regime, cv[cv[column] != value], cv[cv[column] == value]


def _score(model_name: str, train: pd.DataFrame, test: pd.DataFrame, columns: list[str],
           params: dict) -> dict[str, float]:
    """Fit one model on ``train`` and return its metrics and bias on ``test``."""
    model = build_model(model_name).set_params(**params) if params else build_model(model_name)
    model.fit(train[columns], train[TARGET])
    y_pred = model.predict(test[columns])
    # Positive bias means the model predicts too low on average.
    return {**compute_metrics(test[TARGET], y_pred),
            "bias": float(np.mean(test[TARGET] - y_pred))}


def run_shift_tests(
    data: pd.DataFrame,
    feature_set: str = FINAL_FEATURE_SET,
    mlp_name: str = FINAL_MODEL_NAME,
    mlp_seeds: list[int] = MLP_SEEDS,
    mlp_params: dict | None = None,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Run leave-one-chord-out and leave-one-velocity-out for MLP and linear.

    Args:
        data: Dataset with physics features.
        feature_set: Feature set for both models.
        mlp_name: Registry name of the MLP (default: the frozen final MLP).
        mlp_seeds: ``random_state`` values for the repeated MLP runs; an
            empty list skips the MLP.
        mlp_params: Extra fixed MLP parameters (only used by the tests, to
            make fits fast).
        progress: Called with one line per held-out value; ``None`` for
            silence.

    Returns:
        One row per (shift, held-out value, model), columns
        :data:`SHIFT_COLUMNS`. For the MLP, metrics are means over seeds and
        ``rmse_sd_seeds`` is their spread; for linear ``n_runs`` is 1 and
        ``rmse_sd_seeds`` is NaN.
    """
    columns = get_feature_columns(feature_set)
    cv = cv_rows(data)
    rows = []
    for column, shift in SHIFTS.items():
        for value, regime, train, test in shift_splits(cv, column):
            runs = {"linear": [_score("linear", train, test, columns, {})]}
            if mlp_seeds:
                runs["mlp"] = [
                    _score(mlp_name, train, test, columns,
                           {MLP_SEED_KEY: seed, **(mlp_params or {})})
                    for seed in mlp_seeds
                ]
            for model, scores in runs.items():
                frame = pd.DataFrame(scores)
                rows.append({
                    "shift": shift,
                    "held_out": value,
                    "regime": regime,
                    "model": model,
                    "n_runs": len(frame),
                    "n_setups": test["setup_id"].nunique(),
                    "n_rows": len(test),
                    "rmse": frame["rmse"].mean(),
                    "rmse_sd_seeds": frame["rmse"].std(ddof=1) if len(frame) > 1 else np.nan,
                    "mae": frame["mae"].mean(),
                    "r2": frame["r2"].mean(),
                    "bias": frame["bias"].mean(),
                })
            if progress is not None:
                latest = rows[-len(runs):]  # the rows just added for this split
                summary = ", ".join(f"{r['model']} {r['rmse']:.2f} dB" for r in latest)
                progress(f"{shift}: held out {value:g} ({regime}) -> RMSE {summary}")
    return pd.DataFrame(rows, columns=SHIFT_COLUMNS)


def in_distribution_reference(tuning_summary: pd.DataFrame,
                              feature_set: str = FINAL_FEATURE_SET) -> dict[str, float]:
    """Nested-CV RMSE of each model, read from the saved tuning results.

    This is the accuracy on held-out *setups* whose chord and velocity were
    seen in training, so it is the baseline the shift results are compared
    with. It is read, not recomputed.

    Args:
        tuning_summary: Contents of ``results/tuning_summary.csv``.
        feature_set: Feature set of the comparison.

    Returns:
        ``{"mlp": ..., "linear": ...}`` mean RMSE over the 15 outer folds (dB).
    """
    row = tuning_summary.set_index("feature_set").loc[feature_set]
    return {"mlp": float(row["rmse_mean"]), "linear": float(row["linear_rmse_mean"])}
