"""Repeated cross-validation, metrics and error breakdowns for any model.

:func:`cross_validate` trains a model from :mod:`functions.models` on a
feature set from :mod:`functions.feature_sets` using the grouped folds from
:mod:`functions.make_folds`, repeated for every seed in
:data:`~functions.make_folds.CV_SEEDS`. It returns:

* per-fold scores (RMSE, MAE, R², fit and predict time), and
* out-of-fold (OOF) predictions: every cross-validation row predicted once
  per seed by the model that did not see it during training.

Rows in the held-out test set are dropped before anything is fitted, so this
module never trains on, tunes on or scores the test set.

RMSE is the main metric. :func:`oof_rmse_breakdown` splits the OOF RMSE by
chord length, free-stream velocity, angle-of-attack band and frequency band
to show where a model's errors come from. For linear models,
:func:`coefficient_summary` reports each input's effect in dB per standard
deviation and how stable it is across folds.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> from functions.physics_features import add_physics_features
    >>> from functions.evaluate import cross_validate, summarise, oof_rmse_breakdown
    >>> data = add_physics_features(load_airfoil_data())
    >>> result = cross_validate(data, "linear", "A")
    >>> summarise(result)            # doctest: +SKIP
    >>> oof_rmse_breakdown(result)   # doctest: +SKIP
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from functions.data_loader import TARGET
from functions.feature_sets import get_feature_columns
from functions.make_folds import CV_SEEDS, TEST_FOLD, assign_folds
from functions.models import build_model, coefficient_effects

#: Upper edges (deg, inclusive) of the angle-of-attack bands. Zero is its own
#: band because the symmetric NACA 0012 behaves differently at 0 deg.
ANGLE_BAND_EDGES_DEG = [0.0, 5.0, 10.0, 15.0]

#: Labels for the bands defined by :data:`ANGLE_BAND_EDGES_DEG`.
ANGLE_BAND_LABELS = ["0", "0-5", "5-10", "10-15", ">15"]

#: Lower edges (Hz, inclusive) of the frequency bands after the first. The
#: measured frequencies are 1/3-octave centres from 200 Hz to 20 kHz; these
#: edges give half-decade bands containing 5, 5, 5 and 6 frequencies.
FREQUENCY_BAND_EDGES_HZ = [630.0, 2000.0, 6300.0]

#: Labels for the bands defined by :data:`FREQUENCY_BAND_EDGES_HZ`.
FREQUENCY_BAND_LABELS = ["200-500 Hz", "630-1600 Hz", "2-5 kHz", "6.3-20 kHz"]

#: Columns the OOF RMSE is broken down by, in report order.
BREAKDOWN_COLUMNS = ["chord_length_m", "velocity_mps", "angle_band", "frequency_band"]


@dataclass
class CVResult:
    """Output of :func:`cross_validate`.

    Attributes:
        model: Name of the model that was evaluated.
        feature_set: Name of the feature set it was trained on.
        fold_scores: One row per (seed, fold) with columns ``seed``,
            ``fold``, ``n_train``, ``n_val``, ``rmse``, ``mae``, ``r2``,
            ``fit_time_s`` and ``predict_time_s``.
        oof: One row per (seed, cross-validation measurement) with columns
            ``seed``, ``fold``, ``row``, ``y_true``, ``y_pred`` and the
            columns in :data:`BREAKDOWN_COLUMNS`. ``row`` is the index of the
            measurement in the input DataFrame.
        coefficients: One row per (seed, fold, feature) with columns
            ``seed``, ``fold``, ``feature`` and ``effect_db_per_sd`` (see
            :func:`functions.models.coefficient_effects`). Empty for models
            without linear coefficients.
    """

    model: str
    feature_set: str
    fold_scores: pd.DataFrame
    oof: pd.DataFrame
    coefficients: pd.DataFrame


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return RMSE, MAE and R² for a set of predictions.

    Args:
        y_true: Measured sound pressure levels (dB).
        y_pred: Predicted sound pressure levels (dB), same length.

    Returns:
        A dict with keys ``rmse`` and ``mae`` (both in dB) and ``r2``.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def angle_band(angle_deg: pd.Series) -> pd.Series:
    """Label each angle of attack with its band.

    Bands are ``0`` (exactly zero), ``0-5`` (0 < a <= 5), ``5-10``,
    ``10-15`` and ``>15`` degrees.

    Args:
        angle_deg: Angles of attack in degrees.

    Returns:
        An ordered categorical Series of band labels, same index as the input.
    """
    bins = [-np.inf, *ANGLE_BAND_EDGES_DEG, np.inf]
    return pd.cut(angle_deg, bins=bins, labels=ANGLE_BAND_LABELS, right=True)


def frequency_band(frequency_hz: pd.Series) -> pd.Series:
    """Label each frequency with its half-decade band.

    Args:
        frequency_hz: Frequencies in Hz.

    Returns:
        An ordered categorical Series of band labels (see
        :data:`FREQUENCY_BAND_LABELS`), same index as the input.
    """
    bins = [-np.inf, *FREQUENCY_BAND_EDGES_HZ, np.inf]
    # right=False so each edge frequency starts the band above it.
    return pd.cut(frequency_hz, bins=bins, labels=FREQUENCY_BAND_LABELS, right=False)


def oof_frame(seed: int, fold: int, val: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Build the out-of-fold rows for one validation fold.

    Shared by :func:`cross_validate` and :func:`functions.tuning.nested_cv`
    so every model's OOF predictions have the same columns and can go
    through the same breakdowns and plots.

    Args:
        seed: CV seed of this fold.
        fold: Fold number.
        val: The validation rows (with the raw input and target columns).
        y_pred: Predictions for ``val``, in the same order.

    Returns:
        A DataFrame with the columns described in :attr:`CVResult.oof`.
    """
    return pd.DataFrame({
        "seed": seed,
        "fold": int(fold),
        "row": val.index,
        "y_true": val[TARGET].to_numpy(),
        "y_pred": y_pred,
        "chord_length_m": val["chord_length_m"].to_numpy(),
        "velocity_mps": val["velocity_mps"].to_numpy(),
        "angle_band": angle_band(val["attack_angle_deg"]).to_numpy(),
        "frequency_band": frequency_band(val["frequency_hz"]).to_numpy(),
    })


def concat_oof(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Join per-fold OOF frames from :func:`oof_frame` into one table.

    Args:
        parts: One DataFrame per fold.

    Returns:
        The concatenated rows, with the band columns as ordered categoricals.
    """
    oof = pd.concat(parts, ignore_index=True)
    # Restore the band ordering lost when the categoricals were concatenated.
    oof["angle_band"] = pd.Categorical(oof["angle_band"], ANGLE_BAND_LABELS, ordered=True)
    oof["frequency_band"] = pd.Categorical(
        oof["frequency_band"], FREQUENCY_BAND_LABELS, ordered=True
    )
    return oof


def cross_validate(
    data: pd.DataFrame,
    model_name: str,
    feature_set: str,
    cv_seeds: list[int] = CV_SEEDS,
) -> CVResult:
    """Run repeated grouped k-fold cross-validation for one model/feature set.

    For each seed, :func:`functions.make_folds.assign_folds` assigns folds,
    the test hold-out is dropped, and for each fold a fresh model is fitted on
    the other folds and predicts the held-out fold. Fit and predict are
    timed separately with :func:`time.perf_counter`.

    Args:
        data: Dataset from :func:`functions.data_loader.load_airfoil_data`.
            It must already contain any columns the feature set needs (call
            :func:`functions.physics_features.add_physics_features` for set
            ``B``).
        model_name: Key of :data:`functions.models.MODELS`.
        feature_set: Key of :data:`functions.feature_sets.FEATURE_SETS`.
        cv_seeds: Seeds for repeated CV. Defaults to
            :data:`functions.make_folds.CV_SEEDS` and should not be changed
            when comparing models.

    Returns:
        A :class:`CVResult` with per-fold scores, OOF predictions and, for
        linear models, per-fold coefficients.

    Raises:
        KeyError: If the model or feature set name is unknown, or ``data`` is
            missing a column the feature set needs.
    """
    columns = get_feature_columns(feature_set)
    fold_rows = []
    oof_parts = []
    coef_parts = []

    for seed in cv_seeds:
        folded = assign_folds(data, cv_seed=seed)
        # The test set is removed here and never used below.
        cv = folded[folded["fold"] != TEST_FOLD]

        for fold in sorted(cv["fold"].unique()):
            train = cv[cv["fold"] != fold]
            val = cv[cv["fold"] == fold]

            # A new model per fold, so its scalers are fitted on this fold's
            # training rows only.
            model = build_model(model_name)

            start = time.perf_counter()
            model.fit(train[columns], train[TARGET])
            fit_time = time.perf_counter() - start

            start = time.perf_counter()
            y_pred = model.predict(val[columns])
            predict_time = time.perf_counter() - start

            fold_rows.append({
                "seed": seed,
                "fold": int(fold),
                "n_train": len(train),
                "n_val": len(val),
                **compute_metrics(val[TARGET], y_pred),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            })
            oof_parts.append(oof_frame(seed, fold, val, y_pred))

            # Kept per fold so the spread shows how stable each effect is.
            effects = coefficient_effects(model, columns)
            if effects is not None:
                coef_parts.append(pd.DataFrame({
                    "seed": seed,
                    "fold": int(fold),
                    "feature": effects.index,
                    "effect_db_per_sd": effects.to_numpy(),
                }))

    oof = concat_oof(oof_parts)
    coefficients = (
        pd.concat(coef_parts, ignore_index=True)
        if coef_parts
        else pd.DataFrame(columns=["seed", "fold", "feature", "effect_db_per_sd"])
    )
    return CVResult(model_name, feature_set, pd.DataFrame(fold_rows), oof, coefficients)


def summarise(result: CVResult) -> dict[str, float]:
    """Summarise a CV run as one row of headline numbers.

    Each metric is the mean and sample standard deviation of the per-fold
    validation scores over all folds of all seeds (15 for 5 folds x 3
    seeds). Each fold scores a model on a batch of setups it never saw, so
    the standard deviation answers "how much does the score change from one
    batch of unseen setups to the next?". This is wider than the spread of
    seed-level averages, which would only show how much the random fold
    shuffle matters. Timings are also averaged over all folds.

    The mean of per-fold RMSEs is slightly different from the RMSE of all
    pooled OOF predictions, because the square root is taken per fold.

    Args:
        result: Output of :func:`cross_validate`.

    Returns:
        A dict with ``model``, ``feature_set``, ``n_folds`` (number of fold
        scores averaged), ``rmse``, ``rmse_std``, ``mae``, ``mae_std``,
        ``r2``, ``r2_std``, ``fit_time_s`` and ``predict_time_s`` (mean
        seconds per fold).
    """
    folds = result.fold_scores
    summary = {
        "model": result.model,
        "feature_set": result.feature_set,
        "n_folds": len(folds),
    }
    for metric in ["rmse", "mae", "r2"]:
        summary[metric] = folds[metric].mean()
        summary[f"{metric}_std"] = folds[metric].std(ddof=1)
    summary["fit_time_s"] = result.fold_scores["fit_time_s"].mean()
    summary["predict_time_s"] = result.fold_scores["predict_time_s"].mean()
    return summary


def oof_rmse_breakdown(result: CVResult) -> pd.DataFrame:
    """Break the OOF RMSE down by chord, velocity, angle band and frequency band.

    For each breakdown column and each group within it, RMSE is computed on
    that group's OOF predictions separately for every seed, then averaged
    over seeds.

    Args:
        result: Output of :func:`cross_validate`.

    Returns:
        A DataFrame with columns ``breakdown`` (which column was grouped on),
        ``group`` (the value or band label), ``n`` (rows per seed),
        ``rmse`` (mean over seeds, dB) and ``rmse_std`` (std over seeds).
    """
    oof = result.oof.assign(sq_err=(result.oof["y_true"] - result.oof["y_pred"]) ** 2)
    rows = []
    for column in BREAKDOWN_COLUMNS:
        # observed=True skips empty band categories.
        per_seed = oof.groupby([column, "seed"], observed=True).agg(
            n=("sq_err", "size"), mse=("sq_err", "mean")
        )
        per_seed["rmse"] = np.sqrt(per_seed["mse"])
        stats = per_seed.groupby(level=0, observed=True).agg(
            n=("n", "first"), rmse=("rmse", "mean"), rmse_std=("rmse", "std")
        )
        for group, row in stats.iterrows():
            rows.append({"breakdown": column, "group": group, **row.to_dict()})

    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    return out


def coefficient_summary(result: CVResult) -> pd.DataFrame:
    """Summarise a linear model's per-fold coefficients.

    Every (seed, fold) fit gives one coefficient per feature; this reports
    their mean and spread. A coefficient whose sign flips between folds, or
    whose std is comparable to its mean, is not reliably estimated and
    should not be given a physical interpretation.

    Args:
        result: Output of :func:`cross_validate`.

    Returns:
        A DataFrame indexed by feature (in feature-set order) with columns
        ``effect_db_per_sd`` (mean over fits), ``std``, ``min``, ``max`` and
        ``sign_agreement`` (fraction of fits whose sign matches the mean's).
        Empty if the model has no linear coefficients.
    """
    coefs = result.coefficients
    if coefs.empty:
        return pd.DataFrame(
            columns=["effect_db_per_sd", "std", "min", "max", "sign_agreement"]
        )

    grouped = coefs.groupby("feature", sort=False)["effect_db_per_sd"]
    out = grouped.agg(effect_db_per_sd="mean", std="std", min="min", max="max")
    mean_sign = np.sign(out["effect_db_per_sd"])
    out["sign_agreement"] = grouped.apply(
        lambda s: float(np.mean(np.sign(s) == mean_sign[s.name]))
    )
    return out
