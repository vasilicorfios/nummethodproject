"""Building blocks for the single final hold-out evaluation (``run_final.py``).

The test hold-out (``fold == TEST_FOLD``) is used here for the first and only
time. Three models are trained once on **all** cross-validation rows and then
predict the hold-out rows once:

* ``mlp_final`` (:data:`functions.models.FINAL_MLP`): the selected model;
  the acceptance criteria apply to it.
* ``mlp_runner_up`` (:data:`functions.models.RUNNER_UP_MLP`): a cheaper
  alternative, reported for comparison only.
* ``linear``: the baseline.

All three use :data:`functions.models.FINAL_FEATURE_SET` and are built by
:func:`functions.models.build_model`, so no setting is retyped and all
scaling is fitted on the training rows only.

Hold-out predictions are stored in the same out-of-fold table format as the
cross-validation results (:class:`functions.evaluate.CVResult`, with
``seed = TEST_SEED`` and ``fold = TEST_FOLD``), so the existing breakdown and
plotting functions work on them unchanged.
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor

from functions.acceptance import acceptance_table
from functions.data_loader import TARGET
from functions.evaluate import (
    BREAKDOWN_COLUMNS,
    CVResult,
    compute_metrics,
    concat_oof,
    oof_frame,
    oof_rmse_breakdown,
    per_setup_rmse,
)
from functions.feature_sets import get_feature_columns
from functions.make_folds import TEST_FOLD, TEST_SEED, assign_folds
from functions.models import (
    FINAL_FEATURE_SET,
    FINAL_MODEL_NAME,
    RUNNER_UP_MLP,
    RUNNER_UP_MODEL_NAME,
    build_model,
)

#: Models evaluated on the hold-out, in report order, with short labels.
HOLDOUT_MODELS = {
    "linear": "linear",
    RUNNER_UP_MODEL_NAME: "runner-up MLP",
    FINAL_MODEL_NAME: "final MLP",
}

#: The selected model; the only one assessed against the acceptance criteria.
SELECTED_MODEL = FINAL_MODEL_NAME

#: Columns describing a hold-out setup.
SETUP_INFO = ["setup_id", "chord_length_m", "velocity_mps", "attack_angle_deg", "disp_thickness_m"]


def split_train_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into all cross-validation rows and the hold-out rows.

    The test hold-out depends only on ``TEST_SEED``, not on the CV seed, so
    the default :func:`functions.make_folds.assign_folds` call gives it.

    Args:
        data: Dataset with physics features.

    Returns:
        ``(train, holdout)``, both with ``setup_id`` and ``fold`` columns.

    Raises:
        RuntimeError: If the split fails :func:`check_split`.
    """
    folded = assign_folds(data)
    train = folded[folded["fold"] != TEST_FOLD]
    holdout = folded[folded["fold"] == TEST_FOLD]
    check_split(train, holdout, folded)
    return train, holdout


def check_split(train: pd.DataFrame, holdout: pd.DataFrame, folded: pd.DataFrame) -> None:
    """Refuse to continue unless the train/hold-out split is clean.

    Checks that no setup is in both parts, that the hold-out has exactly the
    rows and setups :func:`functions.make_folds.assign_folds` marked as test,
    and that every row is in one part or the other. These are raised errors,
    not ``assert`` statements, so they cannot be switched off.

    Args:
        train: Training rows.
        holdout: Hold-out rows.
        folded: The full output of ``assign_folds``.

    Raises:
        RuntimeError: Describing the first problem found.
    """
    shared = set(train["setup_id"]) & set(holdout["setup_id"])
    if shared:
        raise RuntimeError(f"Setups in both training and hold-out rows: {sorted(shared)}")
    expected = folded[folded["fold"] == TEST_FOLD]
    if len(holdout) != len(expected) or set(holdout.index) != set(expected.index):
        raise RuntimeError("Hold-out rows do not match the rows assign_folds marked as test")
    if holdout["setup_id"].nunique() != expected["setup_id"].nunique():
        raise RuntimeError("Hold-out setup count does not match assign_folds")
    if len(train) + len(holdout) != len(folded):
        raise RuntimeError("Training and hold-out rows do not add up to the whole dataset")


@dataclass
class HoldoutFit:
    """One model trained on all CV rows and its hold-out predictions.

    Attributes:
        model_name: Registry name, e.g. ``"mlp_final"``.
        model: The fitted pipeline.
        y_pred: Predictions for the hold-out rows, in hold-out row order.
        fit_time_s: Training time (one fit, :func:`time.perf_counter`).
        predict_time_s: Time to predict all hold-out rows.
        n_iter: Epochs run (MLPs only, else ``None``).
        hit_max_iter: Whether training stopped at ``max_iter`` rather than by
            early stopping (MLPs only, else ``None``).
    """

    model_name: str
    model: TransformedTargetRegressor
    y_pred: np.ndarray
    fit_time_s: float
    predict_time_s: float
    n_iter: int | None
    hit_max_iter: bool | None


def fit_and_predict(model_name: str, train: pd.DataFrame, holdout: pd.DataFrame,
                    feature_set: str = FINAL_FEATURE_SET) -> HoldoutFit:
    """Train one model once on the training rows, then predict the hold-out once.

    Args:
        model_name: Key of :data:`functions.models.MODELS`.
        train: All cross-validation rows.
        holdout: The hold-out rows.
        feature_set: Feature set to use.

    Returns:
        A :class:`HoldoutFit`.
    """
    columns = get_feature_columns(feature_set)
    model = build_model(model_name)
    start = time.perf_counter()
    model.fit(train[columns], train[TARGET])
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(holdout[columns])
    predict_time = time.perf_counter() - start

    estimator = model.regressor_.named_steps["regressor"]
    is_mlp = isinstance(estimator, MLPRegressor)
    return HoldoutFit(
        model_name=model_name,
        model=model,
        y_pred=np.asarray(y_pred),
        fit_time_s=fit_time,
        predict_time_s=predict_time,
        n_iter=int(estimator.n_iter_) if is_mlp else None,
        hit_max_iter=bool(estimator.n_iter_ >= estimator.max_iter) if is_mlp else None,
    )


def as_result(fit: HoldoutFit, train: pd.DataFrame, holdout: pd.DataFrame,
              feature_set: str = FINAL_FEATURE_SET) -> CVResult:
    """Wrap hold-out predictions as a CVResult so existing tools can use them.

    Args:
        fit: Output of :func:`fit_and_predict`.
        train: The training rows (for the row count).
        holdout: The hold-out rows.
        feature_set: Feature set used.

    Returns:
        A :class:`~functions.evaluate.CVResult` whose single "fold" is the
        hold-out, labelled with the model's short name.
    """
    scores = {
        "seed": TEST_SEED, "fold": TEST_FOLD, "n_train": len(train), "n_val": len(holdout),
        **compute_metrics(holdout[TARGET], fit.y_pred),
        "fit_time_s": fit.fit_time_s, "predict_time_s": fit.predict_time_s,
    }
    return CVResult(
        model=HOLDOUT_MODELS[fit.model_name],
        feature_set=feature_set,
        fold_scores=pd.DataFrame([scores]),
        oof=concat_oof([oof_frame(TEST_SEED, TEST_FOLD, holdout, fit.y_pred)]),
        coefficients=pd.DataFrame(columns=["seed", "fold", "feature", "effect_db_per_sd"]),
    )


def holdout_metrics(fits: list[HoldoutFit], holdout: pd.DataFrame) -> pd.DataFrame:
    """Headline hold-out metrics, one row per model.

    Args:
        fits: Outputs of :func:`fit_and_predict`.
        holdout: The hold-out rows.

    Returns:
        Columns ``model``, ``label``, ``selected``, ``rmse``, ``mae``,
        ``r2``, ``bias`` (mean of measured − predicted, dB; positive means
        the model is too low), ``n_setups``, ``n_rows``, ``fit_time_s``,
        ``predict_time_s``, ``n_iter`` and ``hit_max_iter``.
    """
    rows = []
    for fit in fits:
        rows.append({
            "model": fit.model_name,
            "label": HOLDOUT_MODELS[fit.model_name],
            "selected": fit.model_name == SELECTED_MODEL,
            **compute_metrics(holdout[TARGET], fit.y_pred),
            "bias": float(np.mean(holdout[TARGET].to_numpy() - fit.y_pred)),
            "n_setups": holdout["setup_id"].nunique(),
            "n_rows": len(holdout),
            "fit_time_s": fit.fit_time_s,
            "predict_time_s": fit.predict_time_s,
            "n_iter": fit.n_iter,
            "hit_max_iter": fit.hit_max_iter,
        })
    return pd.DataFrame(rows)


def holdout_breakdown(results: dict[str, CVResult], holdout: pd.DataFrame) -> pd.DataFrame:
    """Hold-out RMSE by chord, velocity, angle band and frequency band.

    Reuses :func:`functions.evaluate.oof_rmse_breakdown` and adds how many
    hold-out setups fall in each group.

    Args:
        results: Model name -> :func:`as_result` output.
        holdout: The hold-out rows.

    Returns:
        Columns ``model``, ``breakdown``, ``group``, ``n_setups``, ``n``
        (rows) and ``rmse``.
    """
    parts = []
    for name, result in results.items():
        bd = oof_rmse_breakdown(result)[["breakdown", "group", "n", "rmse"]]
        oof = result.oof.assign(setup_id=holdout.loc[result.oof["row"], "setup_id"].to_numpy())
        setups = pd.concat([
            oof.groupby(col, observed=True)["setup_id"].nunique().rename("n_setups")
            .reset_index().rename(columns={col: "group"}).assign(breakdown=col)
            for col in BREAKDOWN_COLUMNS
        ])
        parts.append(bd.merge(setups, on=["breakdown", "group"]).assign(model=name))
    out = pd.concat(parts, ignore_index=True)
    return out[["model", "breakdown", "group", "n_setups", "n", "rmse"]]


def holdout_setups(holdout: pd.DataFrame) -> pd.DataFrame:
    """The hold-out setups and how many rows (frequencies) each has.

    Args:
        holdout: The hold-out rows.

    Returns:
        One row per setup: :data:`SETUP_INFO` plus ``n_rows``, ordered by
        chord, velocity and angle.
    """
    info = holdout.groupby("setup_id").agg(
        chord_length_m=("chord_length_m", "first"), velocity_mps=("velocity_mps", "first"),
        attack_angle_deg=("attack_angle_deg", "first"), disp_thickness_m=("disp_thickness_m", "first"),
        n_rows=("spl_db", "size"),
    ).reset_index()
    return info.sort_values(["chord_length_m", "velocity_mps", "attack_angle_deg"]).reset_index(drop=True)


def setup_rmse(results: dict[str, CVResult], data: pd.DataFrame) -> pd.DataFrame:
    """RMSE and bias of every hold-out setup for every model, ranked worst first.

    Args:
        results: Model name -> :func:`as_result` output.
        data: The dataset.

    Returns:
        Long table: ``model``, ``rank`` (1 = worst for that model) and the
        columns of :func:`functions.evaluate.per_setup_rmse`.
    """
    parts = []
    for name, result in results.items():
        table = per_setup_rmse(result, data).drop(columns="rmse_sd_seeds")
        table.insert(0, "rank", range(1, len(table) + 1))
        table.insert(0, "model", name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def paired_setups(setups: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Per-setup RMSE difference ``model_a − model_b`` on identical setups.

    Args:
        setups: Output of :func:`setup_rmse`.
        model_a: First model (e.g. the final MLP).
        model_b: Model it is compared with.

    Returns:
        One row per setup with ``rmse_a``, ``rmse_b`` and ``rmse_diff``
        (negative means ``model_a`` was better), plus ``comparison``.

    Raises:
        ValueError: If the two models were not scored on the same setups.
    """
    a = setups[setups["model"] == model_a].set_index("setup_id")["rmse"]
    b = setups[setups["model"] == model_b].set_index("setup_id")["rmse"]
    if set(a.index) != set(b.index):
        raise ValueError(f"{model_a} and {model_b} were not scored on the same setups")
    paired = pd.DataFrame({"rmse_a": a, "rmse_b": b.loc[a.index]}).reset_index()
    paired["rmse_diff"] = paired["rmse_a"] - paired["rmse_b"]
    paired.insert(0, "comparison", f"{model_a} - {model_b}")
    return paired


def paired_summary(paired: pd.DataFrame) -> dict[str, float]:
    """Mean ± SD of the per-setup differences and the number of wins.

    Args:
        paired: Output of :func:`paired_setups`.

    Returns:
        ``comparison``, ``n_setups``, ``mean_diff``, ``sd_diff``,
        ``a_wins`` and ``b_wins`` (setups where that model's RMSE was lower).
    """
    return {
        "comparison": paired["comparison"].iat[0],
        "n_setups": len(paired),
        "mean_diff": paired["rmse_diff"].mean(),
        "sd_diff": paired["rmse_diff"].std(ddof=1),
        "a_wins": int((paired["rmse_diff"] < 0).sum()),
        "b_wins": int((paired["rmse_diff"] > 0).sum()),
    }


def cv_estimates(tuning_summary: pd.DataFrame, cv_summary: pd.DataFrame,
                 inner_grid: pd.DataFrame, feature_set: str = FINAL_FEATURE_SET) -> pd.DataFrame:
    """Each model's cross-validation RMSE estimate, read from saved results.

    * Final MLP: nested-CV outer-fold mean ± SD (``tuning_summary.csv``).
      This scores the whole tuning procedure on unseen setups.
    * Linear: repeated grouped CV mean ± SD (``cv_summary.csv``).
    * Runner-up MLP: it has no outer-fold estimate of its own, so its mean
      **inner**-CV RMSE (``tuning_inner_grid.csv``, averaged over all seeds
      and outer folds) is used, with the SD over outer folds. It is labelled
      as an inner-CV estimate: inner scores were used for selection and are
      slightly optimistic.

    Args:
        tuning_summary: Contents of ``results/tuning_summary.csv``.
        cv_summary: Contents of ``results/cv_summary.csv``.
        inner_grid: Contents of ``results/tuning_inner_grid.csv``.
        feature_set: Feature set of the models.

    Returns:
        One row per model: ``model``, ``estimate``, ``cv_rmse``, ``cv_sd`` and
        ``n_folds``.
    """
    tuned = tuning_summary.set_index("feature_set").loc[feature_set]
    linear = cv_summary[(cv_summary["model"] == "linear") & (cv_summary["feature_set"] == feature_set)].iloc[0]
    grid = inner_grid[(inner_grid["feature_set"] == feature_set)
                      & (inner_grid["hidden_layer_sizes"] == str(RUNNER_UP_MLP["hidden_layer_sizes"]))
                      & np.isclose(inner_grid["alpha"], RUNNER_UP_MLP["alpha"])]
    return pd.DataFrame([
        {"model": "linear", "estimate": "repeated grouped CV",
         "cv_rmse": linear["rmse"], "cv_sd": linear["rmse_std"], "n_folds": int(linear["n_folds"])},
        {"model": RUNNER_UP_MODEL_NAME, "estimate": "inner CV (selection score)",
         "cv_rmse": grid["mean_inner_rmse"].mean(), "cv_sd": grid["mean_inner_rmse"].std(ddof=1),
         "n_folds": len(grid)},
        {"model": FINAL_MODEL_NAME, "estimate": "nested CV (outer folds)",
         "cv_rmse": tuned["rmse_mean"], "cv_sd": tuned["rmse_sd_folds"], "n_folds": int(tuned["n_folds"])},
    ])


def evaluate_acceptance(paired_cv: pd.DataFrame, tuning_summary: pd.DataFrame,
                        physics_summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every acceptance criterion for the selected (final) model.

    Thresholds come unchanged from :mod:`functions.acceptance`. Only the
    final MLP and linear hold-out RMSEs enter; the runner-up is not assessed.

    Args:
        paired_cv: Contents of ``results/tuning_paired_vs_linear.csv``.
        tuning_summary: Contents of ``results/tuning_summary.csv``.
        physics_summary: Contents of ``results/physics_speed_order_summary.csv``.
        metrics: Output of :func:`holdout_metrics`.

    Returns:
        The acceptance table with every criterion PASS or FAIL.
    """
    rmse = metrics.set_index("model")["rmse"]
    mlp = physics_summary.set_index("model").loc["mlp"]
    return acceptance_table(
        paired_cv, tuning_summary,
        physics_pair_fraction=mlp["pairs_correct"] / mlp["pairs_scored"],
        holdout={"mlp_rmse": float(rmse[SELECTED_MODEL]), "linear_rmse": float(rmse["linear"])},
    )


def predictions_table(holdout: pd.DataFrame, fits: list[HoldoutFit],
                      feature_set: str = FINAL_FEATURE_SET) -> pd.DataFrame:
    """Row-level hold-out predictions of every model.

    Args:
        holdout: The hold-out rows.
        fits: Outputs of :func:`fit_and_predict`.
        feature_set: Feature set used (its columns are included as inputs).

    Returns:
        One row per hold-out measurement: ``row``, ``setup_id``, the input
        columns, ``measured_spl_db`` and ``pred_<model>`` for each model.
    """
    columns = get_feature_columns(feature_set)
    table = holdout[["setup_id", *columns]].copy()
    table.insert(0, "row", holdout.index)
    table["measured_spl_db"] = holdout[TARGET].to_numpy()
    for fit in fits:
        table[f"pred_{fit.model_name}"] = fit.y_pred
    return table.reset_index(drop=True)
