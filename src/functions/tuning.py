"""Nested cross-validation to tune and fairly score the MLP.

Why nested CV?
    If the same folds were used both to pick the best hyperparameters and
    to report the score, the score would be optimistic: the chosen setting
    is partly the one that got lucky on those folds. Nested CV keeps the two
    jobs apart. An **inner** CV, run on the outer training rows only, picks
    the hyperparameters; the **outer** validation fold, which the inner CV
    never sees, then scores the chosen model. The outer scores are
    therefore honest estimates for unseen setups.

Procedure (:func:`nested_cv`), for each seed in
:data:`~functions.make_folds.CV_SEEDS` and each outer fold:

1. Drop the test hold-out (it is never used for training, tuning, early
   stopping or scoring here). Outer validation = rows of this fold; outer
   training = the other cross-validation rows.
2. On the outer training rows only, ``GridSearchCV`` tries every setting in
   :data:`PARAM_GRID` with an inner grouped 5-fold split
   (:func:`inner_splitter`). ``setup_id`` is passed as ``groups``, so no
   setup is ever split between inner training and inner validation.
3. The best setting is refitted on all outer training rows and scored on
   the outer validation fold (RMSE, MAE, R² in dB).

Known limitations (quotable in the report):

* **Early stopping uses random rows, not whole setups.** ``MLPRegressor``'s
  built-in early stopping holds back a random 10% of the rows it is given
  to decide when to stop. Rows from one setup are strongly correlated, so
  the held-back rows usually have near-identical neighbours from the same
  setup in the training part. The internal stopping check is therefore
  mildly optimistic, and training may stop a little later than ideal.
  This does **not** leak into the reported scores: the inner and outer
  validation folds are grouped by setup, and the early-stopping rows are
  taken only from the data being fitted, never from the fold being
  scored. ``MLPRegressor`` was deliberately not rewritten to fix this.
* **Inner folds are grouped but not stratified on chord.** ``GroupKFold``
  keeps setups together but does not balance chord lengths across inner
  folds (the outer folds are chord-stratified by
  :func:`functions.make_folds.assign_folds`).
* **Convergence.** Fits that stop at ``max_iter`` rather than by early
  stopping are counted (:func:`_hit_max_iter`), not hidden. Warnings
  raised in parallel worker processes cannot be caught in the main
  process, so the count is collected as an extra "score" instead.

Example:
    >>> from functions.tuning import nested_cv
    >>> result = nested_cv(data, "A", seeds=[42])   # doctest: +SKIP
    >>> result.outer_scores[["seed", "fold", "rmse"]]  # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas, and evaluating
# ``pd.DataFrame | None`` on the mock at import time would fail.
from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold

from functions.data_loader import TARGET
from functions.evaluate import CVResult, compute_metrics, concat_oof, oof_frame
from functions.feature_sets import get_feature_columns
from functions.make_folds import CV_SEEDS, TEST_FOLD, assign_folds
from functions.models import build_model

#: Registry name of the model being tuned.
MODEL_NAME = "mlp"

#: Name given to the tuned model in tables and figures.
TUNED_LABEL = "mlp (tuned)"

#: Grid keys. The model is TransformedTargetRegressor -> Pipeline ->
#: step ``"regressor"`` (MLPRegressor), hence the double ``regressor__`` prefix.
LAYERS_KEY = "regressor__regressor__hidden_layer_sizes"
ALPHA_KEY = "regressor__regressor__alpha"

#: The 35 settings tried in every inner search (5 architectures x 7 alphas).
#: Layer sizes are listed from smallest to largest network. (128, 128) and
#: alphas 1 and 10 were added after a one-seed trial run in which the
#: largest network (64, 64) and largest alpha (0.1) won most folds, i.e.
#: the best settings sat on the edge of the original 20-setting grid.
PARAM_GRID = {
    LAYERS_KEY: [(32,), (64,), (32, 32), (64, 64), (128, 128)],
    ALPHA_KEY: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
}

#: Number of inner folds used to choose hyperparameters.
N_INNER_SPLITS = 5

#: Seed for shuffling setups into inner folds. Fixed so tuning repeats.
INNER_CV_SEED = 0

#: GridSearchCV maximises its score, so RMSE is used negated.
SCORING = "neg_root_mean_squared_error"

#: Warn if a grid-edge value wins in more than this fraction of outer folds.
EDGE_WARNING_FRACTION = 0.5

#: Columns of :attr:`TuningResult.outer_scores`.
OUTER_COLUMNS = [
    "feature_set", "seed", "fold", "n_train", "n_val",
    "hidden_layer_sizes", "alpha", "inner_best_rmse",
    "rmse", "mae", "r2", "n_iter", "refit_hit_max_iter",
    "inner_fits_hit_max_iter", "tuning_time_s", "fit_time_s", "predict_time_s",
]

#: Columns of :attr:`TuningResult.inner_grid`.
INNER_COLUMNS = [
    "feature_set", "seed", "fold", "hidden_layer_sizes", "alpha",
    "mean_inner_rmse", "std_inner_rmse", "rank", "inner_fits_hit_max_iter",
]


def _hit_max_iter(estimator: TransformedTargetRegressor, X, y) -> float:
    """Scorer that returns 1.0 if the fitted MLP stopped at ``max_iter``.

    Used as an extra GridSearchCV "score" so the number of fits that ran
    out of epochs (instead of stopping early) is reported, even when the
    fits run in parallel worker processes.
    """
    mlp = estimator.regressor_.named_steps["regressor"]
    return float(mlp.n_iter_ >= mlp.max_iter)


def inner_splitter(n_splits: int = N_INNER_SPLITS) -> GroupKFold:
    """Return the grouped splitter used for the inner CV.

    ``shuffle=True`` with a fixed seed randomises which setups go in which
    inner fold, reproducibly. Groups (``setup_id``) are passed at fit time.

    Args:
        n_splits: Number of inner folds.

    Returns:
        A :class:`~sklearn.model_selection.GroupKFold`.
    """
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=INNER_CV_SEED)


@dataclass
class TuningResult:
    """Output of :func:`nested_cv` for one feature set.

    Attributes:
        feature_set: Feature set the MLP was tuned on.
        outer_scores: One row per (seed, outer fold), columns
            :data:`OUTER_COLUMNS`: the chosen setting, inner-CV best RMSE,
            outer RMSE/MAE/R² (dB), epochs run, max-iter hits and timings.
        inner_grid: One row per (seed, outer fold, setting), columns
            :data:`INNER_COLUMNS`: mean and std inner-CV RMSE from
            ``cv_results_``.
        cv_result: The outer predictions as a
            :class:`functions.evaluate.CVResult` (model ``"mlp (tuned)"``),
            so :func:`functions.evaluate.summarise`, the breakdowns and the
            plots in :mod:`functions.plot_results` work on it unchanged.
        selected_models: The refitted best model of each outer fold, keyed
            by ``(seed, fold)``; used for the loss-curve figure.
        param_grid: The grid that was searched.
        n_inner_splits: Number of inner folds used.
    """

    feature_set: str
    outer_scores: pd.DataFrame
    inner_grid: pd.DataFrame
    cv_result: CVResult
    selected_models: dict[tuple[int, int], TransformedTargetRegressor]
    param_grid: dict
    n_inner_splits: int = N_INNER_SPLITS


def nested_cv(
    data: pd.DataFrame,
    feature_set: str,
    seeds: list[int] = CV_SEEDS,
    param_grid: dict = PARAM_GRID,
    n_inner_splits: int = N_INNER_SPLITS,
    n_jobs: int = -1,
    progress: Callable[[str], None] | None = print,
) -> TuningResult:
    """Tune the MLP by inner grouped CV and score it on the outer folds.

    See the module docstring for the procedure and its limitations.

    Args:
        data: Dataset with every column the feature set needs (see
            :func:`functions.evaluate.cross_validate`).
        feature_set: Key of :data:`functions.feature_sets.FEATURE_SETS`.
        seeds: Outer CV seeds to run; a subset of
            :data:`~functions.make_folds.CV_SEEDS` gives a quicker run.
        param_grid: Settings to search (GridSearchCV format).
        n_inner_splits: Number of inner folds.
        n_jobs: Parallel jobs for GridSearchCV (-1 = all CPU cores).
        progress: Called with one line of text per outer fold; ``None`` for
            silence.

    Returns:
        A :class:`TuningResult`.
    """
    columns = get_feature_columns(feature_set)
    outer_rows, inner_rows, oof_parts, selected = [], [], [], {}

    for seed in seeds:
        folded = assign_folds(data, cv_seed=seed)
        # Remove the test hold-out before anything else, so it can never
        # reach tuning, early stopping or scoring.
        cv = folded[folded["fold"] != TEST_FOLD]

        for fold in sorted(cv["fold"].unique()):
            # Outer split: the validation fold is kept completely out of
            # tuning, so its score is a fair test of the chosen setting.
            train = cv[cv["fold"] != fold]
            val = cv[cv["fold"] == fold]

            search = GridSearchCV(
                build_model(MODEL_NAME),
                param_grid,
                scoring={"rmse": SCORING, "hit_max_iter": _hit_max_iter},
                refit="rmse",  # refit the best setting on all outer-training rows
                cv=inner_splitter(n_inner_splits),
                n_jobs=n_jobs,
            )
            start = time.perf_counter()
            # groups=setup_id makes the inner folds keep setups together.
            search.fit(train[columns], train[TARGET], groups=train["setup_id"])
            tuning_time = time.perf_counter() - start

            best = search.best_estimator_
            start = time.perf_counter()
            y_pred = best.predict(val[columns])
            predict_time = time.perf_counter() - start

            mlp = best.regressor_.named_steps["regressor"]
            res = search.cv_results_
            # Mean of a 0/1 score over the inner folds x number of folds =
            # number of inner fits that ran out of epochs.
            hits = np.rint(res["mean_test_hit_max_iter"] * n_inner_splits).astype(int)
            layers = search.best_params_[LAYERS_KEY]
            alpha = search.best_params_[ALPHA_KEY]

            outer_rows.append({
                "feature_set": feature_set,
                "seed": seed,
                "fold": int(fold),
                "n_train": len(train),
                "n_val": len(val),
                "hidden_layer_sizes": str(layers),
                "alpha": alpha,
                "inner_best_rmse": -search.best_score_,
                **compute_metrics(val[TARGET], y_pred),
                "n_iter": mlp.n_iter_,
                "refit_hit_max_iter": bool(mlp.n_iter_ >= mlp.max_iter),
                "inner_fits_hit_max_iter": int(hits.sum()),
                "tuning_time_s": tuning_time,
                "fit_time_s": search.refit_time_,
                "predict_time_s": predict_time,
            })
            for i, params in enumerate(res["params"]):
                inner_rows.append({
                    "feature_set": feature_set,
                    "seed": seed,
                    "fold": int(fold),
                    "hidden_layer_sizes": str(params[LAYERS_KEY]),
                    "alpha": params[ALPHA_KEY],
                    # Negate back: GridSearchCV stores -RMSE.
                    "mean_inner_rmse": -res["mean_test_rmse"][i],
                    "std_inner_rmse": res["std_test_rmse"][i],
                    "rank": int(res["rank_test_rmse"][i]),
                    "inner_fits_hit_max_iter": int(hits[i]),
                })
            oof_parts.append(oof_frame(seed, fold, val, y_pred))
            selected[(seed, int(fold))] = best

            if progress is not None:
                progress(
                    f"[{feature_set}] seed {seed} fold {fold}: best {layers} "
                    f"alpha={alpha:g}  inner RMSE {-search.best_score_:.3f}  "
                    f"outer RMSE {outer_rows[-1]['rmse']:.3f} dB  "
                    f"({tuning_time:.1f} s)"
                )

    outer = pd.DataFrame(outer_rows, columns=OUTER_COLUMNS)
    cv_result = CVResult(
        model=TUNED_LABEL,
        feature_set=feature_set,
        fold_scores=outer,
        oof=concat_oof(oof_parts),
        coefficients=pd.DataFrame(columns=["seed", "fold", "feature", "effect_db_per_sd"]),
    )
    return TuningResult(
        feature_set=feature_set,
        outer_scores=outer,
        inner_grid=pd.DataFrame(inner_rows, columns=INNER_COLUMNS),
        cv_result=cv_result,
        selected_models=selected,
        param_grid=param_grid,
        n_inner_splits=n_inner_splits,
    )


def selection_counts(result: TuningResult) -> pd.DataFrame:
    """Count how often each grid setting was chosen across outer folds.

    Args:
        result: Output of :func:`nested_cv`.

    Returns:
        One row per grid setting (including settings never chosen), with
        columns ``feature_set``, ``hidden_layer_sizes``, ``alpha``,
        ``n_folds_won`` and ``fraction_won``, most-chosen first.
    """
    grid = pd.MultiIndex.from_product(
        [[str(h) for h in result.param_grid[LAYERS_KEY]], result.param_grid[ALPHA_KEY]],
        names=["hidden_layer_sizes", "alpha"],
    )
    counts = (
        result.outer_scores.groupby(["hidden_layer_sizes", "alpha"]).size()
        .reindex(grid, fill_value=0).rename("n_folds_won").reset_index()
    )
    counts["fraction_won"] = counts["n_folds_won"] / len(result.outer_scores)
    counts.insert(0, "feature_set", result.feature_set)
    return counts.sort_values("n_folds_won", ascending=False, kind="stable").reset_index(drop=True)


def seed_level_spread(outer_scores: pd.DataFrame) -> dict[str, float]:
    """Standard deviation of the per-seed mean scores.

    Each seed's 5 outer-fold scores are averaged, then the SD is taken over
    the seeds. This answers "how much does the result depend on how setups
    were shuffled into folds?", a different question from the 15-fold SD
    reported by :func:`functions.evaluate.summarise` ("how much does the
    score change from one batch of unseen setups to the next?").

    Args:
        outer_scores: :attr:`TuningResult.outer_scores` or any per-fold
            table with ``seed``, ``rmse``, ``mae`` and ``r2`` columns.

    Returns:
        ``{"rmse_sd_seeds", "mae_sd_seeds", "r2_sd_seeds", "n_seeds"}``.
        The SDs are NaN when only one seed was run.
    """
    per_seed = outer_scores.groupby("seed")[["rmse", "mae", "r2"]].mean()
    out = {f"{m}_sd_seeds": per_seed[m].std(ddof=1) for m in ["rmse", "mae", "r2"]}
    out["n_seeds"] = len(per_seed)
    return out


def paired_vs_linear(mlp_scores: pd.DataFrame, linear_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare MLP and linear RMSE fold by fold, on identical folds.

    Both models are scored on exactly the same outer folds, so the fair
    comparison is the difference within each fold, not two overlapping
    mean ± SD ranges.

    Args:
        mlp_scores: Per-fold MLP scores with ``seed``, ``fold``, ``rmse``.
        linear_scores: Per-fold linear scores with the same columns, e.g.
            :attr:`functions.evaluate.CVResult.fold_scores`.

    Returns:
        One row per (seed, fold) with ``rmse_mlp``, ``rmse_linear`` and
        ``rmse_diff`` (MLP minus linear; negative means the MLP was better).

    Raises:
        ValueError: If the two tables do not cover the same (seed, fold) pairs.
    """
    keys = ["seed", "fold"]
    merged = mlp_scores[keys + ["rmse"]].merge(
        linear_scores[keys + ["rmse"]], on=keys, how="outer",
        suffixes=("_mlp", "_linear"), indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("MLP and linear scores do not cover the same seeds and folds")
    merged = merged.drop(columns="_merge").sort_values(keys).reset_index(drop=True)
    merged["rmse_diff"] = merged["rmse_mlp"] - merged["rmse_linear"]
    return merged


def paired_summary(paired: pd.DataFrame) -> dict[str, float]:
    """Summarise a :func:`paired_vs_linear` table.

    Args:
        paired: Output of :func:`paired_vs_linear`.

    Returns:
        ``rmse_diff_mean`` and ``rmse_diff_sd`` (dB, MLP minus linear),
        ``mlp_wins`` (folds where the MLP had lower RMSE) and ``n_folds``.
    """
    return {
        "rmse_diff_mean": paired["rmse_diff"].mean(),
        "rmse_diff_sd": paired["rmse_diff"].std(ddof=1),
        "mlp_wins": int((paired["rmse_diff"] < 0).sum()),
        "n_folds": len(paired),
    }


def edge_of_grid_warnings(result: TuningResult,
                          threshold: float = EDGE_WARNING_FRACTION) -> list[str]:
    """Warn when the winning setting is often at the edge of the grid.

    If the best alpha or layer size is the smallest or largest value tried
    in most folds, the true optimum may lie outside the grid, which would
    then need extending. Parameters with a single grid value are skipped.

    Args:
        result: Output of :func:`nested_cv`.
        threshold: Fraction of outer folds above which to warn.

    Returns:
        Warning messages (empty if none apply).
    """
    outer = result.outer_scores
    checks = {
        "hidden_layer_sizes": [str(h) for h in result.param_grid[LAYERS_KEY]],
        "alpha": list(result.param_grid[ALPHA_KEY]),
    }
    messages = []
    for column, values in checks.items():
        if len(values) < 2:
            continue
        for edge, value in (("smallest", values[0]), ("largest", values[-1])):
            fraction = (outer[column] == value).mean()
            if fraction > threshold:
                messages.append(
                    f"[{result.feature_set}] {column} = {value} (the {edge} grid value) "
                    f"won in {fraction:.0%} of outer folds; the grid may need extending."
                )
    return messages


def refit_outer_predictions(
    data: pd.DataFrame,
    outer_scores: pd.DataFrame,
    feature_set: str,
    extra_params: dict | None = None,
    atol: float = 1e-6,
) -> CVResult:
    """Rebuild the tuned MLP's out-of-fold predictions from saved results.

    ``tuning_outer_scores.csv`` records which setting won each outer fold
    but not the row-by-row predictions. Because the MLP has a fixed
    ``random_state`` and is trained on exactly the same rows, refitting the
    winning setting reproduces each fold's model, and so its predictions,
    exactly. This takes seconds instead of re-running the whole nested CV.
    Each refitted fold's RMSE is checked against the saved value as proof.

    Args:
        data: Dataset with every column the feature set needs.
        outer_scores: The saved :attr:`TuningResult.outer_scores` table
            (e.g. read from ``results/tuning_outer_scores.csv``).
        feature_set: Which feature set's rows to rebuild.
        extra_params: Further fixed model parameters that were in the search
            grid (none for the standard grid; used by the tests).
        atol: Largest allowed difference between refitted and saved RMSE.

    Returns:
        A :class:`~functions.evaluate.CVResult` (model ``"mlp (tuned)"``)
        with per-fold scores and OOF predictions.

    Raises:
        ValueError: If no rows match ``feature_set``, or a refitted fold's
            RMSE differs from the saved one (the saved results then came from
            different code or data).
    """
    rows = outer_scores[outer_scores["feature_set"] == feature_set]
    if rows.empty:
        raise ValueError(f"No saved outer scores for feature set {feature_set!r}")
    columns = get_feature_columns(feature_set)
    fold_rows, oof_parts = [], []

    for seed, group in rows.groupby("seed", sort=False):
        folded = assign_folds(data, cv_seed=int(seed))
        cv = folded[folded["fold"] != TEST_FOLD]  # the test hold-out is never used
        for _, saved in group.iterrows():
            train = cv[cv["fold"] != saved["fold"]]
            val = cv[cv["fold"] == saved["fold"]]
            # The CSV stores layer sizes as text such as "(128, 128)".
            params = {
                LAYERS_KEY: ast.literal_eval(str(saved["hidden_layer_sizes"])),
                ALPHA_KEY: float(saved["alpha"]),
                **(extra_params or {}),
            }
            model = build_model(MODEL_NAME).set_params(**params)
            start = time.perf_counter()
            model.fit(train[columns], train[TARGET])
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            y_pred = model.predict(val[columns])
            predict_time = time.perf_counter() - start

            scores = compute_metrics(val[TARGET], y_pred)
            if abs(scores["rmse"] - saved["rmse"]) > atol:
                raise ValueError(
                    f"Refit of {feature_set} seed {seed} fold {saved['fold']} gives RMSE "
                    f"{scores['rmse']:.6f}, saved {saved['rmse']:.6f}; results are out of date"
                )
            fold_rows.append({
                "seed": int(seed), "fold": int(saved["fold"]),
                "n_train": len(train), "n_val": len(val), **scores,
                "fit_time_s": fit_time, "predict_time_s": predict_time,
            })
            oof_parts.append(oof_frame(int(seed), int(saved["fold"]), val, y_pred))

    return CVResult(
        model=TUNED_LABEL,
        feature_set=feature_set,
        fold_scores=pd.DataFrame(fold_rows),
        oof=concat_oof(oof_parts),
        coefficients=pd.DataFrame(columns=["seed", "fold", "feature", "effect_db_per_sd"]),
    )
