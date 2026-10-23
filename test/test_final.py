"""Tests for the final hold-out evaluation (:mod:`functions.final`, ``run_final``).

None of these tests score the real hold-out: that happens only in the single
``python src/run_final.py`` run. The pipeline test uses one cross-validation
fold as a stand-in "hold-out", and the rest use fake data or a temporary
results directory.
"""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import functions.final as final  # noqa: E402
import run_final  # noqa: E402
from functions.acceptance import PENDING_HOLDOUT, PENDING_PHYSICS  # noqa: E402
from functions.make_folds import TEST_FOLD, assign_folds  # noqa: E402
from functions.models import FINAL_MODEL_NAME, RUNNER_UP_MODEL_NAME  # noqa: E402
from functions.plot_final import plot_cv_vs_holdout  # noqa: E402
from functions.plot_tuning import save_png_pdf  # noqa: E402


def test_training_rows_contain_no_holdout_setups(data):
    """The real split: no setup is in both parts, and counts match assign_folds."""
    train, holdout = final.split_train_holdout(data)
    assert set(train["setup_id"]).isdisjoint(holdout["setup_id"])
    folded = assign_folds(data)
    expected = folded[folded["fold"] == TEST_FOLD]
    assert len(holdout) == len(expected)
    assert holdout["setup_id"].nunique() == expected["setup_id"].nunique()
    assert (train["fold"] != TEST_FOLD).all()


def test_check_split_rejects_shared_setups(data):
    """A split that puts one setup on both sides is refused."""
    folded = assign_folds(data)
    train = folded[folded["fold"] != TEST_FOLD]
    holdout = folded[folded["fold"] == TEST_FOLD]
    leaky = pd.concat([train, holdout.iloc[:1]])
    with pytest.raises(RuntimeError, match="both training and hold-out"):
        final.check_split(leaky, holdout, folded)


def test_refuses_to_overwrite_without_force(tmp_path, monkeypatch, capsys):
    """With existing hold-out results, the script stops before training anything."""
    (tmp_path / run_final.GUARD_FILE).write_text("already run")

    def must_not_train(*args, **kwargs):
        raise AssertionError("models were trained despite existing results")

    monkeypatch.setattr(run_final, "fit_and_predict", must_not_train)
    assert run_final.main([], results_dir=tmp_path, doc_dir=tmp_path / "doc") == 1
    assert "Refusing to run" in capsys.readouterr().out
    assert (tmp_path / run_final.GUARD_FILE).read_text() == "already run"


def test_force_flag_is_parsed():
    """--force is available for regenerating identical results."""
    assert run_final.parse_args(["--force"]).force is True
    assert run_final.parse_args([]).force is False


def _metrics(final_rmse: float, linear_rmse: float) -> pd.DataFrame:
    return pd.DataFrame({"model": ["linear", RUNNER_UP_MODEL_NAME, FINAL_MODEL_NAME],
                         "rmse": [linear_rmse, 2.6, final_rmse]})


def test_acceptance_has_no_pending_after_evaluation():
    """Once hold-out metrics exist, every criterion is PASS or FAIL."""
    paired_cv = pd.DataFrame({"feature_set": "B", "rmse_diff": [-1.0] * 15})
    summary = pd.DataFrame({"feature_set": ["B"], "rmse_mean": [2.4], "rmse_sd_folds": [0.2]})
    physics = pd.DataFrame({"model": ["mlp", "linear"], "pairs_correct": [66, 68],
                            "pairs_scored": [68, 68]})
    table = final.evaluate_acceptance(paired_cv, summary, physics, _metrics(2.5, 4.5))
    assert not table["result"].isin([PENDING_HOLDOUT, PENDING_PHYSICS]).any()
    assert set(table["result"]) <= {"PASS", "FAIL"}


def test_paired_comparison_uses_identical_setups():
    """Differences are per setup; models scored on different setups are refused."""
    setups = pd.DataFrame({
        "model": [FINAL_MODEL_NAME] * 3 + ["linear"] * 3,
        "setup_id": [1, 2, 3, 3, 1, 2],  # same setups, different order
        "rmse": [2.0, 3.0, 1.0, 1.5, 4.0, 2.5],
    })
    paired = final.paired_setups(setups, FINAL_MODEL_NAME, "linear")
    assert sorted(paired["setup_id"]) == [1, 2, 3]
    by_setup = paired.set_index("setup_id")["rmse_diff"]
    assert by_setup.to_dict() == {1: -2.0, 2: 0.5, 3: -0.5}
    stats = final.paired_summary(paired)
    assert (stats["a_wins"], stats["b_wins"], stats["n_setups"]) == (2, 1, 3)

    missing = setups[~((setups["model"] == "linear") & (setups["setup_id"] == 3))]
    with pytest.raises(ValueError, match="same setups"):
        final.paired_setups(missing, FINAL_MODEL_NAME, "linear")


def test_cv_estimates_read_saved_results():
    """Final MLP: nested CV; linear: CV summary; runner-up: its inner-CV scores."""
    tuning = pd.DataFrame({"feature_set": ["B"], "rmse_mean": [2.4], "rmse_sd_folds": [0.2],
                           "n_folds": [15]})
    cv = pd.DataFrame({"model": ["linear"], "feature_set": ["B"], "rmse": [4.6],
                       "rmse_std": [0.25], "n_folds": [15]})
    grid = pd.DataFrame({"feature_set": "B", "hidden_layer_sizes": ["(64, 64)", "(64, 64)", "(32,)"],
                         "alpha": [0.1, 0.1, 0.1], "mean_inner_rmse": [2.5, 2.7, 9.9]})
    est = final.cv_estimates(tuning, cv, grid).set_index("model")
    assert est.loc[FINAL_MODEL_NAME, "cv_rmse"] == 2.4
    assert est.loc["linear", "cv_rmse"] == 4.6
    assert est.loc[RUNNER_UP_MODEL_NAME, "cv_rmse"] == pytest.approx(2.6)
    assert "inner" in est.loc[RUNNER_UP_MODEL_NAME, "estimate"]


def test_pipeline_on_a_stand_in_fold(data, tmp_path):
    """The evaluation helpers work end to end, using CV fold 0 in place of the hold-out."""
    folded = assign_folds(data)
    cv = folded[folded["fold"] != TEST_FOLD]
    train, stand_in = cv[cv["fold"] != 0], cv[cv["fold"] == 0]
    fits = [final.fit_and_predict(name, train, stand_in) for name in ("linear", FINAL_MODEL_NAME)]
    results = {f.model_name: final.as_result(f, train, stand_in) for f in fits}

    metrics = final.holdout_metrics(fits, stand_in)
    assert metrics["n_rows"].eq(len(stand_in)).all()
    assert metrics.set_index("model").loc[FINAL_MODEL_NAME, "selected"]
    mlp = metrics.set_index("model").loc[FINAL_MODEL_NAME]
    assert 0 < mlp["n_iter"] < 2000 and not mlp["hit_max_iter"]

    breakdown = final.holdout_breakdown(results, stand_in)
    for (model, column), part in breakdown.groupby(["model", "breakdown"]):
        assert part["n"].sum() == len(stand_in)

    per_setup = final.setup_rmse(results, data)
    paired = final.paired_setups(per_setup, FINAL_MODEL_NAME, "linear")
    assert len(paired) == stand_in["setup_id"].nunique()

    preds = final.predictions_table(stand_in, fits)
    assert np.allclose(preds["pred_linear"], fits[0].y_pred)

    estimates = pd.DataFrame({"model": ["linear", FINAL_MODEL_NAME], "estimate": ["cv", "nested"],
                              "cv_rmse": [4.6, 2.4], "cv_sd": [0.25, 0.2]})
    labelled = metrics.assign(label=metrics["model"].map(final.HOLDOUT_MODELS))
    paths = save_png_pdf(plot_cv_vs_holdout(estimates, labelled), tmp_path, "cv_vs_holdout")
    assert all(p.stat().st_size > 0 for p in paths)


def test_full_script_on_a_stand_in_fold(data, tmp_path, monkeypatch):
    """run_final.main runs end to end and writes every output, using CV fold 0
    in place of the hold-out and a temporary results folder (no real hold-out)."""
    import shutil

    from run_cv import RESULTS_DIR
    for name in ("tuning_summary.csv", "cv_summary.csv", "tuning_inner_grid.csv",
                 "tuning_paired_vs_linear.csv", "physics_speed_order_summary.csv"):
        shutil.copy(RESULTS_DIR / name, tmp_path / name)

    folded = assign_folds(data)
    cv = folded[folded["fold"] != TEST_FOLD]
    monkeypatch.setattr(run_final, "split_train_holdout",
                        lambda _: (cv[cv["fold"] != 0], cv[cv["fold"] == 0]))

    assert run_final.main([], results_dir=tmp_path, doc_dir=tmp_path / "doc") == 0
    for name in ("holdout_metrics.csv", "holdout_breakdown.csv", "holdout_setups.csv",
                 "holdout_setup_rmse.csv", "holdout_paired.csv", "holdout_predictions.csv",
                 "holdout_vs_cv.csv", "acceptance.csv", "holdout_run_info.txt"):
        assert (tmp_path / name).exists(), name
    assert len(list((tmp_path / "final_models").glob("*.joblib"))) == 3
    for stem in ("holdout_pred_vs_measured", "holdout_spectra", "holdout_rmse_breakdown", "holdout_vs_cv"):
        assert (tmp_path / "doc" / f"{stem}.png").exists() and (tmp_path / "doc" / f"{stem}.pdf").exists()
    acceptance = pd.read_csv(tmp_path / "acceptance.csv")
    assert not acceptance["result"].str.startswith("pending").any()
    # A second run without --force is refused.
    assert run_final.main([], results_dir=tmp_path, doc_dir=tmp_path / "doc") == 1


def test_holdout_residual_plot(tmp_path):
    """The residual figure draws one panel per model from a predictions table."""
    from functions.plot_final import plot_holdout_residuals
    rng = np.random.default_rng(0)
    measured = rng.uniform(105, 140, 60)
    predictions = pd.DataFrame({"measured_spl_db": measured,
                                **{f"pred_{m}": measured + rng.normal(0, 2, 60)
                                   for m in final.HOLDOUT_MODELS}})
    fig = plot_holdout_residuals(predictions, {m: m for m in final.HOLDOUT_MODELS})
    assert len(fig.axes) == 3
    paths = save_png_pdf(fig, tmp_path, "residuals")
    assert all(p.stat().st_size > 0 for p in paths)
