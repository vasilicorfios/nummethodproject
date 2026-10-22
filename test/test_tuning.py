"""Tests for :mod:`functions.tuning` (nested cross-validation of the MLP).

The real grid takes minutes, so these tests use a one-setting grid with a
tiny network, ``max_iter=20`` and 2 inner folds. That is enough to check the
data-handling guarantees, which do not depend on the grid size: grouped
inner folds, no test-set rows anywhere, and outer validation rows never
reaching the tuning search.
"""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import functions.tuning as tuning  # noqa: E402
import run_tuning  # noqa: E402
from functions.evaluate import cross_validate  # noqa: E402
from functions.make_folds import CV_SEEDS, N_FOLDS, TEST_FOLD, assign_folds  # noqa: E402
from functions.plot_tuning import (  # noqa: E402
    plot_inner_grid_heatmap,
    plot_loss_curves,
    save_png_pdf,
)
from functions.tuning import (  # noqa: E402
    ALPHA_KEY,
    INNER_COLUMNS,
    LAYERS_KEY,
    OUTER_COLUMNS,
    PARAM_GRID,
    TuningResult,
    edge_of_grid_warnings,
    inner_splitter,
    nested_cv,
    paired_summary,
    paired_vs_linear,
    selection_counts,
)

#: One tiny setting: fast to fit, and it always runs to max_iter (20 epochs
#: cannot trigger early stopping, which needs 20 epochs without improvement).
TINY_GRID = {
    LAYERS_KEY: [(4,)],
    ALPHA_KEY: [1e-3],
    "regressor__regressor__max_iter": [20],
}
N_INNER = 2


class RecordingGridSearchCV(tuning.GridSearchCV):
    """GridSearchCV that remembers which rows each ``fit`` call received."""

    calls: list = []

    def fit(self, X, y=None, **kwargs):
        RecordingGridSearchCV.calls.append(set(X.index))
        return super().fit(X, y, **kwargs)


@pytest.fixture(scope="module")
def tiny_run(data):
    """Nested CV over all 3 seeds with the tiny grid, recording fit inputs."""
    RecordingGridSearchCV.calls = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tuning, "GridSearchCV", RecordingGridSearchCV)
        with pytest.warns(Warning):  # ConvergenceWarning from max_iter=20
            result = nested_cv(data, "A", param_grid=TINY_GRID,
                               n_inner_splits=N_INNER, n_jobs=1, progress=None)
    return result, list(RecordingGridSearchCV.calls)


def test_grid_is_the_requested_20_settings():
    """4 architectures x 5 alphas, keyed to the real pipeline's parameter names."""
    assert len(PARAM_GRID[LAYERS_KEY]) * len(PARAM_GRID[ALPHA_KEY]) == 20
    from functions.models import build_model
    params = build_model("mlp").get_params()
    assert LAYERS_KEY in params and ALPHA_KEY in params


def test_inner_folds_never_split_a_setup(data):
    """No setup_id is in both inner training and inner validation."""
    folded = assign_folds(data)
    cv = folded[folded["fold"] != TEST_FOLD]
    outer_train = cv[cv["fold"] != 0]
    for train_idx, val_idx in inner_splitter().split(outer_train, groups=outer_train["setup_id"]):
        train_setups = set(outer_train["setup_id"].iloc[train_idx])
        val_setups = set(outer_train["setup_id"].iloc[val_idx])
        assert train_setups.isdisjoint(val_setups)


def test_one_row_per_outer_fold(tiny_run):
    """3 seeds x 5 folds = 15 outer rows with the documented columns."""
    result, _ = tiny_run
    assert isinstance(result, TuningResult)
    assert list(result.outer_scores.columns) == OUTER_COLUMNS
    assert len(result.outer_scores) == len(CV_SEEDS) * N_FOLDS
    assert list(result.inner_grid.columns) == INNER_COLUMNS
    assert len(result.inner_grid) == len(CV_SEEDS) * N_FOLDS  # 1 setting each


def test_test_holdout_never_used(data, tiny_run):
    """Test rows never reach GridSearchCV.fit and are never scored."""
    result, calls = tiny_run
    folded = assign_folds(data)
    test_rows = set(folded.index[folded["fold"] == TEST_FOLD])
    assert len(calls) == len(CV_SEEDS) * N_FOLDS
    for rows in calls:
        assert rows.isdisjoint(test_rows)
    assert set(result.cv_result.oof["row"]).isdisjoint(test_rows)


def test_outer_validation_rows_never_tuned_on(data, tiny_run):
    """Each search sees exactly the outer-training rows of its fold."""
    _, calls = tiny_run
    i = 0
    for seed in CV_SEEDS:
        folded = assign_folds(data, cv_seed=seed)
        cv = folded[folded["fold"] != TEST_FOLD]
        for fold in range(N_FOLDS):
            outer_val = set(cv.index[cv["fold"] == fold])
            outer_train = set(cv.index[cv["fold"] != fold])
            assert calls[i].isdisjoint(outer_val)
            assert calls[i] == outer_train
            i += 1


def test_outer_scores_match_oof_predictions(tiny_run):
    """Reported outer RMSE is recomputable from the saved OOF predictions."""
    result, _ = tiny_run
    oof = result.cv_result.oof
    for (seed, fold), g in oof.groupby(["seed", "fold"]):
        rmse = np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))
        row = result.outer_scores.query("seed == @seed and fold == @fold")
        assert row["rmse"].iat[0] == pytest.approx(rmse)


def test_max_iter_hits_counted(tiny_run):
    """With max_iter=20 every fit hits the cap, and the count says so."""
    result, _ = tiny_run
    assert (result.outer_scores["inner_fits_hit_max_iter"] == N_INNER).all()
    assert result.outer_scores["refit_hit_max_iter"].all()
    assert (result.outer_scores["n_iter"] == 20).all()


def test_selection_counts_cover_grid(tiny_run):
    """Every grid setting is listed, and the wins add up to 15 folds."""
    result, _ = tiny_run
    counts = selection_counts(result)
    assert len(counts) == 1
    assert counts["n_folds_won"].sum() == len(CV_SEEDS) * N_FOLDS
    assert counts["fraction_won"].sum() == pytest.approx(1.0)


def test_paired_vs_linear_hand_calculated():
    """Differences are MLP minus linear; negative means the MLP won."""
    mlp = pd.DataFrame({"seed": [1, 1, 2], "fold": [0, 1, 0], "rmse": [3.0, 5.0, 4.0]})
    lin = pd.DataFrame({"seed": [1, 1, 2], "fold": [0, 1, 0], "rmse": [4.0, 4.5, 5.0]})
    paired = paired_vs_linear(mlp, lin)
    assert paired["rmse_diff"].tolist() == [-1.0, 0.5, -1.0]
    s = paired_summary(paired)
    assert s["mlp_wins"] == 2 and s["n_folds"] == 3
    assert s["rmse_diff_mean"] == pytest.approx(-0.5)
    assert s["rmse_diff_sd"] == pytest.approx(np.std([-1.0, 0.5, -1.0], ddof=1))


def test_paired_vs_linear_rejects_mismatched_folds():
    """Comparing different folds would be unfair, so it is refused."""
    mlp = pd.DataFrame({"seed": [1], "fold": [0], "rmse": [3.0]})
    lin = pd.DataFrame({"seed": [1], "fold": [1], "rmse": [4.0]})
    with pytest.raises(ValueError):
        paired_vs_linear(mlp, lin)


def _fake_result(layers: list[str], alphas: list[float]) -> TuningResult:
    """A TuningResult with only the winners filled in, for warning tests."""
    outer = pd.DataFrame({"hidden_layer_sizes": layers, "alpha": alphas})
    return TuningResult("A", outer, pd.DataFrame(), None, {}, PARAM_GRID)


def test_edge_of_grid_warning_when_edge_wins_most_folds():
    """Largest alpha winning 3 of 4 folds triggers a warning; layers do not."""
    result = _fake_result(["(64,)", "(32, 32)", "(64,)", "(32, 32)"], [1e-1, 1e-1, 1e-1, 1e-3])
    messages = edge_of_grid_warnings(result)
    assert len(messages) == 1
    assert "alpha" in messages[0] and "largest" in messages[0]


def test_no_edge_warning_for_interior_winners():
    """Interior winners produce no warnings."""
    result = _fake_result(["(64,)"] * 4, [1e-3] * 4)
    assert edge_of_grid_warnings(result) == []



def test_output_csvs_have_expected_columns_and_15_rows(data, tiny_run, tmp_path):
    """All result CSVs are written with the documented columns; 15 rows for 3 seeds."""
    result, _ = tiny_run
    linear = cross_validate(data, "linear", "A")
    paired = paired_vs_linear(result.outer_scores, linear.fold_scores)
    summary = pd.DataFrame([run_tuning.summary_row(result, paired)])
    run_tuning.write_outputs([result], [paired], summary, [], tmp_path)

    outer = pd.read_csv(tmp_path / "tuning_outer_scores.csv")
    assert list(outer.columns) == OUTER_COLUMNS
    assert len(outer) == len(CV_SEEDS) * N_FOLDS
    inner = pd.read_csv(tmp_path / "tuning_inner_grid.csv")
    assert list(inner.columns) == INNER_COLUMNS
    selected = pd.read_csv(tmp_path / "tuning_selected_params.csv")
    assert {"hidden_layer_sizes", "alpha", "n_folds_won", "fraction_won"} <= set(selected.columns)
    pair = pd.read_csv(tmp_path / "tuning_paired_vs_linear.csv")
    assert {"rmse_mlp", "rmse_linear", "rmse_diff"} <= set(pair.columns)
    assert len(pair) == len(CV_SEEDS) * N_FOLDS
    summ = pd.read_csv(tmp_path / "tuning_summary.csv")
    for column in ["rmse_mean", "rmse_sd_folds", "rmse_sd_seeds", "rmse_diff_mean",
                   "mlp_wins", "tuning_time_total_s", "inner_fits_hit_max_iter"]:
        assert column in summ.columns
    # 15 folds x 1 setting x 2 inner splits, all of which hit max_iter=20.
    assert summ["inner_fits"].iat[0] == summ["inner_fits_hit_max_iter"].iat[0] == 30
    assert (tmp_path / "tuning_warnings.txt").read_text().strip() == "No warnings."


def test_tuning_figures_saved_as_png_and_pdf(tiny_run, tmp_path):
    """The heatmap and loss curves are written in both formats."""
    result, _ = tiny_run
    keys = list(result.selected_models)[:2]
    paths = save_png_pdf(plot_inner_grid_heatmap([result]), tmp_path, "heatmap")
    paths += save_png_pdf(plot_loss_curves(result, keys), tmp_path, "loss")
    for path in paths:
        header = path.read_bytes()[:4]
        # PNG files start with byte 0x89 then "PNG"; PDF files with "%PDF".
        if path.suffix == ".png":
            assert header[1:4] == b"PNG"
        else:
            assert header == b"%PDF"
