"""Tests for :mod:`functions.physics_checks` (faster flow should be louder)."""

import matplotlib

# Non-interactive backend: tests must not open windows.
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import run_physics_check  # noqa: E402
from functions.evaluate import CVResult  # noqa: E402
from functions.make_folds import TEST_FOLD, assign_folds  # noqa: E402
from functions.physics_checks import group_levels, speed_order_check  # noqa: E402
from functions.plot_robustness import plot_speed_order  # noqa: E402
from functions.plot_tuning import save_png_pdf  # noqa: E402


def _levels(groups: dict[tuple[float, float], list[float]], mlp_sign: float = 1.0,
            linear_sign: float = 1.0) -> pd.DataFrame:
    """Synthetic mean levels: measured rises with speed by the given amounts.

    ``mlp_sign`` / ``linear_sign`` of -1 make that model's levels fall with
    speed instead (perfectly reversed order).
    """
    rows = []
    for (chord, angle), measured in groups.items():
        for k, level in enumerate(measured):
            rows.append({"chord_length_m": chord, "attack_angle_deg": angle,
                         "velocity_mps": 30.0 + 10 * k, "setup_id": len(rows), "n_bands": 10,
                         "measured_spl": level, "mlp_spl": mlp_sign * level,
                         "linear_spl": linear_sign * level})
    return pd.DataFrame(rows)


#: Three groups whose measured level rises with speed (6 speed pairs in total).
ORDERED = {(0.1, 0.0): [120.0, 122.0, 125.0], (0.2, 0.0): [118.0, 121.0],
           (0.3, 5.0): [115.0, 117.0, 119.0]}


def _summary_row(summary: pd.DataFrame, model: str) -> pd.Series:
    return summary.set_index("model").loc[model]


def test_perfect_order_scores_100_and_reversed_scores_0():
    """Predictions in the measured order score 100%; perfectly reversed ones 0%."""
    groups, summary = speed_order_check(_levels(ORDERED, mlp_sign=1.0, linear_sign=-1.0))
    mlp, linear = _summary_row(summary, "mlp"), _summary_row(summary, "linear")
    assert mlp["groups_correct_pct"] == pytest.approx(100.0)
    assert mlp["pairs_correct_pct"] == pytest.approx(100.0)
    assert linear["groups_correct_pct"] == pytest.approx(0.0)
    assert linear["pairs_correct_pct"] == pytest.approx(0.0)
    assert mlp["pairs_scored"] == 3 + 1 + 3


def test_group_not_louder_when_faster_is_excluded():
    """A group whose measured level falls with speed is listed and not scored."""
    levels = _levels({**ORDERED, (0.4, 9.0): [125.0, 120.0]})
    groups, summary = speed_order_check(levels)
    assert (~groups["measured_increasing"]).sum() == 1
    row = _summary_row(summary, "mlp")
    assert row["groups_excluded"] == 1 and row["groups_scored"] == 3


def test_partly_correct_pairs_hand_calculated():
    """Speeds 1 < 2 < 3 predicted as 1 < 3 < 2: two of three pairs right, order wrong."""
    levels = _levels({(0.1, 0.0): [120.0, 122.0, 125.0]})
    levels["mlp_spl"] = [120.0, 126.0, 124.0]
    groups, summary = speed_order_check(levels)
    assert not groups["mlp_order_correct"].iat[0]
    assert groups["mlp_pairs_correct"].iat[0] == 2


@pytest.fixture(scope="module")
def perfect_levels(data):
    """group_levels for a 'model' that predicts every CV row exactly."""
    folded = assign_folds(data)
    cv = folded[folded["fold"] != TEST_FOLD]
    oof = pd.DataFrame({"seed": 42, "row": cv.index, "y_true": cv["spl_db"].to_numpy(),
                        "y_pred": cv["spl_db"].to_numpy()})
    perfect = CVResult("perfect", "B", pd.DataFrame(), oof, pd.DataFrame())
    return group_levels(data, perfect, perfect)


def test_group_levels_use_cv_setups_only(data, perfect_levels):
    """No test hold-out setup appears in the physics check."""
    folded = assign_folds(data)
    holdout_setups = set(folded.loc[folded["fold"] == TEST_FOLD, "setup_id"])
    assert set(perfect_levels["setup_id"]).isdisjoint(holdout_setups)


def test_group_levels_compare_like_with_like(perfect_levels):
    """Every group has at least 2 speeds and one shared band count for all its setups."""
    by_group = perfect_levels.groupby(["chord_length_m", "attack_angle_deg"])
    assert (by_group["velocity_mps"].nunique() >= 2).all()
    assert (by_group["n_bands"].nunique() == 1).all()
    # A perfect model's averages equal the measured ones exactly.
    assert perfect_levels["mlp_spl"].to_numpy() == pytest.approx(perfect_levels["measured_spl"].to_numpy())


def test_perfect_model_matches_measured_order(perfect_levels):
    """A perfect model is right in every scored group and pair."""
    _, summary = speed_order_check(perfect_levels)
    assert _summary_row(summary, "mlp")["pairs_correct_pct"] == pytest.approx(100.0)


def test_example_groups_put_mlp_failures_first(tmp_path):
    """The figure shows groups where the MLP is wrong before correct ones."""
    levels = _levels(ORDERED)
    levels.loc[(levels["chord_length_m"] == 0.2), "mlp_spl"] = [121.0, 118.0]  # reversed
    groups, _ = speed_order_check(levels)
    picks = run_physics_check.example_groups(groups, n=3)
    assert picks[0] == (0.2, 0.0)
    assert len(picks) == 3
    paths = save_png_pdf(plot_speed_order(levels, picks), tmp_path, "speed")
    assert all(p.stat().st_size > 0 for p in paths)
