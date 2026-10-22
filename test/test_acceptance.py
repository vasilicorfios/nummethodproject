"""Tests for :mod:`functions.acceptance` (criteria fixed before the hold-out run)."""

import pandas as pd
import pytest

from functions.acceptance import (
    C1_MIN_CV_FOLD_WINS,
    C3_MAX_RMSE_DB,
    FAIL,
    P1_MIN_PAIR_FRACTION,
    PASS,
    PENDING_HOLDOUT,
    PENDING_PHYSICS,
    acceptance_table,
    c2_threshold,
)


def _inputs(mlp_wins: int):
    """Paired fold table with ``mlp_wins`` MLP wins out of 15, plus a summary."""
    diff = [-1.0] * mlp_wins + [0.5] * (15 - mlp_wins)
    paired = pd.DataFrame({"feature_set": "B", "seed": 42, "fold": range(15), "rmse_diff": diff})
    summary = pd.DataFrame({"feature_set": ["A", "B"], "rmse_mean": [2.6, 2.4],
                            "rmse_sd_folds": [0.3, 0.2]})
    return paired, summary


def _result(table: pd.DataFrame, criterion: str) -> str:
    return table.set_index("criterion").loc[criterion, "result"]


def test_fixed_thresholds():
    """The thresholds are the ones agreed before the hold-out run."""
    assert C1_MIN_CV_FOLD_WINS == 13
    assert C3_MAX_RMSE_DB == 3.0
    assert P1_MIN_PAIR_FRACTION == 0.90


def test_holdout_criteria_pending_without_holdout_results():
    """Without hold-out results, C1b, C2 and C3 are pending, and P1 too without physics."""
    table = acceptance_table(*_inputs(15))
    assert table["criterion"].tolist() == ["C1a", "C1b", "C2", "C3", "P1"]
    for criterion in ("C1b", "C2", "C3"):
        assert _result(table, criterion) == PENDING_HOLDOUT
    assert _result(table, "P1") == PENDING_PHYSICS
    assert _result(table, "C1a") == PASS


def test_c1a_needs_13_of_15_wins():
    """12 wins fails, 13 passes."""
    assert _result(acceptance_table(*_inputs(12)), "C1a") == FAIL
    assert _result(acceptance_table(*_inputs(13)), "C1a") == PASS


def test_c2_threshold_uses_set_b_mean_plus_two_sd():
    """Threshold = 2.4 + 2 x 0.2 = 2.8 dB for the made-up set B summary."""
    _, summary = _inputs(15)
    assert c2_threshold(summary) == pytest.approx(2.8)


def test_holdout_and_physics_results_are_judged():
    """Given results, each criterion becomes PASS or FAIL against its threshold."""
    paired, summary = _inputs(15)
    good = acceptance_table(paired, summary, physics_pair_fraction=0.95,
                            holdout={"mlp_rmse": 2.5, "linear_rmse": 4.5})
    assert set(good["result"]) == {PASS}
    bad = acceptance_table(paired, summary, physics_pair_fraction=0.80,
                           holdout={"mlp_rmse": 3.2, "linear_rmse": 3.1})
    for criterion in ("C1b", "C2", "C3", "P1"):
        assert _result(bad, criterion) == FAIL
