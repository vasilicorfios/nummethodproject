"""Tests for :mod:`functions.acceptance` (criteria fixed before the hold-out run)."""

import pandas as pd
import pytest

import functions.acceptance as acceptance
from functions.acceptance import (
    C3_MAX_RMSE_DB,
    FAIL,
    P1_MIN_PAIR_FRACTION,
    PASS,
    PENDING_HOLDOUT,
    PENDING_PHYSICS,
    acceptance_table,
    c2_threshold,
)

#: A made-up tuning summary: set B nested-CV mean 2.4 dB, fold SD 0.2 dB.
SUMMARY = pd.DataFrame({"feature_set": ["A", "B"], "rmse_mean": [2.6, 2.4],
                        "rmse_sd_folds": [0.3, 0.2]})


def _result(table: pd.DataFrame, criterion: str) -> str:
    return table.set_index("criterion").loc[criterion, "result"]


def test_fixed_thresholds():
    """The thresholds are the ones agreed before the hold-out run."""
    assert C3_MAX_RMSE_DB == 3.0
    assert P1_MIN_PAIR_FRACTION == 0.90


def test_only_blind_criteria_remain():
    """The non-blind CV-fold criterion (C1a) is gone; four blind criteria remain."""
    table = acceptance_table(SUMMARY)
    assert table["criterion"].tolist() == ["C1", "C2", "C3", "P1"]
    assert not hasattr(acceptance, "C1_MIN_CV_FOLD_WINS")


def test_holdout_criteria_pending_without_holdout_results():
    """Without hold-out results, C1, C2 and C3 are pending, and P1 too without physics."""
    table = acceptance_table(SUMMARY)
    for criterion in ("C1", "C2", "C3"):
        assert _result(table, criterion) == PENDING_HOLDOUT
    assert _result(table, "P1") == PENDING_PHYSICS


def test_c2_threshold_uses_set_b_mean_plus_two_sd():
    """Threshold = 2.4 + 2 x 0.2 = 2.8 dB for the made-up set B summary."""
    assert c2_threshold(SUMMARY) == pytest.approx(2.8)


def test_holdout_and_physics_results_are_judged():
    """Given results, each criterion becomes PASS or FAIL against its threshold."""
    good = acceptance_table(SUMMARY, physics_pair_fraction=0.95,
                            holdout={"mlp_rmse": 2.5, "linear_rmse": 4.5})
    assert set(good["result"]) == {PASS}
    bad = acceptance_table(SUMMARY, physics_pair_fraction=0.80,
                           holdout={"mlp_rmse": 3.2, "linear_rmse": 3.1})
    for criterion in ("C1", "C2", "C3", "P1"):
        assert _result(bad, criterion) == FAIL
