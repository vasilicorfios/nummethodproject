"""Tests for :mod:`functions.make_folds`.

These check the properties the validation design relies on: whole setups
stay together, the test set never changes between CV repeats, and the
split is reproducible.
"""

import pandas as pd

from functions.make_folds import (
    CV_SEEDS,
    N_FOLDS,
    SETUP_KEYS,
    TEST_FOLD,
    TEST_FRACTION,
    assign_folds,
)


def test_setups_never_split(raw_data):
    """Every row of a setup gets the same fold, so no setup leaks across splits."""
    folded = assign_folds(raw_data)
    assert (folded.groupby("setup_id")["fold"].nunique() == 1).all()
    # One setup id per unique (chord, velocity, angle) combination.
    assert folded["setup_id"].nunique() == len(raw_data.drop_duplicates(SETUP_KEYS))


def test_folds_and_test_set_present(raw_data):
    """Every CV fold and the test set are non-empty, and nothing is unassigned."""
    folded = assign_folds(raw_data)
    assert set(folded["fold"].unique()) == set(range(N_FOLDS)) | {TEST_FOLD}
    assert folded["fold"].notna().all()


def test_test_fraction(raw_data):
    """About TEST_FRACTION of setups are held out."""
    folded = assign_folds(raw_data)
    setups = folded.groupby("setup_id")["fold"].first()
    fraction = (setups == TEST_FOLD).mean()
    assert abs(fraction - TEST_FRACTION) < 0.02


def test_test_set_identical_across_cv_seeds(raw_data):
    """Repeated CV reshuffles folds but never changes which setups are test."""
    test_sets = [
        set(assign_folds(raw_data, cv_seed=s).query("fold == @TEST_FOLD")["setup_id"])
        for s in CV_SEEDS
    ]
    assert all(t == test_sets[0] for t in test_sets)


def test_cv_seeds_change_folds(raw_data):
    """Different CV seeds produce different fold assignments."""
    a = assign_folds(raw_data, cv_seed=CV_SEEDS[0])["fold"]
    b = assign_folds(raw_data, cv_seed=CV_SEEDS[1])["fold"]
    assert not a.equals(b)


def test_every_chord_in_test_set(raw_data):
    """Stratification puts every chord length in the test set."""
    folded = assign_folds(raw_data)
    test_chords = set(folded.loc[folded["fold"] == TEST_FOLD, "chord_length_m"])
    assert test_chords == set(raw_data["chord_length_m"])


def test_reproducible_and_input_unchanged(raw_data):
    """Same seeds give the same split, and the input DataFrame is not modified."""
    before = raw_data.copy()
    pd.testing.assert_frame_equal(assign_folds(raw_data), assign_folds(raw_data))
    pd.testing.assert_frame_equal(raw_data, before)
