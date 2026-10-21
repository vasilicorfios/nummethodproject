"""Grouped, chord-stratified test hold-out and 5-fold cross-validation split.

Each wind-tunnel setup (a unique chord length, free-stream velocity and angle
of attack) contributes one noise spectrum of 8-19 rows. Rows from the same
spectrum are strongly correlated, so whole setups are assigned to a split:
a setup's rows are never divided between training and validation/test.

The split is stratified on chord length because chord is the only variable
tested at the same values throughout the dataset; the angles tested differ
for every chord and some velocity/angle combinations were never measured.

Splitting happens on a table with one row per setup, so ordinary stratified
splitters give a grouped split automatically. The result is mapped back onto
the measurement rows as two columns:

* ``setup_id``: integer id of the setup, numbered in order of
  (chord, velocity, angle).
* ``fold``: ``0`` to ``N_FOLDS - 1`` for cross-validation rows, or
  :data:`TEST_FOLD` for rows in the held-out test set.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> from functions.make_folds import assign_folds, CV_SEEDS, TEST_FOLD
    >>> raw = load_airfoil_data()
    >>> for seed in CV_SEEDS:
    ...     df = assign_folds(raw, cv_seed=seed)
    ...     test = df[df["fold"] == TEST_FOLD]  # same setups for every seed
    ...     cv = df[df["fold"] != TEST_FOLD]
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

#: Columns that together identify one wind-tunnel setup.
SETUP_KEYS = ["chord_length_m", "velocity_mps", "attack_angle_deg"]

#: Column the split is stratified on.
STRATIFY_ON = "chord_length_m"

#: Number of cross-validation folds.
N_FOLDS = 5

#: Fraction of setups held out as the untouched test set.
TEST_FRACTION = 0.15

#: ``fold`` value given to rows in the held-out test set.
TEST_FOLD = -1

#: Seed for the test hold-out, distinct from every CV seed. Kept fixed across
#: CV repeats so the test set stays the same setups and is never used for
#: tuning in any repeat.
TEST_SEED = 41

#: Seeds for repeated cross-validation. Each seed reshuffles the 5 folds
#: within the same non-test setups; averaging scores over all repeats shows
#: how much results depend on the particular split.
CV_SEEDS = [42, 43, 44]


def assign_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    test_fraction: float = TEST_FRACTION,
    cv_seed: int = CV_SEEDS[0],
    test_seed: int = TEST_SEED,
) -> pd.DataFrame:
    """Return a copy of the dataset with ``setup_id`` and ``fold`` columns.

    First holds out ``test_fraction`` of the setups as a test set, then
    splits the remaining setups into ``n_folds`` cross-validation folds.
    Both steps are stratified on chord length and work on whole setups.
    For repeated CV, call once per seed in :data:`CV_SEEDS`; the test set
    depends only on ``test_seed`` and so is identical across repeats.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data` (physics feature
            columns, if present, are carried through unchanged).
        n_folds: Number of cross-validation folds.
        test_fraction: Fraction of setups held out as the test set.
        cv_seed: Random seed for shuffling setups into CV folds.
        test_seed: Random seed for choosing the test hold-out.

    Returns:
        A new DataFrame with the original columns plus ``setup_id`` and
        ``fold``. The input DataFrame is not modified.
    """
    out = df.copy()
    # sort=True numbers setups in (chord, velocity, angle) order, so ids are
    # stable regardless of row order in the Excel file.
    out["setup_id"] = out.groupby(SETUP_KEYS, sort=True).ngroup()

    # One row per setup: stratifying this table is a grouped split. Chord is
    # cast to str because sklearn rejects float labels as "continuous".
    setups = out.groupby("setup_id")[STRATIFY_ON].first().astype(str)

    cv_ids, test_ids = train_test_split(
        setups.index,
        test_size=test_fraction,
        stratify=setups,
        random_state=test_seed,
    )

    setup_fold = pd.Series(TEST_FOLD, index=setups.index)
    cv_setups = setups.loc[cv_ids]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    # Only the validation indices are needed: each setup lands in exactly
    # one validation fold, which is its fold number.
    for fold, (_, val_idx) in enumerate(splitter.split(cv_setups, cv_setups)):
        setup_fold.loc[cv_setups.index[val_idx]] = fold

    out["fold"] = out["setup_id"].map(setup_fold)
    return out
