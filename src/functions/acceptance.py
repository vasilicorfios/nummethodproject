"""Acceptance criteria for the final model, fixed BEFORE the hold-out evaluation.

Why fix them in advance? If the pass marks were chosen after seeing the
hold-out score, it would always be possible to pick ones the model passes.
Writing them down first, and committing this file on its own, makes the
final check honest: the git history shows the thresholds existed before the
hold-out was used.

All criteria concern the final MLP (:data:`functions.models.FINAL_MLP`) on
:data:`functions.models.FINAL_FEATURE_SET`, against the linear baseline on
the same features. Some can be evaluated now from cross-validation results;
the rest wait for the single final hold-out run.

* **C1 Beats the baseline.**

  * C1a (now): the MLP has a lower RMSE than linear in at least
    :data:`C1_MIN_CV_FOLD_WINS` of the 15 nested-CV outer folds
    (``results/tuning_paired_vs_linear.csv``).
  * C1b (hold-out): the MLP has a lower RMSE than linear on the hold-out.

* **C2 Generalises.** Hold-out RMSE ≤ nested-CV mean RMSE +
  :data:`C2_N_SD` × the fold-to-fold SD (``results/tuning_summary.csv``).
  The threshold is computed now; the check waits for the hold-out. A
  hold-out score far worse than the spread seen across CV folds would mean
  the CV results were optimistic.
* **C3 Useful accuracy.** Hold-out MLP RMSE ≤ :data:`C3_MAX_RMSE_DB` dB.
  3 dB is about a factor of two in acoustic power, a change a listener
  notices but which is small compared with the roughly 40 dB range of SPL
  in the data.
* **P1 Physics.** The MLP predicts that faster flow is louder in at least
  :data:`P1_MIN_PAIR_FRACTION` of speed pairs (same chord and angle),
  evaluated by ``src/run_physics_check.py`` on out-of-fold predictions.

Example:
    >>> from functions.acceptance import acceptance_table
    >>> acceptance_table(paired, tuning_summary)   # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

import pandas as pd

from functions.models import FINAL_FEATURE_SET

#: C1a: the MLP must beat linear in at least this many of the 15 CV folds.
C1_MIN_CV_FOLD_WINS = 13

#: C2: allowed hold-out RMSE = CV mean + this many fold-to-fold SDs.
C2_N_SD = 2.0

#: C3: largest acceptable hold-out MLP RMSE (dB).
C3_MAX_RMSE_DB = 3.0

#: P1: smallest acceptable fraction of speed pairs in the correct order.
P1_MIN_PAIR_FRACTION = 0.90

#: Result labels.
PASS, FAIL = "PASS", "FAIL"
PENDING_HOLDOUT = "pending (hold-out)"
PENDING_PHYSICS = "pending (physics check)"


def _verdict(ok: bool) -> str:
    """PASS or FAIL."""
    return PASS if ok else FAIL


def c2_threshold(tuning_summary: pd.DataFrame, feature_set: str = FINAL_FEATURE_SET) -> float:
    """C2's allowed hold-out RMSE: nested-CV mean + C2_N_SD × fold SD (dB).

    Args:
        tuning_summary: Contents of ``results/tuning_summary.csv``.
        feature_set: Feature set of the final model.

    Returns:
        The threshold in dB.
    """
    row = tuning_summary.set_index("feature_set").loc[feature_set]
    return float(row["rmse_mean"] + C2_N_SD * row["rmse_sd_folds"])


def acceptance_table(
    paired: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    physics_pair_fraction: float | None = None,
    holdout: dict[str, float] | None = None,
    feature_set: str = FINAL_FEATURE_SET,
) -> pd.DataFrame:
    """Evaluate every criterion that can be evaluated with the results given.

    Args:
        paired: Contents of ``results/tuning_paired_vs_linear.csv``.
        tuning_summary: Contents of ``results/tuning_summary.csv``.
        physics_pair_fraction: Fraction of speed pairs the MLP orders
            correctly (from the physics check), or ``None`` if not run yet.
        holdout: Hold-out results ``{"mlp_rmse": ..., "linear_rmse": ...}``
            in dB, or ``None`` before the hold-out run.
        feature_set: Feature set of the final model.

    Returns:
        One row per criterion with ``criterion``, ``description``,
        ``threshold``, ``value`` and ``result`` (PASS, FAIL or pending).
    """
    folds = paired[paired["feature_set"] == feature_set]
    wins = int((folds["rmse_diff"] < 0).sum())
    threshold_c2 = c2_threshold(tuning_summary, feature_set)
    have_holdout = holdout is not None

    rows = [
        {
            "criterion": "C1a",
            "description": "MLP beats linear in nested-CV outer folds",
            "threshold": f">= {C1_MIN_CV_FOLD_WINS} of {len(folds)} folds",
            "value": f"{wins} of {len(folds)} folds",
            "result": _verdict(wins >= C1_MIN_CV_FOLD_WINS),
        },
        {
            "criterion": "C1b",
            "description": "MLP RMSE < linear RMSE on the hold-out",
            "threshold": "MLP < linear",
            "value": (f"MLP {holdout['mlp_rmse']:.3f} vs linear {holdout['linear_rmse']:.3f} dB"
                      if have_holdout else ""),
            "result": (_verdict(holdout["mlp_rmse"] < holdout["linear_rmse"])
                       if have_holdout else PENDING_HOLDOUT),
        },
        {
            "criterion": "C2",
            "description": f"Hold-out RMSE <= CV mean + {C2_N_SD:g} x fold SD",
            "threshold": f"<= {threshold_c2:.3f} dB",
            "value": f"{holdout['mlp_rmse']:.3f} dB" if have_holdout else "",
            "result": (_verdict(holdout["mlp_rmse"] <= threshold_c2)
                       if have_holdout else PENDING_HOLDOUT),
        },
        {
            "criterion": "C3",
            "description": "Hold-out MLP RMSE small enough to be useful",
            "threshold": f"<= {C3_MAX_RMSE_DB:g} dB",
            "value": f"{holdout['mlp_rmse']:.3f} dB" if have_holdout else "",
            "result": (_verdict(holdout["mlp_rmse"] <= C3_MAX_RMSE_DB)
                       if have_holdout else PENDING_HOLDOUT),
        },
        {
            "criterion": "P1",
            "description": "MLP predicts faster flow is louder (speed pairs)",
            "threshold": f">= {P1_MIN_PAIR_FRACTION:.0%} of pairs",
            "value": f"{physics_pair_fraction:.1%}" if physics_pair_fraction is not None else "",
            "result": (_verdict(physics_pair_fraction >= P1_MIN_PAIR_FRACTION)
                       if physics_pair_fraction is not None else PENDING_PHYSICS),
        },
    ]
    return pd.DataFrame(rows)
