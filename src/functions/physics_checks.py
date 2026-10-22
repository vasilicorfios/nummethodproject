"""Physics check: at the same chord and angle, faster flow should be louder.

Physics: for a fixed airfoil (chord) and angle of attack, a higher
free-stream velocity produces more turbulent boundary-layer trailing-edge
noise, so the overall noise level should rise with speed. The measured
spectra in the BPM report (Brooks, Pope & Marcolini, NASA RP-1218, 1989)
show this. A model that has learned real physics, not just patterns in the
training data, should predict the same ordering.

Method (:func:`group_levels`, :func:`speed_order_check`):

1. Use the cross-validation rows only (the test hold-out is dropped first).
2. Group setups by (chord, angle). Keep groups tested at 2 or more speeds.
3. Within a group, keep only the frequency bands measured at **every** speed
   in that group, so like is compared with like (a band missing at one speed
   would otherwise shift that speed's average).
4. For each setup, average the SPL over those shared bands: measured, MLP
   and linear.
5. A group is **fully correct** if the predicted averages rise with speed in
   the same order as the measured ones. Each **pair** of speeds is also
   checked separately (faster predicted louder than slower).

Groups where the *measured* average does not rise with speed are listed and
left out of the score, because the expected ordering does not hold there.

**Why the predictions are honest.** The predictions are out-of-fold: every
setup was predicted by a model that never saw it during training. The MLP's
come from ``results/tuning_oof_predictions.csv``, made by the nested-CV MLPs,
where each outer fold used its own tuned setting (the final setting was the
most common choice). The linear model's come from
:func:`functions.evaluate.cross_validate` on the same folds. Both are
averaged over the 3 CV seeds.

Example:
    >>> levels = group_levels(data, mlp_result, linear_result)   # doctest: +SKIP
    >>> groups, summary = speed_order_check(levels)             # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

from itertools import combinations

import pandas as pd

from functions.evaluate import CVResult
from functions.shift_tests import cv_rows

#: A group needs at least this many speeds to be checked.
MIN_SPEEDS = 2

#: Models compared, and the column holding each one's mean SPL.
MODEL_COLUMNS = {"mlp": "mlp_spl", "linear": "linear_spl"}


def _mean_prediction(result: CVResult) -> pd.Series:
    """Out-of-fold prediction per data row, averaged over the CV seeds."""
    return result.oof.groupby("row")["y_pred"].mean()


def group_levels(data: pd.DataFrame, mlp: CVResult, linear: CVResult) -> pd.DataFrame:
    """Mean SPL per setup over the bands shared within its (chord, angle) group.

    Args:
        data: Dataset with physics features.
        mlp: Tuned-MLP out-of-fold result.
        linear: Linear out-of-fold result on the same folds.

    Returns:
        One row per setup in a group with at least :data:`MIN_SPEEDS` speeds,
        with ``chord_length_m``, ``attack_angle_deg``, ``velocity_mps``,
        ``setup_id``, ``n_bands`` (shared bands averaged over),
        ``measured_spl``, ``mlp_spl`` and ``linear_spl`` (dB).
    """
    cv = cv_rows(data).copy()
    cv["mlp_spl"] = _mean_prediction(mlp).reindex(cv.index)
    cv["linear_spl"] = _mean_prediction(linear).reindex(cv.index)

    rows = []
    for (chord, angle), group in cv.groupby(["chord_length_m", "attack_angle_deg"]):
        speeds = sorted(group["velocity_mps"].unique())
        if len(speeds) < MIN_SPEEDS:
            continue
        # Bands measured at every speed of this group: compare like with like.
        shared = set.intersection(*(set(group.loc[group["velocity_mps"] == s, "frequency_hz"])
                                    for s in speeds))
        if not shared:
            continue
        common = group[group["frequency_hz"].isin(shared)]
        for speed, setup in common.groupby("velocity_mps"):
            rows.append({
                "chord_length_m": chord,
                "attack_angle_deg": angle,
                "velocity_mps": speed,
                "setup_id": int(setup["setup_id"].iat[0]),
                "n_bands": len(shared),
                "measured_spl": setup["spl_db"].mean(),
                "mlp_spl": setup["mlp_spl"].mean(),
                "linear_spl": setup["linear_spl"].mean(),
            })
    return pd.DataFrame(rows)


def _increasing(values: list[float]) -> bool:
    """True if every value is strictly larger than the one before."""
    return all(b > a for a, b in zip(values, values[1:]))


def speed_order_check(levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check whether predictions rise with speed like the measurements do.

    Args:
        levels: Output of :func:`group_levels` (or any table with the same
            columns).

    Returns:
        ``(groups, summary)``. ``groups`` has one row per (chord, angle)
        group: ``n_speeds``, ``n_bands``, ``measured_increasing`` (False
        means the group is excluded from the score) and, for each model,
        ``<model>_order_correct``, ``<model>_pairs_correct`` and
        ``n_pairs``. ``summary`` has one row per model: groups and pairs
        scored, how many are correct, the percentages, and the number of
        excluded groups.
    """
    group_rows = []
    for (chord, angle), g in levels.groupby(["chord_length_m", "attack_angle_deg"]):
        g = g.sort_values("velocity_mps")
        measured = g["measured_spl"].tolist()
        row = {
            "chord_length_m": chord,
            "attack_angle_deg": angle,
            "n_speeds": len(g),
            "speeds": ", ".join(f"{s:g}" for s in g["velocity_mps"]),
            "n_bands": int(g["n_bands"].iat[0]),
            "measured_increasing": _increasing(measured),
            "n_pairs": len(g) * (len(g) - 1) // 2,
        }
        for model, column in MODEL_COLUMNS.items():
            predicted = g[column].tolist()
            row[f"{model}_order_correct"] = _increasing(predicted)
            # Pairs (slower, faster): correct if the faster one is predicted louder.
            row[f"{model}_pairs_correct"] = sum(
                predicted[j] > predicted[i] for i, j in combinations(range(len(predicted)), 2))
        group_rows.append(row)
    groups = pd.DataFrame(group_rows)

    scored = groups[groups["measured_increasing"]]
    summary = pd.DataFrame([
        {
            "model": model,
            "groups_scored": len(scored),
            "groups_correct": int(scored[f"{model}_order_correct"].sum()),
            "groups_correct_pct": 100 * scored[f"{model}_order_correct"].mean() if len(scored) else float("nan"),
            "pairs_scored": int(scored["n_pairs"].sum()),
            "pairs_correct": int(scored[f"{model}_pairs_correct"].sum()),
            "pairs_correct_pct": (100 * scored[f"{model}_pairs_correct"].sum() / scored["n_pairs"].sum()
                                  if len(scored) else float("nan")),
            "groups_excluded": int((~groups["measured_increasing"]).sum()),
        }
        for model in MODEL_COLUMNS
    ])
    return groups, summary
