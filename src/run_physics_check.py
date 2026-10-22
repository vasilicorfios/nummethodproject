"""Entry point for the "faster flow is louder" physics check and acceptance table.

Checks whether the MLP's and linear model's out-of-fold predictions rise
with free-stream velocity, at fixed chord and angle, as the measurements and
the physics do (:mod:`functions.physics_checks`). Only cross-validation rows
are used; the test hold-out is never touched. It then evaluates the
acceptance criteria fixed in :mod:`functions.acceptance`: those that need
the hold-out are reported as pending::

    python src/run_physics_check.py    # about 10 seconds

Needs ``results/tuning_oof_predictions.csv`` and the other tuning results
from ``src/run_tuning.py``.

Writes:

* ``results/physics_speed_order.csv``: one row per (chord, angle) group:
  speeds, shared bands, whether the measured data rises with speed, and each
  model's order and pair results.
* ``results/physics_speed_order_levels.csv``: the mean SPL of every setup used.
* ``results/physics_speed_order_summary.csv``: the scores per model.
* ``results/acceptance.csv``: the acceptance criteria table.
* ``doc/generated/physics_speed_order.csv`` and ``acceptance.csv``: rounded copies.
* ``doc/generated/physics_speed_order.png`` and ``.pdf``: example groups.
"""

import pandas as pd

from functions.acceptance import acceptance_table
from functions.data_loader import load_airfoil_data
from functions.evaluate import cross_validate
from functions.models import FINAL_FEATURE_SET
from functions.physics_checks import group_levels, speed_order_check
from functions.physics_features import add_physics_features
from functions.plot_robustness import plot_speed_order
from functions.plot_tuning import save_png_pdf
from functions.tuning import OOF_FILENAME, tuned_result_from_saved
from run_cv import DOC_TABLES_DIR, RESULTS_DIR

#: Number of example groups in the figure.
N_EXAMPLES = 4


def example_groups(groups: pd.DataFrame, n: int = N_EXAMPLES) -> list[tuple[float, float]]:
    """Pick example (chord, angle) groups for the figure.

    Groups where the MLP gets the order wrong come first (they are the
    interesting cases); the rest are correctly ordered groups with the most
    speeds, each from a different chord.

    Args:
        groups: The per-group table from
            :func:`functions.physics_checks.speed_order_check`.
        n: Number of groups to return.

    Returns:
        Up to ``n`` ``(chord, angle)`` pairs.
    """
    scored = groups[groups["measured_increasing"]]
    wrong = scored[~scored["mlp_order_correct"]]
    picks = list(zip(wrong["chord_length_m"], wrong["attack_angle_deg"]))[:n]
    right = scored[scored["mlp_order_correct"]].sort_values(
        ["n_speeds", "chord_length_m"], ascending=[False, True])
    used_chords = {chord for chord, _ in picks}
    for _, row in right.iterrows():
        if len(picks) >= n:
            break
        if row["chord_length_m"] not in used_chords:
            picks.append((row["chord_length_m"], row["attack_angle_deg"]))
            used_chords.add(row["chord_length_m"])
    return picks


def write_doc_tables(summary: pd.DataFrame, groups: pd.DataFrame,
                     acceptance: pd.DataFrame) -> None:
    """Write rounded physics-check and acceptance tables for the docs.

    Args:
        summary: Scores per model.
        groups: Per-group results (for the list of excluded groups).
        acceptance: The acceptance criteria table.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Model": summary["model"].map({"mlp": "MLP (nested-CV, B)", "linear": "linear / B"}),
        "Groups fully in order": summary["groups_correct"].astype(str) + " / "
                                 + summary["groups_scored"].astype(str)
                                 + " (" + summary["groups_correct_pct"].round(1).astype(str) + "%)",
        "Speed pairs in order": summary["pairs_correct"].astype(str) + " / "
                                + summary["pairs_scored"].astype(str)
                                + " (" + summary["pairs_correct_pct"].round(1).astype(str) + "%)",
        "Groups excluded (measured not louder at higher speed)": summary["groups_excluded"],
    }).to_csv(DOC_TABLES_DIR / "physics_speed_order.csv", index=False)

    excluded = groups[~groups["measured_increasing"]]
    pd.DataFrame({
        "Chord (m)": excluded["chord_length_m"],
        "Angle (deg)": excluded["attack_angle_deg"],
        "Speeds (m/s)": excluded["speeds"],
        "Shared bands": excluded["n_bands"],
    }).to_csv(DOC_TABLES_DIR / "physics_speed_order_excluded.csv", index=False)

    acceptance.rename(columns=str.capitalize).to_csv(DOC_TABLES_DIR / "acceptance.csv", index=False)


def main() -> None:
    """Run the speed-order check, then evaluate and print the acceptance table."""
    data = add_physics_features(load_airfoil_data())
    # Out-of-fold predictions only: each setup was predicted by a model that
    # never saw it, so the check cannot be passed by memorising the data.
    mlp = tuned_result_from_saved(pd.read_csv(RESULTS_DIR / OOF_FILENAME),
                                  pd.read_csv(RESULTS_DIR / "tuning_outer_scores.csv"),
                                  FINAL_FEATURE_SET)
    linear = cross_validate(data, "linear", FINAL_FEATURE_SET)

    levels = group_levels(data, mlp, linear)
    groups, summary = speed_order_check(levels)

    print("Physics check: faster flow should be louder (same chord and angle)")
    print(summary.round(1).to_string(index=False))
    excluded = groups[~groups["measured_increasing"]]
    print(f"\n{len(excluded)} group(s) left out because the measured data is not louder at "
          "every higher speed:")
    print(excluded[["chord_length_m", "attack_angle_deg", "speeds", "n_bands"]].to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    groups.to_csv(RESULTS_DIR / "physics_speed_order.csv", index=False)
    levels.to_csv(RESULTS_DIR / "physics_speed_order_levels.csv", index=False)
    summary.to_csv(RESULTS_DIR / "physics_speed_order_summary.csv", index=False)
    save_png_pdf(plot_speed_order(levels, example_groups(groups)), DOC_TABLES_DIR,
                 "physics_speed_order")

    # Acceptance criteria (fixed before the hold-out run); hold-out ones stay pending.
    mlp_pairs = summary.set_index("model").loc["mlp"]
    acceptance = acceptance_table(
        pd.read_csv(RESULTS_DIR / "tuning_paired_vs_linear.csv"),
        pd.read_csv(RESULTS_DIR / "tuning_summary.csv"),
        physics_pair_fraction=mlp_pairs["pairs_correct"] / mlp_pairs["pairs_scored"],
    )
    acceptance.to_csv(RESULTS_DIR / "acceptance.csv", index=False)
    write_doc_tables(summary, groups, acceptance)
    print("\nAcceptance criteria (fixed before the hold-out evaluation)")
    print(acceptance[["criterion", "threshold", "value", "result"]].to_string(index=False))
    print(f"\nResults written to {RESULTS_DIR}; doc tables and figure to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
