"""Entry point that tunes the MLP by nested CV and compares it with linear.

For each feature set, :func:`functions.tuning.nested_cv` tunes and scores
the MLP on the outer folds, the linear baseline is cross-validated on the
same seeds and folds, and the two are compared fold by fold. The test
hold-out is never used::

    python src/run_tuning.py                         # all seeds, sets A and B
    python src/run_tuning.py --seeds 42 --feature-sets A   # quick timing check

Writes to ``results/``:

* ``tuning_outer_scores.csv``: one row per (feature set, seed, outer fold):
  chosen setting, inner-CV best RMSE, outer RMSE/MAE/R², epochs run,
  max-iter hits and timings.
* ``tuning_inner_grid.csv``: mean and std inner-CV RMSE of all 20 settings
  for every (feature set, seed, outer fold).
* ``tuning_selected_params.csv``: how often each setting was chosen.
* ``tuning_paired_vs_linear.csv``: per-fold RMSE of MLP and linear and their
  difference (MLP minus linear).
* ``tuning_summary.csv``: headline numbers per feature set (see
  :func:`summary_row`).
* ``tuning_warnings.txt``: edge-of-grid and convergence warnings, if any.

Rounded tables and the figures (PNG + PDF) go to ``doc/generated/`` for the
HTML results page.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.evaluate import cross_validate, summarise
from functions.feature_sets import FEATURE_SETS
from functions.make_folds import CV_SEEDS
from functions.physics_features import add_physics_features
from functions.plot_results import plot_predicted_vs_measured, plot_rmse_breakdown
from functions.plot_tuning import plot_inner_grid_heatmap, plot_loss_curves, save_png_pdf
from functions.tuning import (
    TuningResult,
    edge_of_grid_warnings,
    nested_cv,
    paired_summary,
    paired_vs_linear,
    seed_level_spread,
    selection_counts,
)
from run_cv import DOC_TABLES_DIR, RESULTS_DIR, _mean_std

#: Number of outer folds shown in the loss-curve figure.
N_LOSS_CURVE_FOLDS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``--seeds`` and ``--feature-sets`` command-line options.

    Args:
        argv: Arguments to parse; ``None`` reads the command line.

    Returns:
        Namespace with ``seeds`` and ``feature_sets`` lists.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CV_SEEDS), choices=CV_SEEDS,
                        help="outer CV seeds to run (default: all three)")
    parser.add_argument("--feature-sets", nargs="+", default=["A", "B"],
                        choices=sorted(FEATURE_SETS))
    return parser.parse_args(argv)


def summary_row(result: TuningResult, paired: pd.DataFrame) -> dict:
    """Headline numbers for one feature set.

    Two different spreads are reported, and labelled separately:

    * ``*_sd_folds``: SD over all outer folds (15 with 3 seeds). Answers
      "how much does the score change from one batch of unseen setups to
      the next?"
    * ``*_sd_seeds``: SD of the per-seed means. Answers "how much does the
      result depend on the random shuffle of setups into folds?"

    Args:
        result: Output of :func:`functions.tuning.nested_cv`.
        paired: Output of :func:`functions.tuning.paired_vs_linear`.

    Returns:
        A flat dict, one value per column of ``tuning_summary.csv``.
    """
    folds = summarise(result.cv_result)
    seeds = seed_level_spread(result.outer_scores)
    pair = paired_summary(paired)
    outer = result.outer_scores
    row = {"feature_set": result.feature_set, "n_folds": folds["n_folds"],
           "n_seeds": seeds["n_seeds"]}
    for metric in ["rmse", "mae", "r2"]:
        row[f"{metric}_mean"] = folds[metric]
        row[f"{metric}_sd_folds"] = folds[f"{metric}_std"]
        row[f"{metric}_sd_seeds"] = seeds[f"{metric}_sd_seeds"]
    row["linear_rmse_mean"] = paired["rmse_linear"].mean()
    row.update(pair)
    row["tuning_time_total_s"] = outer["tuning_time_s"].sum()
    row["inner_fits"] = len(result.inner_grid) * result.n_inner_splits
    row["inner_fits_hit_max_iter"] = int(outer["inner_fits_hit_max_iter"].sum())
    row["refits_hit_max_iter"] = int(outer["refit_hit_max_iter"].sum())
    return row


def write_outputs(results: list[TuningResult], paired: list[pd.DataFrame],
                  summary: pd.DataFrame, warnings: list[str], out_dir: Path) -> None:
    """Write the full-precision result tables to ``out_dir``.

    Args:
        results: One :class:`~functions.tuning.TuningResult` per feature set.
        paired: Matching :func:`~functions.tuning.paired_vs_linear` tables.
        summary: One :func:`summary_row` per feature set.
        warnings: Warning messages to save (may be empty).
        out_dir: Destination directory (created if missing).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([r.outer_scores for r in results], ignore_index=True).to_csv(
        out_dir / "tuning_outer_scores.csv", index=False)
    pd.concat([r.inner_grid for r in results], ignore_index=True).to_csv(
        out_dir / "tuning_inner_grid.csv", index=False)
    pd.concat([selection_counts(r) for r in results], ignore_index=True).to_csv(
        out_dir / "tuning_selected_params.csv", index=False)
    pd.concat([p.assign(feature_set=r.feature_set) for r, p in zip(results, paired)],
              ignore_index=True).to_csv(out_dir / "tuning_paired_vs_linear.csv", index=False)
    summary.to_csv(out_dir / "tuning_summary.csv", index=False)
    (out_dir / "tuning_warnings.txt").write_text(
        "\n".join(warnings) + "\n" if warnings else "No warnings.\n", encoding="utf-8")


def write_doc_tables(results: list[TuningResult], summary: pd.DataFrame) -> None:
    """Write rounded tables for the HTML results page to ``doc/generated/``.

    Args:
        results: One :class:`~functions.tuning.TuningResult` per feature set.
        summary: One :func:`summary_row` per feature set.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Feature set": summary["feature_set"],
        "Outer folds": summary["n_folds"],
        "MLP RMSE (dB), mean ± SD over folds": _mean_std(summary["rmse_mean"], summary["rmse_sd_folds"], 3),
        "MLP RMSE SD across seeds (dB)": summary["rmse_sd_seeds"].round(3),
        "MLP MAE (dB)": _mean_std(summary["mae_mean"], summary["mae_sd_folds"], 3),
        "MLP R²": _mean_std(summary["r2_mean"], summary["r2_sd_folds"], 3),
        "Linear RMSE (dB)": summary["linear_rmse_mean"].round(3),
        "MLP − linear RMSE per fold (dB)": _mean_std(summary["rmse_diff_mean"], summary["rmse_diff_sd"], 3),
        "Folds MLP won": summary["mlp_wins"].astype(str) + " / " + summary["n_folds"].astype(str),
        "Tuning time (min)": (summary["tuning_time_total_s"] / 60).round(1),
    }).to_csv(DOC_TABLES_DIR / "tuning_summary.csv", index=False)

    chosen = pd.concat([selection_counts(r) for r in results], ignore_index=True)
    chosen = chosen[chosen["n_folds_won"] > 0]
    pd.DataFrame({
        "Feature set": chosen["feature_set"],
        "Hidden layer sizes": chosen["hidden_layer_sizes"],
        "alpha": chosen["alpha"].map("{:g}".format),
        "Outer folds won": chosen["n_folds_won"],
    }).to_csv(DOC_TABLES_DIR / "tuning_selected_params.csv", index=False)


def write_figures(results: list[TuningResult], linear: list, seed: int) -> None:
    """Draw the tuning and comparison figures as PNG + PDF in ``doc/generated/``.

    Args:
        results: One :class:`~functions.tuning.TuningResult` per feature set.
        linear: Matching linear :class:`~functions.evaluate.CVResult` objects.
        seed: Seed whose predictions are shown in per-row figures.
    """
    save_png_pdf(plot_inner_grid_heatmap(results), DOC_TABLES_DIR, "tuning_inner_grid_heatmap")
    for result, lin in zip(results, linear):
        fs = result.feature_set
        keys = [k for k in result.selected_models if k[0] == seed][:N_LOSS_CURVE_FOLDS]
        save_png_pdf(plot_loss_curves(result, keys), DOC_TABLES_DIR, f"tuning_loss_curves_{fs}")
        # Linear first (blue), tuned MLP second (orange), as in every comparison.
        pair = [lin, result.cv_result]
        save_png_pdf(plot_predicted_vs_measured(pair, seed=seed), DOC_TABLES_DIR,
                     f"tuning_pred_vs_measured_{fs}")
        save_png_pdf(plot_rmse_breakdown(pair), DOC_TABLES_DIR, f"tuning_rmse_breakdown_{fs}")


def main(argv: list[str] | None = None) -> None:
    """Tune, compare, print and save results for every requested feature set."""
    args = parse_args(argv)
    data = add_physics_features(load_airfoil_data())
    start = time.perf_counter()

    results, linear, paired, rows, warnings = [], [], [], [], []
    for feature_set in args.feature_sets:
        result = nested_cv(data, feature_set, seeds=args.seeds)
        # The baseline is re-run on exactly the same seeds and folds, so the
        # comparison is like for like (it takes well under a second).
        lin = cross_validate(data, "linear", feature_set, cv_seeds=args.seeds)
        pair = paired_vs_linear(result.outer_scores, lin.fold_scores)
        results.append(result)
        linear.append(lin)
        paired.append(pair)
        rows.append(summary_row(result, pair))
        warnings += edge_of_grid_warnings(result)

    summary = pd.DataFrame(rows)
    for row in rows:
        if row["inner_fits_hit_max_iter"] or row["refits_hit_max_iter"]:
            warnings.append(
                f"[{row['feature_set']}] {row['inner_fits_hit_max_iter']} of {row['inner_fits']} "
                f"inner fits and {row['refits_hit_max_iter']} refits stopped at max_iter "
                "instead of by early stopping."
            )
    wall_time = time.perf_counter() - start

    print("\n=== Tuned MLP: outer-fold results ===")
    for row in rows:
        print(
            f"\nFeature set {row['feature_set']} ({row['n_folds']} outer folds, {row['n_seeds']} seed(s))\n"
            f"  RMSE {row['rmse_mean']:.3f} dB   SD over folds {row['rmse_sd_folds']:.3f}"
            f"   SD across seeds {row['rmse_sd_seeds']:.3f}\n"
            f"  MAE  {row['mae_mean']:.3f} dB   SD over folds {row['mae_sd_folds']:.3f}"
            f"   SD across seeds {row['mae_sd_seeds']:.3f}\n"
            f"  R²   {row['r2_mean']:.3f}      SD over folds {row['r2_sd_folds']:.3f}"
            f"   SD across seeds {row['r2_sd_seeds']:.3f}\n"
            f"  vs linear (same folds): linear RMSE {row['linear_rmse_mean']:.3f} dB; "
            f"MLP minus linear per fold {row['rmse_diff_mean']:+.3f} ± {row['rmse_diff_sd']:.3f} dB; "
            f"MLP won {row['mlp_wins']} of {row['n_folds']} folds\n"
            f"  tuning time {row['tuning_time_total_s'] / 60:.1f} min"
        )
    print("\n(SD over folds: change from one batch of unseen setups to the next."
          "\n SD across seeds: dependence on the random shuffle into folds.)")
    print(f"\nTotal runtime: {wall_time / 60:.1f} min")
    for message in warnings:
        print(f"WARNING: {message}")

    write_outputs(results, paired, summary, warnings, RESULTS_DIR)
    write_doc_tables(results, summary)
    write_figures(results, linear, seed=args.seeds[0])
    print(f"\nCSV files written to {RESULTS_DIR}")
    print(f"Documentation tables and figures written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
