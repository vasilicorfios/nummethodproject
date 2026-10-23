"""The single final evaluation on the test hold-out.

**Run this once.** The test hold-out (``fold == TEST_FOLD``, about 15% of the
wind-tunnel setups, chosen with ``TEST_SEED = 41`` and stratified on chord)
was never used for training, tuning, model selection or any analysis before
this script. Here three models are trained once on all cross-validation rows
and predict the hold-out once:

* the **final MLP** (``FINAL_MLP``): the selected model. The acceptance
  criteria apply to it.
* the **runner-up MLP** (``RUNNER_UP_MLP``): a cheaper alternative, reported
  for comparison only.
* **linear** least squares: the baseline.

**The selected model does not change because of these results.** Model
selection was finished using nested cross-validation only. If the runner-up
scores better on the hold-out, that is reported, but the final MLP stays the
selected model: choosing a model *because* it did well on the hold-out would
turn the hold-out into another round of model selection, and it would no
longer be an honest estimate of performance on unseen data. For the same
reason, nothing is re-tuned, no threshold changes, and there is no second run
to "improve" the numbers. If a criterion fails, it is reported as FAIL.

To make the once-only rule visible, the script refuses to run if
``results/holdout_metrics.csv`` already exists, unless ``--force`` is given::

    python src/run_final.py            # about 10 seconds
    python src/run_final.py --force    # only to regenerate identical results

Because every model has a fixed ``random_state`` and the same training rows,
a forced re-run reproduces the same numbers; it cannot change the outcome.

Writes to ``results/``:

* ``holdout_metrics.csv``: RMSE, MAE, R², bias, timings and epochs per model.
* ``holdout_breakdown.csv``: RMSE by chord, velocity, angle and frequency band.
* ``holdout_setups.csv``: the hold-out setups (chord, velocity, angle, rows).
* ``holdout_setup_rmse.csv``: RMSE per hold-out setup and model, ranked.
* ``holdout_paired.csv``: per-setup differences, final MLP − linear and final
  MLP − runner-up.
* ``holdout_predictions.csv``: every hold-out row with each model's prediction.
* ``holdout_vs_cv.csv``: each model's CV estimate next to its hold-out RMSE.
* ``acceptance.csv``: every acceptance criterion, now PASS or FAIL.
* ``holdout_run_info.txt``: when and with what code and settings it was run.
* ``final_models/*.joblib``: the three trained models.

Rounded tables and PNG + PDF figures go to ``doc/generated/``.
"""

import argparse
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from functions.data_loader import load_airfoil_data
from functions.final import (
    HOLDOUT_MODELS,
    SELECTED_MODEL,
    as_result,
    cv_estimates,
    evaluate_acceptance,
    fit_and_predict,
    holdout_breakdown,
    holdout_metrics,
    holdout_setups,
    paired_setups,
    paired_summary,
    predictions_table,
    setup_rmse,
    split_train_holdout,
)
from functions.make_folds import TEST_SEED
from functions.models import FINAL_FEATURE_SET, FINAL_MODEL_NAME, RUNNER_UP_MODEL_NAME
from functions.physics_features import add_physics_features
from functions.plot_final import HOLDOUT_COLOURS, plot_cv_vs_holdout
from functions.plot_results import plot_predicted_vs_measured, plot_rmse_breakdown, plot_setup_spectra
from functions.plot_tuning import save_png_pdf
from run_cv import DOC_TABLES_DIR, RESULTS_DIR

#: File whose existence means the hold-out has already been evaluated.
GUARD_FILE = "holdout_metrics.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``--force`` option.

    Args:
        argv: Arguments to parse; ``None`` reads the command line.

    Returns:
        Namespace with ``force``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="run even though hold-out results already exist")
    return parser.parse_args(argv)


def _git_commit() -> str:
    """Current git commit hash, flagged if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return commit + (" (with uncommitted changes in src/)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (git not available)"


def run_info_text(fits, train: pd.DataFrame, holdout: pd.DataFrame) -> str:
    """Describe when and how the hold-out evaluation was run.

    Args:
        fits: The :class:`functions.final.HoldoutFit` objects.
        train: Training rows.
        holdout: Hold-out rows.

    Returns:
        Plain text for ``holdout_run_info.txt``.
    """
    lines = [
        "Final hold-out evaluation",
        f"Run at: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Git commit: {_git_commit()}",
        f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}",
        f"Platform: {platform.platform()}; logical CPUs: {os.cpu_count()}",
        f"Feature set: {FINAL_FEATURE_SET}; selected model: {SELECTED_MODEL}",
        f"Training rows (all CV rows): {len(train)} rows, {train['setup_id'].nunique()} setups",
        f"Hold-out rows: {len(holdout)} rows, {holdout['setup_id'].nunique()} setups",
        "",
        "Model settings (estimator inside the scaling pipeline):",
    ]
    for fit in fits:
        estimator = fit.model.regressor_.named_steps["regressor"]
        params = {k: v for k, v in sorted(estimator.get_params().items())}
        lines.append(f"- {fit.model_name}: {type(estimator).__name__}({params})")
    return "\n".join(lines) + "\n"


def write_doc_tables(metrics, estimates, acceptance, breakdown, setups, paired_stats, doc_dir: Path) -> None:
    """Write rounded hold-out tables for the HTML results page.

    Args:
        metrics: Output of :func:`functions.final.holdout_metrics`.
        estimates: CV estimates joined with hold-out RMSE.
        acceptance: The evaluated acceptance table.
        breakdown: Output of :func:`functions.final.holdout_breakdown`.
        setups: Output of :func:`functions.final.holdout_setups`.
        paired_stats: Rows of :func:`functions.final.paired_summary`.
        doc_dir: Destination directory.
    """
    doc_dir.mkdir(parents=True, exist_ok=True)
    labels = {m: HOLDOUT_MODELS[m] + (" (selected)" if m == SELECTED_MODEL else "") for m in HOLDOUT_MODELS}
    pd.DataFrame({
        "Model": metrics["model"].map(labels),
        "Hold-out RMSE (dB)": metrics["rmse"].round(3),
        "MAE (dB)": metrics["mae"].round(3),
        "R²": metrics["r2"].round(3),
        "Bias (dB)": metrics["bias"].round(3),
        "Fit time (s)": metrics["fit_time_s"].round(3),
        "Predict time (ms)": (metrics["predict_time_s"] * 1e3).round(2),
        "Epochs": metrics["n_iter"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}"),
    }).to_csv(doc_dir / "holdout_metrics.csv", index=False)

    pd.DataFrame({
        "Model": estimates["model"].map(labels),
        "CV estimate": estimates["estimate"],
        "CV RMSE (dB)": estimates["cv_rmse"].round(3).astype(str) + " ± " + estimates["cv_sd"].round(3).astype(str),
        "Hold-out RMSE (dB)": estimates["rmse"].round(3),
    }).to_csv(doc_dir / "holdout_vs_cv.csv", index=False)

    acceptance.rename(columns=str.capitalize).to_csv(doc_dir / "acceptance.csv", index=False)

    wide = breakdown.pivot_table(index=["breakdown", "group", "n_setups", "n"], columns="model",
                                 values="rmse", sort=False).reset_index()
    wide = wide.rename(columns={"breakdown": "Grouped by", "group": "Group", "n_setups": "Setups",
                                "n": "Rows", **{m: f"{HOLDOUT_MODELS[m]} RMSE (dB)" for m in HOLDOUT_MODELS}})
    wide.columns.name = None
    wide.round(2).to_csv(doc_dir / "holdout_breakdown.csv", index=False)

    pd.DataFrame({
        "Setup": setups["setup_id"], "Chord (m)": setups["chord_length_m"],
        "Velocity (m/s)": setups["velocity_mps"], "Angle (deg)": setups["attack_angle_deg"],
        "Frequencies": setups["n_rows"],
    }).to_csv(doc_dir / "holdout_setups.csv", index=False)

    stats = pd.DataFrame(paired_stats)
    pd.DataFrame({
        "Comparison (per setup)": stats["comparison"].map(
            lambda c: c.replace(FINAL_MODEL_NAME, "final MLP").replace(RUNNER_UP_MODEL_NAME, "runner-up MLP")),
        "Setups": stats["n_setups"],
        "Mean ± SD of RMSE difference (dB)": stats["mean_diff"].round(2).astype(str) + " ± "
                                             + stats["sd_diff"].round(2).astype(str),
        "Setups final MLP better": stats["a_wins"],
        "Setups other model better": stats["b_wins"],
    }).to_csv(doc_dir / "holdout_paired.csv", index=False)


def main(argv: list[str] | None = None, results_dir: Path = RESULTS_DIR,
         doc_dir: Path = DOC_TABLES_DIR) -> int:
    """Evaluate the three models on the hold-out once and save everything.

    Args:
        argv: Command-line arguments (``None`` reads the real ones).
        results_dir: Where results go (a temporary folder in the tests).
        doc_dir: Where doc tables and figures go.

    Returns:
        0 on success, 1 if it refused to run because results already exist.
    """
    args = parse_args(argv)
    if (results_dir / GUARD_FILE).exists() and not args.force:
        print(f"Refusing to run: {results_dir / GUARD_FILE} already exists.\n"
              "The hold-out is meant to be evaluated once. Use --force only to regenerate "
              "the same results (fixed seeds make them identical).")
        return 1

    data = add_physics_features(load_airfoil_data())
    # All CV rows to train, hold-out rows to test; the split is checked to be clean.
    train, holdout = split_train_holdout(data)
    print(f"Training on {len(train)} rows ({train['setup_id'].nunique()} setups); "
          f"testing on {len(holdout)} hold-out rows ({holdout['setup_id'].nunique()} setups)")

    # Train each model once and predict the hold-out once.
    fits = [fit_and_predict(name, train, holdout) for name in HOLDOUT_MODELS]
    results = {fit.model_name: as_result(fit, train, holdout) for fit in fits}

    metrics = holdout_metrics(fits, holdout)
    breakdown = holdout_breakdown(results, holdout)
    setups = holdout_setups(holdout)
    per_setup = setup_rmse(results, data)
    paired = pd.concat([paired_setups(per_setup, FINAL_MODEL_NAME, other)
                        for other in ("linear", RUNNER_UP_MODEL_NAME)], ignore_index=True)
    paired_stats = [paired_summary(g) for _, g in paired.groupby("comparison", sort=False)]
    estimates = cv_estimates(pd.read_csv(results_dir / "tuning_summary.csv"),
                             pd.read_csv(results_dir / "cv_summary.csv"),
                             pd.read_csv(results_dir / "tuning_inner_grid.csv")
                             ).merge(metrics[["model", "rmse"]], on="model")
    # Thresholds are the ones committed before this run; only FINAL_MLP is assessed.
    acceptance = evaluate_acceptance(pd.read_csv(results_dir / "tuning_paired_vs_linear.csv"),
                                     pd.read_csv(results_dir / "tuning_summary.csv"),
                                     pd.read_csv(results_dir / "physics_speed_order_summary.csv"),
                                     metrics)

    with pd.option_context("display.width", 200):
        print("\nHold-out metrics (dB)")
        print(metrics.drop(columns=["label"]).round(3).to_string(index=False))
        print("\nCV estimate vs hold-out")
        print(estimates.round(3).to_string(index=False))
        print("\nPaired per-setup comparison (negative = final MLP better)")
        print(pd.DataFrame(paired_stats).round(3).to_string(index=False))
        print("\nAcceptance criteria (thresholds fixed before this run)")
        print(acceptance[["criterion", "threshold", "value", "result"]].to_string(index=False))

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / GUARD_FILE, index=False)
    breakdown.to_csv(results_dir / "holdout_breakdown.csv", index=False)
    setups.to_csv(results_dir / "holdout_setups.csv", index=False)
    per_setup.to_csv(results_dir / "holdout_setup_rmse.csv", index=False)
    paired.to_csv(results_dir / "holdout_paired.csv", index=False)
    predictions_table(holdout, fits).to_csv(results_dir / "holdout_predictions.csv", index=False)
    estimates.to_csv(results_dir / "holdout_vs_cv.csv", index=False)
    acceptance.to_csv(results_dir / "acceptance.csv", index=False)
    (results_dir / "holdout_run_info.txt").write_text(run_info_text(fits, train, holdout), encoding="utf-8")
    # Save the trained models so the results can be reproduced without retraining.
    model_dir = results_dir / "final_models"
    model_dir.mkdir(exist_ok=True)
    for fit in fits:
        joblib.dump(fit.model, model_dir / f"{fit.model_name}.joblib")

    write_doc_tables(metrics, estimates, acceptance, breakdown, setups, paired_stats, doc_dir)
    ordered = [results[m] for m in HOLDOUT_MODELS]
    colours = [HOLDOUT_COLOURS[m] for m in HOLDOUT_MODELS]
    # Hold-out predictions are stored under seed TEST_SEED, so select that seed.
    save_png_pdf(plot_predicted_vs_measured(ordered, seed=TEST_SEED, colours=colours,
                                            title_note="hold-out"),
                 doc_dir, "holdout_pred_vs_measured")
    final_rmse = per_setup[per_setup["model"] == FINAL_MODEL_NAME].set_index("setup_id")["rmse"]
    titles = [f"c={r.chord_length_m:.4f} m, U={r.velocity_mps:.1f} m/s, α={r.attack_angle_deg:.1f}°\n"
              f"final MLP RMSE {final_rmse[r.setup_id]:.2f} dB" for r in setups.itertuples()]
    save_png_pdf(plot_setup_spectra(ordered, data, setups["setup_id"].tolist(), titles=titles,
                                    ncols=4, colours=colours), doc_dir, "holdout_spectra")
    save_png_pdf(plot_rmse_breakdown(ordered, colours=colours), doc_dir, "holdout_rmse_breakdown")
    save_png_pdf(plot_cv_vs_holdout(estimates.drop(columns="rmse"), metrics), doc_dir, "holdout_vs_cv")

    print(f"\nResults written to {results_dir}; doc tables and figures to {doc_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
