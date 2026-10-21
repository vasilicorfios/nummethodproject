"""Entry point that cross-validates models on feature sets and reports results.

Every combination of the requested models and feature sets is evaluated with
repeated grouped 5-fold CV (:func:`functions.evaluate.cross_validate`). The
test hold-out is never used. Defaults run the linear baseline on sets A and
B::

    python src/run_cv.py
    python src/run_cv.py --models linear --feature-sets A

Results are printed and written as CSV files to ``results/``:

* ``cv_summary.csv``: one row per model/feature set (OOF RMSE, MAE, R² as
  mean and std over seeds; mean fit and predict time per fold).
* ``cv_fold_scores.csv``: scores and timings for every seed and fold.
* ``cv_rmse_breakdown.csv``: OOF RMSE by chord, velocity, angle band and
  frequency band.
* ``cv_coefficients.csv``: for linear models, each input's effect in dB per
  standard deviation (mean and spread over all folds and seeds).
"""

import argparse
from pathlib import Path

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.evaluate import (
    coefficient_summary,
    cross_validate,
    oof_rmse_breakdown,
    summarise,
)
from functions.feature_sets import FEATURE_SETS
from functions.models import MODELS
from functions.physics_features import add_physics_features

#: Directory the CSV results are written to: ``<project root>/results``.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    """Parse the ``--models`` and ``--feature-sets`` command-line options.

    Returns:
        Namespace with ``models`` and ``feature_sets`` lists.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["linear"], choices=sorted(MODELS))
    parser.add_argument(
        "--feature-sets", nargs="+", default=["A", "B"], choices=sorted(FEATURE_SETS)
    )
    return parser.parse_args()


def main() -> None:
    """Run CV for every model/feature-set pair, print and save the results."""
    args = parse_args()
    # Physics columns are added for every run; feature set A simply ignores them.
    data = add_physics_features(load_airfoil_data())

    summaries, fold_scores, breakdowns, coefficients = [], [], [], []
    for model_name in args.models:
        for feature_set in args.feature_sets:
            result = cross_validate(data, model_name, feature_set)
            summaries.append(summarise(result))
            labels = {"model": model_name, "feature_set": feature_set}
            fold_scores.append(result.fold_scores.assign(**labels))
            breakdowns.append(oof_rmse_breakdown(result).assign(**labels))
            coefs = coefficient_summary(result)
            if not coefs.empty:
                coefficients.append(coefs.reset_index().assign(**labels))

    summary = pd.DataFrame(summaries)
    breakdown = pd.concat(breakdowns, ignore_index=True)

    with pd.option_context("display.width", 120, "display.float_format", "{:.4f}".format):
        print("Out-of-fold metrics (mean and std over CV seeds; times are mean per fold)")
        print(summary.to_string(index=False))
        print("\nOut-of-fold RMSE breakdown (dB)")
        # One column per model/feature set so they can be compared side by side.
        wide = breakdown.pivot_table(
            index=["breakdown", "group", "n"],
            columns=["model", "feature_set"],
            values="rmse",
            sort=False,
        )
        print(wide.to_string())

        for coefs in coefficients:
            model_name, feature_set = coefs["model"].iat[0], coefs["feature_set"].iat[0]
            print(f"\nCoefficients, {model_name} on set {feature_set} "
                  "(dB per input std; spread over all folds and seeds)")
            print(coefs.drop(columns=["model", "feature_set"]).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(
            RESULTS_DIR / "cv_coefficients.csv", index=False
        )
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    pd.concat(fold_scores, ignore_index=True).to_csv(RESULTS_DIR / "cv_fold_scores.csv", index=False)
    breakdown.to_csv(RESULTS_DIR / "cv_rmse_breakdown.csv", index=False)
    print(f"\nCSV files written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
