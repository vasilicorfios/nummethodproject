"""Entry point that cross-validates models on feature sets and reports results.

Every combination of the requested models and feature sets is evaluated with
repeated grouped 5-fold CV (:func:`functions.evaluate.cross_validate`). The
test hold-out is never used. Defaults run the linear baseline on sets A and
B::

    python src/run_cv.py
    python src/run_cv.py --models linear --feature-sets A

Results are printed and written as CSV files to ``results/``:

* ``cv_summary.csv``: one row per model/feature set (validation RMSE, MAE,
  R² as mean and std over all 15 folds, 5 folds x 3 seeds; mean fit and
  predict time per fold).
* ``cv_fold_scores.csv``: scores and timings for every seed and fold.
* ``cv_rmse_breakdown.csv``: OOF RMSE by chord, velocity, angle band and
  frequency band.
* ``cv_coefficients.csv``: for linear models, each input's effect in dB per
  standard deviation (mean and spread over all folds and seeds).

Rounded copies of the summary, breakdown and coefficient tables, and the
results figures (:mod:`functions.plot_results`), are also written to
``doc/generated/`` for the HTML results page; rebuild the docs afterwards to
publish them.
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
from functions.plot_results import save_result_figures

#: Directory the CSV results are written to: ``<project root>/results``.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Directory for rounded, presentation-ready tables included by the Sphinx
#: results page (``doc/results.rst``): ``<project root>/doc/generated``.
DOC_TABLES_DIR = Path(__file__).resolve().parent.parent / "doc" / "generated"


def _mean_std(mean: pd.Series, std: pd.Series, decimals: int) -> pd.Series:
    """Format paired mean and std columns as ``"mean ± std"`` strings."""
    return mean.round(decimals).map(f"{{:.{decimals}f}}".format) + " ± " + std.map(
        f"{{:.{decimals}f}}".format
    )


def write_doc_tables(
    summary: pd.DataFrame, breakdown: pd.DataFrame, coefficients: pd.DataFrame
) -> None:
    """Write the CV results as rounded CSV tables for the HTML documentation.

    The full-precision CSVs in :data:`RESULTS_DIR` are for analysis; these
    copies are rounded, labelled with units and laid out with one column per
    model/feature set so ``doc/results.rst`` can show them with
    ``csv-table`` directives. Re-running this script and rebuilding the docs
    keeps the published numbers in sync with the code.

    Args:
        summary: One row per model/feature set from
            :func:`functions.evaluate.summarise`.
        breakdown: Concatenated :func:`functions.evaluate.oof_rmse_breakdown`
            outputs with ``model`` and ``feature_set`` columns.
        coefficients: Concatenated :func:`functions.evaluate.coefficient_summary`
            outputs (reset index) with ``model`` and ``feature_set`` columns;
            may be empty.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Model": summary["model"],
        "Feature set": summary["feature_set"],
        "RMSE (dB)": _mean_std(summary["rmse"], summary["rmse_std"], 3),
        "MAE (dB)": _mean_std(summary["mae"], summary["mae_std"], 3),
        "R²": _mean_std(summary["r2"], summary["r2_std"], 3),
        "Fit time per fold (ms)": (summary["fit_time_s"] * 1e3).round(2),
        "Predict time per fold (ms)": (summary["predict_time_s"] * 1e3).round(2),
    }).to_csv(DOC_TABLES_DIR / "cv_summary.csv", index=False)

    # One RMSE column per model/feature set, so rows compare like for like.
    labelled = breakdown.assign(column=breakdown["model"] + " / " + breakdown["feature_set"])
    wide = labelled.pivot_table(
        index=["breakdown", "group", "n"], columns="column", values="rmse", sort=False
    ).round(3).reset_index()
    wide["breakdown"] = wide["breakdown"].map({
        "chord_length_m": "Chord (m)",
        "velocity_mps": "Velocity (m/s)",
        "angle_band": "Angle band (deg)",
        "frequency_band": "Frequency band",
    })
    wide = wide.rename(columns={"breakdown": "Grouped by", "group": "Group", "n": "Rows"})
    wide.columns.name = None
    wide.to_csv(DOC_TABLES_DIR / "cv_rmse_breakdown.csv", index=False)

    if not coefficients.empty:
        cells = coefficients.assign(
            column=coefficients["model"] + " / " + coefficients["feature_set"],
            cell=_mean_std(coefficients["effect_db_per_sd"], coefficients["std"], 2)
            + " [" + (coefficients["sign_agreement"] * 100).round().astype(int).astype(str) + "%]",
        )
        table = cells.pivot(index="feature", columns="column", values="cell")
        # Keep feature-set order (raw inputs first), not alphabetical.
        table = table.reindex(cells["feature"].drop_duplicates()).fillna("")
        table.index.name = "Feature"
        table.columns.name = None
        table.reset_index().to_csv(DOC_TABLES_DIR / "cv_coefficients.csv", index=False)


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

    results, summaries, fold_scores, breakdowns, coefficients = [], [], [], [], []
    for model_name in args.models:
        for feature_set in args.feature_sets:
            result = cross_validate(data, model_name, feature_set)
            results.append(result)
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
        print("Validation metrics (mean and std over all folds of all seeds; times are mean per fold)")
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

    coefficient_table = (
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    if not coefficient_table.empty:
        coefficient_table.to_csv(RESULTS_DIR / "cv_coefficients.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    pd.concat(fold_scores, ignore_index=True).to_csv(RESULTS_DIR / "cv_fold_scores.csv", index=False)
    breakdown.to_csv(RESULTS_DIR / "cv_rmse_breakdown.csv", index=False)
    write_doc_tables(summary, breakdown, coefficient_table)
    save_result_figures(data, results, DOC_TABLES_DIR)
    print(f"\nCSV files written to {RESULTS_DIR}")
    print(f"Documentation tables and figures written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
