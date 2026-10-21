"""Entry point that profiles runtime, memory and FLOPs of models.

Each requested model/feature-set pair is trained on all cross-validation
rows (never the test hold-out) by :func:`functions.profiling.profile_model`::

    python src/run_profile.py
    python src/run_profile.py --models linear --feature-sets B

Writes to ``results/``:

* ``profile_summary.csv``: time, peak memory, estimated FLOPs and achieved
  FLOP rate per model/feature set.
* ``profile_<model>_<feature set>.txt``: the cProfile report of one fit.

A rounded copy of the summary is written to ``doc/generated/`` for the HTML
results page.
"""

import argparse

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.feature_sets import FEATURE_SETS
from functions.models import MODELS
from functions.physics_features import add_physics_features
from functions.profiling import profile_model
from run_cv import DOC_TABLES_DIR, RESULTS_DIR


def write_doc_table(summary: pd.DataFrame) -> None:
    """Write the profile summary as a rounded table for the HTML documentation.

    Times are converted to ms and memory to kB so the table reads at a glance
    on ``doc/results.rst``.

    Args:
        summary: One row per model/feature set from
            :meth:`functions.profiling.ProfileResult.summary`.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Model": summary["model"],
        "Feature set": summary["feature_set"],
        "Rows x features": summary["n_rows"].astype(str) + " x " + summary["n_features"].astype(str),
        "Fit time (ms)": (summary["fit_time_s"] * 1e3).round(2),
        "Predict time (ms)": (summary["predict_time_s"] * 1e3).round(2),
        "Input size (kB)": (summary["input_bytes"] / 1e3).round(1),
        "Peak fit memory (kB)": (summary["peak_fit_bytes"] / 1e3).round(1),
        "Fit FLOPs (est.)": summary["fit_flops"].map("{:.2e}".format),
        "Fit rate (MFLOP/s)": summary["fit_mflops_per_s"].round(1),
    }).to_csv(DOC_TABLES_DIR / "profile_summary.csv", index=False)


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
    """Profile every model/feature-set pair, print and save the results."""
    args = parse_args()
    data = add_physics_features(load_airfoil_data())
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for model_name in args.models:
        for feature_set in args.feature_sets:
            result = profile_model(data, model_name, feature_set)
            rows.append(result.summary())
            profile_path = RESULTS_DIR / f"profile_{model_name}_{feature_set}.txt"
            profile_path.write_text(result.profile_text)

    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print("Profile of one fit on all CV rows (times are medians of repeats)")
        print(summary.to_string(index=False))

    summary.to_csv(RESULTS_DIR / "profile_summary.csv", index=False)
    write_doc_table(summary)
    print(f"\nSummary and cProfile reports written to {RESULTS_DIR}")
    print(f"Documentation table written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
