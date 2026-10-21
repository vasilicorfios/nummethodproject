"""Entry point that profiles runtime, memory and FLOPs of models.

Each requested model/feature-set pair is trained on all cross-validation
rows (never the test hold-out) by :func:`functions.profiling.profile_model`::

    python src/run_profile.py
    python src/run_profile.py --models linear --feature-sets B

Writes to ``results/``:

* ``profile_summary.csv``: time, peak memory, estimated FLOPs and achieved
  FLOP rate per model/feature set.
* ``profile_<model>_<feature set>.txt``: the cProfile report of one fit.
"""

import argparse

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.feature_sets import FEATURE_SETS
from functions.models import MODELS
from functions.physics_features import add_physics_features
from functions.profiling import profile_model
from run_cv import RESULTS_DIR


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
    print(f"\nSummary and cProfile reports written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
