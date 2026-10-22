"""Entry point that profiles runtime, memory, size and FLOPs of the models.

By default it profiles the models compared in the report, all in one
session on the same machine so the numbers are comparable:

* linear on feature sets A and B (the baseline);
* the final MLP (:data:`functions.models.FINAL_MLP`) on
  :data:`functions.models.FINAL_FEATURE_SET`;
* the cheaper runner-up MLP (:data:`functions.models.RUNNER_UP_MLP`) on the
  same feature set.

Every model is trained on all cross-validation rows, never the test
hold-out (:func:`functions.profiling.profile_model`)::

    python src/run_profile.py                                   # the four above, ~1 min
    python src/run_profile.py --models linear --feature-sets B  # a custom selection

Writes to ``results/``:

* ``profile_summary.csv``: one row per model/feature set: median and
  minimum fit and predict times, peak memory, model size, weights, epochs,
  estimated FLOPs and achieved FLOP rate.
* ``profile_comparison.csv``: final MLP vs linear (B) and vs the runner-up,
  as ratios.
* ``profile_machine.csv``: CPU, operating system, library versions and
  thread settings used.
* ``profile_<model>_<feature set>.txt``: the cProfile report of one fit.

Rounded copies of the summary, comparison and machine tables are written to
``doc/generated/`` for the HTML results page.
"""

import argparse

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.feature_sets import FEATURE_SETS
from functions.models import FINAL_FEATURE_SET, FINAL_MODEL_NAME, MODELS, RUNNER_UP_MODEL_NAME
from functions.physics_features import add_physics_features
from functions.profiling import machine_info, profile_model
from run_cv import DOC_TABLES_DIR, RESULTS_DIR

#: (model, feature set) pairs profiled by default.
PROFILE_PLAN = [
    ("linear", "A"),
    ("linear", "B"),
    (FINAL_MODEL_NAME, FINAL_FEATURE_SET),
    (RUNNER_UP_MODEL_NAME, FINAL_FEATURE_SET),
]

#: Quantities compared between models in :func:`compare`.
COMPARED = {
    "fit_time_s": "Fit time (median)",
    "predict_time_s": "Predict time (median)",
    "peak_fit_bytes": "Peak fit memory",
    "model_bytes": "Model size (pickled)",
    "fit_flops": "Fit FLOPs (estimate)",
    "n_weights": "Weights",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional ``--models`` and ``--feature-sets`` options.

    Args:
        argv: Arguments to parse; ``None`` reads the command line.

    Returns:
        Namespace with ``models`` and ``feature_sets`` (``None`` when not
        given, meaning :data:`PROFILE_PLAN` is used).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        help="models to profile (default: the report's four combinations)")
    parser.add_argument("--feature-sets", nargs="+", choices=sorted(FEATURE_SETS))
    return parser.parse_args(argv)


def profile_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The (model, feature set) pairs to profile.

    Args:
        args: Output of :func:`parse_args`.

    Returns:
        :data:`PROFILE_PLAN` if no options were given, otherwise every
        combination of the given models and feature sets (missing options
        default to ``linear`` and both feature sets).
    """
    if args.models is None and args.feature_sets is None:
        return list(PROFILE_PLAN)
    models = args.models or ["linear"]
    feature_sets = args.feature_sets or ["A", "B"]
    return [(m, fs) for m in models for fs in feature_sets]


def compare(summary: pd.DataFrame, numerator: tuple[str, str],
            denominator: tuple[str, str]) -> pd.DataFrame:
    """Ratios of costs between two profiled models.

    Args:
        summary: The profile summary table.
        numerator: ``(model, feature set)`` on top of the ratio.
        denominator: ``(model, feature set)`` underneath.

    Returns:
        One row per quantity in :data:`COMPARED` with ``comparison``,
        ``quantity`` and ``ratio`` (e.g. 40 means 40 times larger).
    """
    rows = summary.set_index(["model", "feature_set"])
    top, bottom = rows.loc[numerator], rows.loc[denominator]
    label = f"{numerator[0]} ({numerator[1]}) / {denominator[0]} ({denominator[1]})"
    return pd.DataFrame([
        {"comparison": label, "quantity": name, "ratio": top[col] / bottom[col]}
        for col, name in COMPARED.items()
    ])


def write_doc_tables(summary: pd.DataFrame, comparison: pd.DataFrame,
                     machine: dict[str, str]) -> None:
    """Write rounded profile tables, with units in the headers, for the docs.

    Args:
        summary: One row per model/feature set from
            :meth:`functions.profiling.ProfileResult.summary`.
        comparison: Output of :func:`compare` (possibly several, joined).
        machine: Output of :func:`functions.profiling.machine_info`.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def optional(series, fmt):
        # Blank cells (not "nan") for quantities a model does not have.
        return series.map(lambda v: "" if pd.isna(v) else fmt.format(v))

    pd.DataFrame({
        "Model": summary["model"],
        "Feature set": summary["feature_set"],
        "Weights": optional(summary["n_weights"], "{:,.0f}"),
        "Epochs": optional(summary["n_iter"], "{:.0f}"),
        "Fit time, median (ms)": (summary["fit_time_s"] * 1e3).round(1),
        "Fit time, min (ms)": (summary["fit_time_min_s"] * 1e3).round(1),
        "Predict time, median (ms)": (summary["predict_time_s"] * 1e3).round(2),
        "Peak fit memory (MB)": (summary["peak_fit_bytes"] / 1e6).round(2),
        "Weights (kB)": optional(summary["weight_bytes"] / 1e3, "{:,.2f}"),
        "Pickled model (kB)": (summary["model_bytes"] / 1e3).round(1),
        "Fit FLOPs, estimate (MFLOP)": optional(summary["fit_flops"] / 1e6, "{:,.1f}"),
        "Fit rate (MFLOP/s)": optional(summary["fit_mflops_per_s"], "{:,.0f}"),
    }).to_csv(DOC_TABLES_DIR / "profile_summary.csv", index=False)

    if not comparison.empty:
        comparison.assign(ratio=comparison["ratio"].map("{:.1f}×".format)).rename(columns={
            "comparison": "Comparison", "quantity": "Quantity", "ratio": "Ratio",
        }).to_csv(DOC_TABLES_DIR / "profile_comparison.csv", index=False)

    pd.DataFrame({"Item": list(machine), "Value": list(machine.values())}).to_csv(
        DOC_TABLES_DIR / "profile_machine.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    """Profile the requested models, print a comparison and save the results."""
    args = parse_args(argv)
    data = add_physics_features(load_airfoil_data())
    RESULTS_DIR.mkdir(exist_ok=True)
    machine = machine_info()

    rows = []
    for model_name, feature_set in profile_plan(args):
        print(f"Profiling {model_name} on set {feature_set} ...")
        result = profile_model(data, model_name, feature_set)
        rows.append(result.summary())
        profile_path = RESULTS_DIR / f"profile_{model_name}_{feature_set}.txt"
        profile_path.write_text(result.profile_text)
    summary = pd.DataFrame(rows)

    # Compare the final MLP with linear on the same features, and with the
    # cheaper runner-up, whenever all three were profiled.
    final = (FINAL_MODEL_NAME, FINAL_FEATURE_SET)
    pairs = [(final, ("linear", FINAL_FEATURE_SET)), (final, (RUNNER_UP_MODEL_NAME, FINAL_FEATURE_SET))]
    present = set(zip(summary["model"], summary["feature_set"]))
    comparisons = [compare(summary, a, b) for a, b in pairs if a in present and b in present]
    comparison = pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()

    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print("\nProfile of one fit on all CV rows "
              "(times: median and min of repeats; FLOPs are estimates)")
        print(summary.to_string(index=False))
        if not comparison.empty:
            print("\nComparison (ratios; e.g. 40 = 40 times larger)")
            print(comparison.to_string(index=False))
    print("\nMachine: " + "; ".join(f"{k} = {v}" for k, v in machine.items()))

    summary.to_csv(RESULTS_DIR / "profile_summary.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "profile_comparison.csv", index=False)
    pd.DataFrame({"item": list(machine), "value": list(machine.values())}).to_csv(
        RESULTS_DIR / "profile_machine.csv", index=False)
    write_doc_tables(summary, comparison, machine)
    print(f"\nSummary and cProfile reports written to {RESULTS_DIR}")
    print(f"Documentation tables written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
