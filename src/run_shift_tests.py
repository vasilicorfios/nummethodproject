"""Entry point for the robustness shift tests (leave one chord / velocity out).

Trains the final MLP (3 random seeds) and the linear baseline, both on the
final feature set, with every setup of one chord (or one velocity) held out
in turn, and scores them on the held-out setups. Only cross-validation rows
are used; the test hold-out is never touched::

    python src/run_shift_tests.py      # about 1-2 minutes

See :mod:`functions.shift_tests` for the method and its caveats.

Writes:

* ``results/shift_tests.csv``: one row per (shift, held-out value, model),
  with RMSE, MAE, R², bias, setups and rows tested, and the nested-CV
  reference RMSE.
* ``doc/generated/shift_tests.csv``: a rounded copy for the docs.
* ``doc/generated/shift_tests.png`` and ``.pdf``: the bar chart.
"""

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.physics_features import add_physics_features
from functions.plot_robustness import MODEL_STYLE, plot_shift_tests
from functions.plot_tuning import save_png_pdf
from functions.shift_tests import in_distribution_reference, run_shift_tests
from run_cv import DOC_TABLES_DIR, RESULTS_DIR


def write_doc_table(table: pd.DataFrame) -> None:
    """Write a rounded shift-test table for the HTML results page.

    Args:
        table: Shift-test results with the ``reference_rmse`` column added.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rmse = table["rmse"].map("{:.2f}".format)
    spread = table["rmse_sd_seeds"].map(lambda v: "" if pd.isna(v) else f" ± {v:.2f}")
    pd.DataFrame({
        "Shift test": table["shift"],
        "Held out": table["held_out"].map("{:g}".format),
        "Regime": table["regime"],
        "Model": table["model"].map(lambda m: MODEL_STYLE[m][1]),
        "Setups / rows": table["n_setups"].astype(str) + " / " + table["n_rows"].astype(str),
        "RMSE (dB)": rmse + spread,
        "Nested-CV RMSE (dB)": table["reference_rmse"].round(2),
        "MAE (dB)": table["mae"].round(2),
        "R²": table["r2"].round(2),
        "Bias (dB)": table["bias"].round(2),
    }).to_csv(DOC_TABLES_DIR / "shift_tests.csv", index=False)


def main() -> None:
    """Run the shift tests, print and save the results and the figure."""
    data = add_physics_features(load_airfoil_data())
    reference = in_distribution_reference(pd.read_csv(RESULTS_DIR / "tuning_summary.csv"))

    table = run_shift_tests(data)
    # How far each shift result is from the model's in-distribution accuracy.
    table["reference_rmse"] = table["model"].map(reference)
    table["rmse_increase"] = table["rmse"] - table["reference_rmse"]

    print("\nShift tests (RMSE in dB; MLP is the mean over 3 seeds)")
    print(table[["shift", "held_out", "regime", "model", "n_setups", "rmse", "rmse_sd_seeds",
                 "r2", "bias", "reference_rmse"]].round(3).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "shift_tests.csv", index=False)
    write_doc_table(table)
    save_png_pdf(plot_shift_tests(table, reference), DOC_TABLES_DIR, "shift_tests")
    print(f"\nResults written to {RESULTS_DIR / 'shift_tests.csv'}")
    print(f"Documentation table and figure written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
