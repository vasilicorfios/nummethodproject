"""Entry point for the manufactured-solution verification checks.

Runs the linear and MLP manufactured-solution checks
(:mod:`functions.verification`) and saves a table and a figure for the HTML
results page. These checks use made-up data only; the airfoil dataset and
the test hold-out are not involved::

    python src/run_verification.py    # a few seconds

Writes:

* ``results/verification.csv`` and ``doc/generated/verification.csv``: one
  row per check, with value, pass mark and PASS/FAIL.
* ``doc/generated/verification_sine.png`` and ``.pdf``: the MLP sine check.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions.models import FINAL_MODEL_NAME
from functions.plot_final import HOLDOUT_COLOURS
from functions.plot_results import INK, INK_SECONDARY, MUTED, SURFACE, _style
from functions.plot_tuning import save_png_pdf
from functions.verification import (
    CONTROL_MIN_RMSE,
    LINEAR_MAX_REL_ERROR,
    SINE_MAX_RMSE,
    SINE_MIN_R2,
    check_linear,
    check_sine,
    manufactured_sine,
)
from run_cv import DOC_TABLES_DIR, RESULTS_DIR


def verification_table(linear: dict, mlp: dict, control: dict) -> pd.DataFrame:
    """One row per verification check, with value, pass mark and result.

    Args:
        linear: Output of :func:`functions.verification.check_linear`.
        mlp: :func:`functions.verification.check_sine` for the final MLP.
        control: :func:`functions.verification.check_sine` for linear.

    Returns:
        Columns ``check``, ``model``, ``quantity``, ``value``, ``pass_mark``
        and ``result``.
    """
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    return pd.DataFrame([
        {"check": "Linear law y = 3 + 2x1 - 5x2", "model": "linear",
         "quantity": "largest relative coefficient error",
         "value": f"{linear['max_coef_rel_error']:.1e}", "pass_mark": f"<= {LINEAR_MAX_REL_ERROR:.0e}",
         "result": verdict(linear["passed"])},
        {"check": "Linear law y = 3 + 2x1 - 5x2", "model": "linear",
         "quantity": "largest prediction error",
         "value": f"{linear['max_abs_pred_error']:.1e}", "pass_mark": "(round-off)", "result": ""},
        {"check": "y = sin(x), 200 points", "model": "final MLP",
         "quantity": "RMSE between training points",
         "value": f"{mlp['rmse']:.3f}", "pass_mark": f"<= {SINE_MAX_RMSE:g}",
         "result": verdict(mlp["rmse"] <= SINE_MAX_RMSE)},
        {"check": "y = sin(x), 200 points", "model": "final MLP",
         "quantity": "R² between training points",
         "value": f"{mlp['r2']:.4f}", "pass_mark": f">= {SINE_MIN_R2:g}",
         "result": verdict(mlp["r2"] >= SINE_MIN_R2)},
        {"check": "y = sin(x), 200 points", "model": "final MLP",
         "quantity": f"largest error (at x = {mlp['max_error_x']:.2f})",
         "value": f"{mlp['max_error']:.3f}", "pass_mark": "(reported only)", "result": ""},
        {"check": "y = sin(x), negative control", "model": "linear",
         "quantity": "RMSE between training points (must fail)",
         "value": f"{control['rmse']:.3f}", "pass_mark": f">= {CONTROL_MIN_RMSE:g}",
         "result": verdict(control["rmse"] >= CONTROL_MIN_RMSE)},
    ])


def plot_sine_check(mlp: dict, control: dict):
    """Sine fit (left) and the MLP's error along x (right).

    Args:
        mlp: :func:`functions.verification.check_sine` for the final MLP.
        control: The same for linear (the negative control).

    Returns:
        The figure (not shown or saved).
    """
    X, y, _, _ = manufactured_sine()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    fig.set_facecolor(SURFACE)
    for ax in (left, right):
        _style(ax)
        ax.set_xlabel("x")

    left.plot(mlp["x"], np.sin(mlp["x"]), color=INK, linewidth=1.5, label="true y = sin(x)")
    left.scatter(X["x"], y, s=8, color=MUTED, zorder=3, label="200 training points")
    left.plot(control["x"], control["prediction"], color=HOLDOUT_COLOURS["linear"], linewidth=2,
              label=f"linear (control), RMSE {control['rmse']:.2f}")
    left.plot(mlp["x"], mlp["prediction"], color=HOLDOUT_COLOURS[FINAL_MODEL_NAME], linewidth=2,
              linestyle="--", label=f"final MLP, RMSE {mlp['rmse']:.3f}")
    left.set_ylabel("y")
    left.set_title("Manufactured solution y = sin(x)", fontsize=10, loc="left")
    left.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower left")

    right.axhline(0, color=INK_SECONDARY, linewidth=1)
    right.plot(mlp["x"], mlp["prediction"] - np.sin(mlp["x"]),
               color=HOLDOUT_COLOURS[FINAL_MODEL_NAME], linewidth=2)
    right.set_ylabel("Final MLP error, predicted − true")
    right.set_title("Error between the training points", fontsize=10, loc="left")
    return fig


def main() -> None:
    """Run the checks, print the table and save the outputs."""
    linear = check_linear()
    mlp = check_sine(FINAL_MODEL_NAME)
    control = check_sine("linear")
    table = verification_table(linear, mlp, control)
    print(table.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "verification.csv", index=False)
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.rename(columns={"check": "Check", "model": "Model", "quantity": "Quantity",
                          "value": "Value", "pass_mark": "Pass mark", "result": "Result"}
                 ).to_csv(DOC_TABLES_DIR / "verification.csv", index=False)
    save_png_pdf(plot_sine_check(mlp, control), DOC_TABLES_DIR, "verification_sine")
    print(f"\nWritten to {RESULTS_DIR / 'verification.csv'} and {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
