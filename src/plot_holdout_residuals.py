"""Draw the hold-out residual plot from the saved hold-out predictions.

This only **reads** ``results/holdout_predictions.csv``, written by the single
run of ``src/run_final.py``. No model is trained and the hold-out is not
scored again, so the hold-out evaluation remains a single run::

    python src/plot_holdout_residuals.py    # a few seconds

Writes ``doc/generated/holdout_residuals.png`` and ``.pdf``
(see :func:`functions.plot_final.plot_holdout_residuals`).
"""

import pandas as pd

from functions.final import HOLDOUT_MODELS
from functions.plot_final import plot_holdout_residuals
from functions.plot_tuning import save_png_pdf
from run_cv import DOC_TABLES_DIR, RESULTS_DIR


def main() -> None:
    """Read the saved hold-out predictions and save the residual figure."""
    predictions = pd.read_csv(RESULTS_DIR / "holdout_predictions.csv")
    # Same panel order and names as the other hold-out figures.
    labels = {model: f"{name} / B" for model, name in HOLDOUT_MODELS.items()}
    paths = save_png_pdf(plot_holdout_residuals(predictions, labels), DOC_TABLES_DIR,
                         "holdout_residuals")
    print("Written: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
