"""Predict SPL for new inputs with the final model, refusing bad inputs.

This is the safe way to use the trained model on new conditions:
:func:`predict_spl` first runs :func:`functions.input_checks.validate_inputs`
(missing, invalid and out-of-range inputs stop with an error), then adds the
physics features and predicts with the saved final MLP.

Example:
    >>> import pandas as pd
    >>> from functions.predict import predict_spl
    >>> predict_spl(pd.DataFrame({"frequency_hz": [1000], "attack_angle_deg": [5.0],
    ...     "chord_length_m": [0.1524], "velocity_mps": [55.5],
    ...     "disp_thickness_m": [0.0028]}))   # doctest: +SKIP
    array([131.46...])
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from functions.feature_sets import get_feature_columns
from functions.input_checks import validate_inputs
from functions.models import FINAL_FEATURE_SET, FINAL_MODEL_NAME
from functions.physics_features import add_physics_features

#: Where ``run_final.py`` saved the trained final model.
FINAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "results" / "final_models" / f"{FINAL_MODEL_NAME}.joblib"


def predict_spl(inputs: pd.DataFrame, model=None, model_path: Path = FINAL_MODEL_PATH) -> np.ndarray:
    """Predict scaled SPL (dB) for each row of ``inputs``.

    The inputs are checked **before** anything else, so a bad row stops the
    whole call with an error naming the problem; no prediction is made for any
    row in that case.

    Args:
        inputs: The five raw inputs (see
            :data:`functions.input_checks.INPUT_COLUMNS`), one row per prediction.
        model: A fitted model; by default the saved final MLP is loaded.
        model_path: Where to load the model from when ``model`` is None.

    Returns:
        Predicted SPL in dB, one value per input row.

    Raises:
        functions.input_checks.InputError: If any input is missing, invalid
            or out of range.
    """
    clean = validate_inputs(inputs)
    # The physics features (log f, Strouhal, Reynolds) need the checked inputs.
    features = add_physics_features(clean)[get_feature_columns(FINAL_FEATURE_SET)]
    if model is None:
        model = joblib.load(model_path)
    return np.asarray(model.predict(features))
