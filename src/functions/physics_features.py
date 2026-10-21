"""Physics-derived features for the airfoil self-noise dataset.

The raw inputs are dimensional, but airfoil noise is governed largely by
dimensionless groups. This module adds three engineered features computed
row-wise from the raw columns, so they are recomputed on every load rather
than stored in the Excel file.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> from functions.physics_features import add_physics_features
    >>> df = add_physics_features(load_airfoil_data())
    >>> df[["log_frequency", "strouhal", "reynolds"]].shape[1]
    3
"""

import numpy as np
import pandas as pd

#: Kinematic viscosity of air at roughly 20 °C (m^2/s), used for the
#: Reynolds number. The wind-tunnel temperature is not recorded in the
#: dataset, so a standard room-temperature value is assumed.
KINEMATIC_VISCOSITY_M2PS = 1.5e-5

#: Names of the columns added by :func:`add_physics_features`.
PHYSICS_FEATURES = [
    "log_frequency",
    "strouhal",
    "reynolds",
]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataset with physics-derived feature columns.

    Adds:

    * ``log_frequency``: ``log10(frequency_hz)``. Frequencies span about two
      decades, and acoustic spectra are naturally read on a log axis.
    * ``strouhal``: ``frequency_hz * disp_thickness_m / velocity_mps``. Uses
      the suction-side displacement thickness as the length scale, the
      scaling used for turbulent boundary-layer trailing-edge noise in the
      Brooks-Pope-Marcolini (BPM) model.
    * ``reynolds``: ``velocity_mps * chord_length_m / nu``, the chord-based
      Reynolds number with ``nu`` = :data:`KINEMATIC_VISCOSITY_M2PS`.

    Each feature depends only on its own row, so adding them before the
    train/validation/test split does not leak information between sets.

    Args:
        df: Dataset as returned by
            :func:`functions.data_loader.load_airfoil_data`.

    Returns:
        A new DataFrame with the original columns plus the columns listed in
        :data:`PHYSICS_FEATURES`. The input DataFrame is not modified.
    """
    out = df.copy()
    out["log_frequency"] = np.log10(out["frequency_hz"])
    out["strouhal"] = out["frequency_hz"] * out["disp_thickness_m"] / out["velocity_mps"]
    out["reynolds"] = out["velocity_mps"] * out["chord_length_m"] / KINEMATIC_VISCOSITY_M2PS
    return out
