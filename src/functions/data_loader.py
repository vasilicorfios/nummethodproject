"""Load the NASA/UCI Airfoil Self-Noise dataset.

The dataset contains 1503 aerodynamic/acoustic measurements of NACA 0012
airfoils taken in a wind tunnel. Five input features describe the test
conditions and one target, the scaled sound pressure level, describes the
measured noise.

The Excel file is expected at ``data/airfoil_self_noise.xlsx``, where it is
committed to the repository (it is small, about 75 kB), so a fresh clone can
run every script.

Example:
    >>> from functions.data_loader import load_airfoil_data
    >>> df = load_airfoil_data()
    >>> df.columns.tolist()[-1]
    'spl_db'
"""

from pathlib import Path

import pandas as pd

#: Default location of the dataset: ``<project root>/data/airfoil_self_noise.xlsx``.
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "airfoil_self_noise.xlsx"

#: Mapping from the verbose source column headers to short, code-friendly names.
#: Unit suffixes (``_hz``, ``_deg``, ``_m``, ``_mps``, ``_db``) are kept in the
#: new names so units are never ambiguous downstream.
COLUMN_RENAME = {
    "frequency (Hz)": "frequency_hz",
    "attack-angle (deg)": "attack_angle_deg",
    "chord-length (m)": "chord_length_m",
    "free-stream-velocity (m/s)": "velocity_mps",
    "suction-side-displacement-thickness (m)": "disp_thickness_m",
    "scaled-sound-pressure (dB)": "spl_db",
}

#: Name of the target column (scaled sound pressure level, dB).
TARGET = "spl_db"

#: Input feature columns, in the order they appear in the dataset.
FEATURES = [
    "frequency_hz",
    "attack_angle_deg",
    "chord_length_m",
    "velocity_mps",
    "disp_thickness_m",
]


def load_airfoil_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the airfoil self-noise dataset.

    Reads the Excel file, drops the ``Source.Name`` column that Power Query
    style exports add (if present), and renames the remaining columns
    according to :data:`COLUMN_RENAME`.

    Args:
        path: Location of the ``.xlsx`` file. Defaults to :data:`DATA_PATH`.

    Returns:
        A DataFrame with one row per measurement and the columns
        ``frequency_hz``, ``attack_angle_deg``, ``chord_length_m``,
        ``velocity_mps``, ``disp_thickness_m`` (features) and ``spl_db``
        (target).

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    df = pd.read_excel(path)
    # errors="ignore": the column only exists in some exports of the file.
    df = df.drop(columns=["Source.Name"], errors="ignore")
    df = df.rename(columns=COLUMN_RENAME)
    return df


if __name__ == "__main__":
    # Quick sanity check when run directly: shape, first rows, summary stats.
    data = load_airfoil_data()
    print(data.shape)
    print(data.head())
    print(data.describe())
