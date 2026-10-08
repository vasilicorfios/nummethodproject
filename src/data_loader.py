"""Load the NASA/UCI Airfoil Self-Noise dataset.

Expects the Excel file at data/airfoil_self_noise.xlsx (not committed to
the repo — kept locally per the dataset-size note in the project README).
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "airfoil_self_noise.xlsx"

COLUMN_RENAME = {
    "frequency (Hz)": "frequency_hz",
    "attack-angle (deg)": "attack_angle_deg",
    "chord-length (m)": "chord_length_m",
    "free-stream-velocity (m/s)": "velocity_mps",
    "suction-side-displacement-thickness (m)": "disp_thickness_m",
    "scaled-sound-pressure (dB)": "spl_db",
}


def load_airfoil_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the airfoil self-noise dataset.

    Drops the source-file column and renames the remaining columns to
    short, code-friendly names (see COLUMN_RENAME).
    """
    df = pd.read_excel(path)
    df = df.drop(columns=["Source.Name"], errors="ignore")
    df = df.rename(columns=COLUMN_RENAME)
    return df


if __name__ == "__main__":
    data = load_airfoil_data()
    print(data.shape)
    print(data.head())
    print(data.describe())
