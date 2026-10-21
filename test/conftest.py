"""Shared pytest fixtures.

Loading the Excel file takes about a second, so the dataset is loaded once
per test session and shared. Tests must not modify these DataFrames; every
project function that changes data returns a copy, and tests check that.
"""

import pandas as pd
import pytest

from functions.data_loader import load_airfoil_data
from functions.physics_features import add_physics_features


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    """The dataset as returned by :func:`load_airfoil_data`."""
    return load_airfoil_data()


@pytest.fixture(scope="session")
def data(raw_data) -> pd.DataFrame:
    """The dataset with physics features added, as used by every model."""
    return add_physics_features(raw_data)
