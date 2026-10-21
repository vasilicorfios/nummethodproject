"""Tests for :mod:`functions.physics_features`."""

import pandas as pd
import pytest

from functions.physics_features import (
    KINEMATIC_VISCOSITY_M2PS,
    PHYSICS_FEATURES,
    add_physics_features,
)


def test_hand_calculated_values():
    """Features match values worked out by hand for one row."""
    row = pd.DataFrame({
        "frequency_hz": [1000.0],
        "chord_length_m": [0.1],
        "velocity_mps": [50.0],
        "disp_thickness_m": [0.001],
    })
    out = add_physics_features(row)
    assert out["log_frequency"].iat[0] == pytest.approx(3.0)
    # St = f * delta / U = 1000 * 0.001 / 50
    assert out["strouhal"].iat[0] == pytest.approx(0.02)
    # Re = U * c / nu = 50 * 0.1 / 1.5e-5
    assert out["reynolds"].iat[0] == pytest.approx(50 * 0.1 / KINEMATIC_VISCOSITY_M2PS)


def test_adds_columns_without_modifying_input(raw_data):
    """Returns a copy with the new columns; the original is unchanged."""
    before = raw_data.copy()
    out = add_physics_features(raw_data)
    assert set(PHYSICS_FEATURES) <= set(out.columns)
    assert not set(PHYSICS_FEATURES) & set(raw_data.columns)
    pd.testing.assert_frame_equal(raw_data, before)


def test_features_finite(data):
    """No division by zero or log of non-positive values on the real data."""
    values = data[PHYSICS_FEATURES]
    assert values.notna().all().all()
    assert (values.abs() < float("inf")).all().all()
