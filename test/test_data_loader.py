"""Tests for :mod:`functions.data_loader`."""

import pytest

from functions.data_loader import FEATURES, TARGET, load_airfoil_data


def test_shape_and_columns(raw_data):
    """The UCI dataset has 1503 rows: five inputs then the target."""
    assert raw_data.shape == (1503, 6)
    assert raw_data.columns.tolist() == FEATURES + [TARGET]


def test_no_missing_values(raw_data):
    """Models cannot handle NaNs, so the loaded data must have none."""
    assert not raw_data.isna().any().any()


def test_values_in_physical_domain(raw_data):
    """Inputs lie in the documented wind-tunnel test ranges."""
    assert raw_data["frequency_hz"].between(200, 20000).all()
    assert raw_data["attack_angle_deg"].between(0, 22.2).all()
    assert raw_data["chord_length_m"].between(0.0254, 0.3048).all()
    assert raw_data["velocity_mps"].between(31.7, 71.3).all()
    assert (raw_data["disp_thickness_m"] > 0).all()


def test_missing_file_raises(tmp_path):
    """A wrong path fails loudly instead of returning empty data."""
    with pytest.raises(FileNotFoundError):
        load_airfoil_data(tmp_path / "does_not_exist.xlsx")
