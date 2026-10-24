"""Tests for :mod:`functions.input_checks` and :mod:`functions.predict`.

Every kind of bad input must stop with the right error, naming the
offending column and row; valid inputs (including the exact range limits)
must pass unchanged. Nothing may ever be imputed.
"""

import numpy as np
import pandas as pd
import pytest

from functions.data_loader import FEATURES, load_airfoil_data
from functions.input_checks import (
    FEATURE_RANGES,
    INPUT_COLUMNS,
    InputError,
    InvalidInputError,
    MissingInputError,
    OutOfRangeError,
    validate_inputs,
)
from functions.make_folds import TEST_FOLD, assign_folds
from functions.predict import predict_spl

#: A typical valid input row (inside every training range).
VALID = {"frequency_hz": 1000.0, "attack_angle_deg": 5.0, "chord_length_m": 0.1524,
         "velocity_mps": 55.5, "disp_thickness_m": 0.0028}


def _rows(n: int = 3, **changes) -> pd.DataFrame:
    """``n`` valid rows, with ``changes`` applied to row 1 only."""
    table = pd.DataFrame([VALID] * n)
    for column, value in changes.items():
        table[column] = table[column].astype(object)
        table.loc[1, column] = value
    return table


def test_input_columns_match_the_model_features():
    """The checked columns are exactly the five raw model inputs."""
    assert INPUT_COLUMNS == FEATURES


def test_ranges_match_the_training_data(data):
    """The allowed ranges are the min and max of the rows the model was trained on."""
    folded = assign_folds(data)
    train = folded[folded["fold"] != TEST_FOLD]
    for column, (low, high) in FEATURE_RANGES.items():
        assert low == pytest.approx(train[column].min(), rel=1e-9)
        assert high == pytest.approx(train[column].max(), rel=1e-9)


def test_valid_inputs_pass_unchanged():
    """Valid rows come back as floats with the same values and row labels."""
    table = _rows()
    clean = validate_inputs(table)
    np.testing.assert_allclose(clean.to_numpy(), table[INPUT_COLUMNS].to_numpy(dtype=float))
    assert list(clean.index) == list(table.index)


def test_exact_range_limits_are_allowed():
    """The smallest and largest trained values themselves are accepted, as is 0 deg."""
    low = {c: lo for c, (lo, hi) in FEATURE_RANGES.items()}
    high = {c: hi for c, (lo, hi) in FEATURE_RANGES.items()}
    validate_inputs(pd.DataFrame([low, high]))
    assert low["attack_angle_deg"] == 0.0


def test_missing_column_names_the_column():
    """An absent column stops with an error naming it."""
    table = _rows().drop(columns="velocity_mps")
    with pytest.raises(MissingInputError, match="velocity_mps"):
        validate_inputs(table)


@pytest.mark.parametrize("empty", [np.nan, None])
def test_missing_value_names_column_and_row_without_imputing(empty):
    """A NaN or empty value stops with an error naming the column and the row."""
    table = _rows(chord_length_m=empty)
    before = table.copy()
    with pytest.raises(MissingInputError, match=r"chord_length_m.*row\(s\) \[1\]"):
        validate_inputs(table)
    pd.testing.assert_frame_equal(table, before)  # nothing filled in


def test_several_missing_values_are_all_reported():
    """Every column with missing values is named, with its rows."""
    table = _rows(5)
    table.loc[[0, 3], "frequency_hz"] = np.nan
    table.loc[4, "disp_thickness_m"] = np.nan
    with pytest.raises(MissingInputError) as error:
        validate_inputs(table)
    message = str(error.value)
    assert "frequency_hz" in message and "[0, 3]" in message
    assert "disp_thickness_m" in message and "[4]" in message


def test_non_number_is_invalid():
    """Text where a number is expected stops with an error naming column and row."""
    with pytest.raises(InvalidInputError, match=r"velocity_mps.*\[1\]"):
        validate_inputs(_rows(velocity_mps="fast"))


@pytest.mark.parametrize("column", ["frequency_hz", "chord_length_m", "velocity_mps", "disp_thickness_m"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_zero_or_negative_physical_quantities_are_invalid(column, value):
    """Frequency, chord, velocity and thickness must be > 0 (they feed log f, St and Re)."""
    with pytest.raises(InvalidInputError, match=column):
        validate_inputs(_rows(**{column: value}))


def test_negative_angle_is_invalid_and_explained():
    """A negative angle is refused, explaining the missing pressure-side data."""
    with pytest.raises(InvalidInputError, match="pressure"):
        validate_inputs(_rows(attack_angle_deg=-2.0))


def test_invalid_is_reported_before_out_of_range():
    """A negative velocity is 'invalid', not merely 'out of range'."""
    with pytest.raises(InvalidInputError):
        validate_inputs(_rows(velocity_mps=-50.0))


@pytest.mark.parametrize("column, value", [
    ("frequency_hz", 100.0), ("frequency_hz", 25000.0),
    ("attack_angle_deg", 25.0),
    ("chord_length_m", 0.01), ("chord_length_m", 0.5),
    ("velocity_mps", 20.0), ("velocity_mps", 80.0),
    ("disp_thickness_m", 0.0001), ("disp_thickness_m", 0.1),
])
def test_values_outside_training_range_are_refused(column, value):
    """Valid but untrained values below the minimum or above the maximum are refused."""
    low, high = FEATURE_RANGES[column]
    with pytest.raises(OutOfRangeError, match=rf"{column}.*{low:g}-{high:g}"):
        validate_inputs(_rows(**{column: value}))


def test_errors_are_value_errors():
    """All input errors can be caught as ValueError."""
    assert issubclass(InputError, ValueError)
    for error in (MissingInputError, InvalidInputError, OutOfRangeError):
        assert issubclass(error, InputError)


def test_loader_refuses_a_file_with_missing_values(tmp_path):
    """A dataset file with an empty cell is refused when loaded, not imputed."""
    raw = pd.read_excel(load_airfoil_data.__defaults__[0]).head(5)
    raw.loc[2, "frequency (Hz)"] = np.nan  # the source file's frequency column
    path = tmp_path / "damaged.xlsx"
    raw.to_excel(path, index=False)
    with pytest.raises(MissingInputError, match=r"frequency_hz.*\[2\]"):
        load_airfoil_data(path)


def test_predict_spl_on_valid_input():
    """The saved final model gives a plausible SPL for a valid input."""
    prediction = predict_spl(_rows(1))
    assert prediction.shape == (1,)
    assert 90 < prediction[0] < 150


def test_predict_spl_refuses_bad_input_before_predicting():
    """A bad row stops the call before the model is even used."""
    class MustNotPredict:
        def predict(self, X):
            raise AssertionError("predicted despite bad input")

    with pytest.raises(OutOfRangeError):
        predict_spl(_rows(velocity_mps=90.0), model=MustNotPredict())
