"""Checks that refuse missing, invalid or out-of-range inputs before predicting.

A prediction is only trustworthy for inputs like the ones the model was
trained on. Instead of guessing, the checks below **stop with an error** that
names the offending column(s), row(s) and value(s). Nothing is ever filled in
(imputed) or silently clipped: a wrong guess would give a confident-looking
but meaningless noise prediction.

The checks run in this order (:func:`validate_inputs`):

1. **Missing** (:class:`MissingInputError`): a required column is absent, or
   a value is empty/NaN.
2. **Invalid** (:class:`InvalidInputError`): a value is not a number, or is
   physically impossible:

   * frequency, chord, velocity and displacement thickness must be
     **greater than 0**. Zero or negative values are impossible, and they would
     also break the physics-derived features: ``log10(f)`` needs ``f > 0``,
     the Strouhal number ``f δ*/U`` divides by the velocity, and the Reynolds
     number ``U c / ν`` needs a positive velocity and chord;
   * the angle of attack must be **0 or more**. The NACA 0012 is symmetric,
     so a negative angle is in principle the same flow mirrored. It could be
     treated as merely out of range, but the dataset gives only the
     *suction-side* displacement thickness, so a negative angle (which swaps
     the suction and pressure sides) cannot be described correctly by these
     inputs. It is therefore treated as invalid.

3. **Out of range** (:class:`OutOfRangeError`): a valid value outside the
   range the final model was trained on (:data:`FEATURE_RANGES`, both the
   minimum and the maximum). Predicting there would be extrapolating to a new
   setup whose noise could follow different physics, for example a smaller
   chord with stronger tonal noise, or a higher speed. The shift tests showed
   the error grows sharply outside the training range.

**What these checks do not catch.** Each input is checked on its own. A
combination of individually valid values that was never tested together (for
example the largest chord at the highest angle) passes, even though it is
also an extrapolation. See the "Valid inputs" section of the usage docs.

Example:
    >>> import pandas as pd
    >>> from functions.input_checks import validate_inputs
    >>> validate_inputs(pd.DataFrame({"frequency_hz": [1000], "attack_angle_deg": [-2.0],
    ...     "chord_length_m": [0.1], "velocity_mps": [50.0],
    ...     "disp_thickness_m": [0.002]}))   # doctest: +SKIP
    InvalidInputError: Invalid input: 'attack_angle_deg' must be >= 0 ...
"""

# Postponed annotations: the docs build mocks pandas.
from __future__ import annotations

import pandas as pd

#: The five raw input columns a prediction needs (the same as
#: :data:`functions.data_loader.FEATURES`; a test keeps them in sync).
INPUT_COLUMNS = [
    "frequency_hz",
    "attack_angle_deg",
    "chord_length_m",
    "velocity_mps",
    "disp_thickness_m",
]

#: Inputs that must be strictly greater than zero.
POSITIVE_COLUMNS = ["frequency_hz", "chord_length_m", "velocity_mps", "disp_thickness_m"]

#: Inputs that may be zero but not negative.
NON_NEGATIVE_COLUMNS = ["attack_angle_deg"]

#: (minimum, maximum) of each input over the rows the final model was trained
#: on (all cross-validation rows). These equal the full dataset's ranges; a
#: test checks they match the data exactly.
FEATURE_RANGES = {
    "frequency_hz": (200.0, 20000.0),
    "attack_angle_deg": (0.0, 22.2),
    "chord_length_m": (0.0254, 0.3048),
    "velocity_mps": (31.7, 71.3),
    "disp_thickness_m": (0.000400682, 0.0584113),
}

#: Units, for readable error messages.
UNITS = {
    "frequency_hz": "Hz",
    "attack_angle_deg": "deg",
    "chord_length_m": "m",
    "velocity_mps": "m/s",
    "disp_thickness_m": "m",
}

#: At most this many offending rows are listed per column in a message.
MAX_ROWS_LISTED = 10


class InputError(ValueError):
    """Base class for every input problem (a ``ValueError``)."""


class MissingInputError(InputError):
    """A required column is absent, or a value is missing (empty or NaN)."""


class InvalidInputError(InputError):
    """A value is not a number, or is physically impossible."""


class OutOfRangeError(InputError):
    """A valid value lies outside the range the model was trained on."""


def _rows(mask: pd.Series) -> str:
    """List the row labels where ``mask`` is True (shortened if very long)."""
    rows = list(mask.index[mask])
    shown = ", ".join(str(r) for r in rows[:MAX_ROWS_LISTED])
    extra = len(rows) - MAX_ROWS_LISTED
    return f"row(s) [{shown}{f', ... and {extra} more' if extra > 0 else ''}]"


def check_missing(inputs: pd.DataFrame, columns: list[str] = INPUT_COLUMNS) -> None:
    """Stop if a required column is absent or any value is empty/NaN.

    Args:
        inputs: Table of inputs, one row per prediction.
        columns: Required columns.

    Raises:
        MissingInputError: Naming the absent columns, or every column with
            missing values and the rows where they are missing.
    """
    absent = [c for c in columns if c not in inputs.columns]
    if absent:
        raise MissingInputError(f"Missing input column(s): {absent}. Required: {columns}.")
    problems = [f"'{c}' at {_rows(inputs[c].isna())}" for c in columns if inputs[c].isna().any()]
    if problems:
        raise MissingInputError(
            "Missing value(s) (empty or NaN), which are not filled in: " + "; ".join(problems) + ".")


def check_numeric(inputs: pd.DataFrame, columns: list[str] = INPUT_COLUMNS) -> pd.DataFrame:
    """Stop if a value is not a number; otherwise return the inputs as floats.

    Args:
        inputs: Table of inputs with no missing values.
        columns: Columns to check.

    Returns:
        A copy of ``inputs[columns]`` with every value as a float.

    Raises:
        InvalidInputError: Naming each column and the rows with non-numbers.
    """
    converted = inputs[columns].apply(pd.to_numeric, errors="coerce")
    # A value that was present but became NaN on conversion was not a number.
    bad = converted.isna() & inputs[columns].notna()
    problems = [f"'{c}' at {_rows(bad[c])}" for c in columns if bad[c].any()]
    if problems:
        raise InvalidInputError("Invalid input, not a number: " + "; ".join(problems) + ".")
    return converted.astype(float)


def check_valid(inputs: pd.DataFrame) -> None:
    """Stop if any value is physically impossible.

    Frequency, chord, velocity and displacement thickness must be > 0;
    the angle of attack must be >= 0 (see the module docstring for why
    negative angles are treated as invalid).

    Args:
        inputs: Numeric inputs with no missing values.

    Raises:
        InvalidInputError: Naming each column, the rows and the bad values.
    """
    problems = []
    for column in POSITIVE_COLUMNS:
        bad = inputs[column] <= 0
        if bad.any():
            problems.append(f"'{column}' must be > 0 (got {inputs.loc[bad, column].tolist()[:MAX_ROWS_LISTED]} "
                            f"at {_rows(bad)}); zero or negative values would also break "
                            "log10(f), the Strouhal number or the Reynolds number")
    for column in NON_NEGATIVE_COLUMNS:
        bad = inputs[column] < 0
        if bad.any():
            problems.append(f"'{column}' must be >= 0 (got {inputs.loc[bad, column].tolist()[:MAX_ROWS_LISTED]} "
                            f"at {_rows(bad)}); negative angles swap the suction and pressure sides, "
                            "and no pressure-side displacement thickness is available")
    if problems:
        raise InvalidInputError("Invalid input: " + "; ".join(problems) + ".")


def check_in_range(inputs: pd.DataFrame, ranges: dict[str, tuple[float, float]] = FEATURE_RANGES) -> None:
    """Stop if any value is outside the range the model was trained on.

    Args:
        inputs: Valid numeric inputs.
        ranges: (minimum, maximum) per column; both ends are allowed.

    Raises:
        OutOfRangeError: Naming each column, the rows, the values and the
            allowed range.
    """
    problems = []
    for column, (low, high) in ranges.items():
        bad = (inputs[column] < low) | (inputs[column] > high)
        if bad.any():
            unit = UNITS.get(column, "")
            problems.append(f"'{column}' must be within {low:g}-{high:g} {unit} "
                            f"(got {inputs.loc[bad, column].tolist()[:MAX_ROWS_LISTED]} at {_rows(bad)})")
    if problems:
        raise OutOfRangeError(
            "Input outside the range the model was trained on (refused: predicting there "
            "would extrapolate to setups whose noise may follow different physics): "
            + "; ".join(problems) + ".")


def validate_inputs(inputs: pd.DataFrame, check_range: bool = True) -> pd.DataFrame:
    """Run every check and return clean inputs, or stop with an error.

    Args:
        inputs: Table with the columns in :data:`INPUT_COLUMNS` (extra
            columns are ignored), one row per prediction.
        check_range: Also refuse values outside :data:`FEATURE_RANGES`. Set
            to False only when loading the training data itself.

    Returns:
        A new table of the input columns as floats, with the original row
        labels. The input table is not modified.

    Raises:
        MissingInputError: A column or value is missing.
        InvalidInputError: A value is not a number or is physically impossible.
        OutOfRangeError: A value is outside the trained range.
    """
    check_missing(inputs)
    clean = check_numeric(inputs)
    check_valid(clean)
    if check_range:
        check_in_range(clean)
    return clean
