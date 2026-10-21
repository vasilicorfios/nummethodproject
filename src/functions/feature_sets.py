"""Named input feature sets, so models can be compared on different inputs.

Each feature set is a list of column names. Any model in
:mod:`functions.models` can be trained on any feature set, so swapping inputs
is a matter of changing one name.

* ``A``: the five raw wind-tunnel inputs.
* ``B``: ``A`` plus the physics-derived features ``log_frequency``,
  ``strouhal`` and ``reynolds``. These columns only exist after
  :func:`functions.physics_features.add_physics_features` has been applied.

Example:
    >>> from functions.feature_sets import get_feature_columns
    >>> get_feature_columns("B")[-3:]
    ['log_frequency', 'strouhal', 'reynolds']
"""

from functions.data_loader import FEATURES
from functions.physics_features import PHYSICS_FEATURES

#: Feature set name -> input columns. Add a new entry here to make another
#: set available to every model and to ``src/run_cv.py``.
FEATURE_SETS = {
    "A": list(FEATURES),
    "B": list(FEATURES) + list(PHYSICS_FEATURES),
}


def get_feature_columns(name: str) -> list[str]:
    """Return the input columns for a named feature set.

    Args:
        name: Key of :data:`FEATURE_SETS`, e.g. ``"A"``.

    Returns:
        A new list of column names (safe for the caller to modify).

    Raises:
        KeyError: If ``name`` is not a known feature set.
    """
    if name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])
