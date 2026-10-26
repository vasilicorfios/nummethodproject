"""Hand-calculated checks shown in the report (Section 4.1).

Each test reproduces a calculation done by hand in the report and checks
that the code gives the same answer. The tests also print the hand value
next to the code's value, so the comparison can be seen directly::

    python -m pytest test/test_hand_calculations.py -v -s

(``-s`` shows the printed tables; without it the tests still run and pass or
fail as normal.)
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from functions.data_loader import load_airfoil_data
from functions.physics_features import add_physics_features

# --- Section 4.1.2: hand-chosen network (report table) ---------------------

#: Weights into the hidden layer. Rows = inputs (x1, x2), columns = hidden
#: neurons (1, 2), as scikit-learn stores them.
HIDDEN_WEIGHTS = np.array([[1.0, -1.0],
                           [2.0, 2.0]])

#: Biases of hidden neurons 1 and 2.
HIDDEN_BIASES = np.array([3.0, -6.0])

#: Weights from hidden neurons 1 and 2 to the output, and the output bias.
OUTPUT_WEIGHTS = np.array([[1.0], [2.0]])
OUTPUT_BIAS = np.array([1.0])

#: Inputs used in the hand calculation.
X = np.array([[1.0, 2.0]])

#: Hand-calculated values from the report.
HAND_Z = (8.0, -3.0)      # z1, z2 before ReLU
HAND_H = (8.0, 0.0)       # h1, h2 after ReLU
HAND_OUTPUT = 9.0         # y-hat

# --- Section 4.1.3: row 471 of the dataset ----------------------------------

#: Row checked by hand: f = 1000 Hz, angle 0, c = 0.1524 m, U = 55.5 m/s,
#: delta* = 0.00172668 m, SPL = 128.583 dB.
ROW = 471

#: Hand-calculated physics features, to the precision written in the report.
HAND_LOG_F = 3.0
HAND_STROUHAL = 0.031111
HAND_REYNOLDS = 563_880.0


def _network(output_weights: np.ndarray, output_bias: np.ndarray) -> MLPRegressor:
    """A 2-input, 2-hidden-neuron ReLU MLP with the report's hidden layer.

    scikit-learn only creates the weight arrays when a model is fitted, so
    it is fitted for one step on dummy data and the weights are then
    replaced by the hand-chosen values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = MLPRegressor(hidden_layer_sizes=(2,), activation="relu", max_iter=1)
        mlp.fit(np.zeros((3, 2)), np.zeros(3))
    mlp.coefs_ = [HIDDEN_WEIGHTS, output_weights]
    mlp.intercepts_ = [HIDDEN_BIASES, output_bias]
    return mlp


def _show(title: str, rows: list[tuple[str, float, float]]) -> None:
    """Print a small 'hand vs code' table (visible with pytest -s)."""
    print(f"\n{title}")
    print(f"  {'quantity':<28}{'hand':>14}{'code':>14}")
    for name, hand, code in rows:
        print(f"  {name:<28}{hand:>14.6g}{code:>14.6g}")


def test_mlp_forward_pass_matches_hand_calculation():
    """The network with the report's weights gives y-hat = 9 for x = (1, 2).

    Hand calculation (report, Section 4.1.2):
        z1 = (1)(1) + (2)(2) + 3  = 8   -> h1 = max(0, 8)  = 8
        z2 = (-1)(1) + (2)(2) - 6 = -3  -> h2 = max(0, -3) = 0
        y  = (1)(8) + (2)(0) + 1  = 9
    """
    # z before the ReLU, from the same weights (plain matrix arithmetic).
    z = (X @ HIDDEN_WEIGHTS + HIDDEN_BIASES)[0]

    # Hidden activations measured through scikit-learn itself: setting the
    # output weights to (1, 0) or (0, 1) with zero bias makes the network
    # output h1 or h2 directly, so ReLU is checked inside the real model.
    h1 = _network(np.array([[1.0], [0.0]]), np.array([0.0])).predict(X)[0]
    h2 = _network(np.array([[0.0], [1.0]]), np.array([0.0])).predict(X)[0]
    output = _network(OUTPUT_WEIGHTS, OUTPUT_BIAS).predict(X)[0]

    _show("MLP forward pass, x1 = 1, x2 = 2 (report 4.1.2)", [
        ("z1 (before ReLU)", HAND_Z[0], z[0]),
        ("h1 = max(0, z1)", HAND_H[0], h1),
        ("z2 (before ReLU)", HAND_Z[1], z[1]),
        ("h2 = max(0, z2)", HAND_H[1], h2),
        ("output y-hat", HAND_OUTPUT, output),
    ])

    np.testing.assert_allclose(z, HAND_Z)
    np.testing.assert_allclose([h1, h2], HAND_H)
    assert np.isclose(output, HAND_OUTPUT)


def test_physics_features_match_hand_calculation():
    """Row 471's log f, Strouhal and Reynolds numbers match the hand values.

    Hand calculation (report, Section 4.1.3):
        log10(1000)                    = 3
        St = 0.00172668 * 1000 / 55.5  = 0.031111
        Re = 55.5 * 0.1524 / 1.5e-5    = 563,880
    """
    row = add_physics_features(load_airfoil_data()).loc[ROW]

    # The row really is the one in the report's table.
    assert (row["frequency_hz"], row["attack_angle_deg"], row["chord_length_m"],
            row["velocity_mps"], row["disp_thickness_m"], row["spl_db"]) == (
        1000, 0.0, 0.1524, 55.5, 0.00172668, 128.583)

    _show(f"Physics-informed features, row {ROW} (report 4.1.3)", [
        ("log10(f)", HAND_LOG_F, row["log_frequency"]),
        ("Strouhal  f*delta*/U", HAND_STROUHAL, row["strouhal"]),
        ("Reynolds  U*c/nu", HAND_REYNOLDS, row["reynolds"]),
    ])

    assert np.isclose(row["log_frequency"], HAND_LOG_F)
    # The hand value was written to 6 decimal places, so allow half a unit there.
    assert abs(row["strouhal"] - HAND_STROUHAL) <= 5e-7
    assert np.isclose(row["reynolds"], HAND_REYNOLDS, rtol=1e-9)
