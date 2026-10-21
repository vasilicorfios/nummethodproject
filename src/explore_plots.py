"""Entry point that displays the exploratory plots for the airfoil dataset.

The plotting functions live in their own modules
(:mod:`functions.plot_spl_vs_features`, :mod:`functions.plot_spl_by_chord_length`
and :mod:`functions.plot_spl_vs_physics_features`); this script just loads the
data, adds the physics features and shows all three figures::

    python src/explore_plots.py
"""

import matplotlib.pyplot as plt

from functions.data_loader import load_airfoil_data
from functions.physics_features import add_physics_features
from functions.plot_spl_by_chord_length import plot_spl_by_chord_length
from functions.plot_spl_vs_features import plot_spl_vs_features
from functions.plot_spl_vs_physics_features import plot_spl_vs_physics_features


def main() -> None:
    """Load the dataset and display all exploratory figures."""
    df = add_physics_features(load_airfoil_data())
    plot_spl_vs_features(df)
    plot_spl_by_chord_length(df)
    plot_spl_vs_physics_features(df)
    plt.show()


if __name__ == "__main__":
    main()
