"""Entry point that displays the exploratory plots for the airfoil dataset.

The plotting functions live in their own modules
(:mod:`functions.plot_spl_vs_features` and :mod:`functions.plot_spl_by_chord_length`); this
script just loads the data and shows both figures::

    python src/explore_plots.py
"""

import matplotlib.pyplot as plt

from functions.data_loader import load_airfoil_data
from functions.plot_spl_by_chord_length import plot_spl_by_chord_length
from functions.plot_spl_vs_features import plot_spl_vs_features


def main() -> None:
    """Load the dataset and display both exploratory figures."""
    df = load_airfoil_data()
    plot_spl_vs_features(df)
    plot_spl_by_chord_length(df)
    plt.show()


if __name__ == "__main__":
    main()
