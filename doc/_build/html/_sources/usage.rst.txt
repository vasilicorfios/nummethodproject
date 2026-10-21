Usage
=====

Setup
-----

Place ``airfoil_self_noise.xlsx`` in the ``data/`` directory (it is not
committed to the repository), then install the dependencies::

    pip install pandas openpyxl matplotlib scikit-learn

Loading the data
----------------

.. code-block:: python

    from functions.data_loader import load_airfoil_data

    df = load_airfoil_data()
    print(df.head())

Adding physics features
-----------------------

Log-frequency, Strouhal number and Reynolds number are computed from the raw
columns on load rather than stored in the Excel file:

.. code-block:: python

    from functions.data_loader import load_airfoil_data
    from functions.physics_features import add_physics_features

    df = add_physics_features(load_airfoil_data())

Splitting the data
------------------

About 15% of the 106 setups are held out as a test set, and the rest are
split into 5 cross-validation folds. Both steps keep each setup's rows
together and are stratified on chord length. Cross-validation is repeated
with seeds 42, 43 and 44; the test set (seed 41) is the same in every
repeat:

.. code-block:: python

    from functions.make_folds import assign_folds, CV_SEEDS, TEST_FOLD

    data = add_physics_features(load_airfoil_data())
    for seed in CV_SEEDS:
        df = assign_folds(data, cv_seed=seed)
        test = df[df["fold"] == TEST_FOLD]
        cv = df[df["fold"] != TEST_FOLD]

        for k in range(5):
            train, val = cv[cv["fold"] != k], cv[cv["fold"] == k]

Exploratory plots
-----------------

.. code-block:: console

    python src/explore_plots.py

Building these docs
-------------------

.. code-block:: console

    pip install sphinx
    sphinx-build -b html doc doc/_build/html
