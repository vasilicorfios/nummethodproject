Usage
=====

Setup
-----

Place ``airfoil_self_noise.xlsx`` in the ``data/`` directory (it is not
committed to the repository), then install the dependencies::

    pip install pandas openpyxl matplotlib

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

Exploratory plots
-----------------

.. code-block:: console

    python src/explore_plots.py

Building these docs
-------------------

.. code-block:: console

    pip install sphinx
    sphinx-build -b html doc doc/_build/html
