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

Exploratory plots
-----------------

.. code-block:: console

    python src/explore_plots.py

Building these docs
-------------------

.. code-block:: console

    pip install sphinx
    sphinx-build -b html doc doc/_build/html
