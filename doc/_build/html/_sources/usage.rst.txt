Usage
=====

Setup
-----

Place ``airfoil_self_noise.xlsx`` in the ``data/`` directory (it is not
committed to the repository), then install the dependencies::

    pip install pandas openpyxl matplotlib scikit-learn pytest pytest-html

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

Training and cross-validating models
------------------------------------

Models (:mod:`functions.models`) and feature sets
(:mod:`functions.feature_sets`) are chosen by name, so any model can be run
on any feature set. Feature set ``A`` is the five raw inputs; ``B`` adds
log-frequency, Strouhal number and Reynolds number. Each model is a
scikit-learn pipeline that standardises the inputs and, through
``TransformedTargetRegressor``, the target. Both scalers are fitted on the
training folds only.

.. code-block:: console

    python src/run_cv.py                                   # linear on A and B
    python src/run_cv.py --models linear --feature-sets B

The script reports out-of-fold RMSE (the main metric), MAE and R², averaged
over the three CV seeds, along with the mean fit and predict time per fold.
It also breaks the out-of-fold RMSE down by chord, velocity, angle band
(0, 0-5, 5-10, 10-15, >15 deg) and frequency band. Results are saved as CSV
files in ``results/``. The test hold-out is not used.

From Python:

.. code-block:: python

    from functions.evaluate import cross_validate, summarise, oof_rmse_breakdown

    data = add_physics_features(load_airfoil_data())
    result = cross_validate(data, "linear", "B")
    print(summarise(result))
    print(oof_rmse_breakdown(result))
    print(coefficient_summary(result))   # linear models only

For linear models the script also prints each input's coefficient in dB per
standard deviation of that input, with its spread over all 15 fits (5 folds
x 3 seeds). A coefficient whose sign changes between fits (``sign_agreement``
below 1) is not reliably estimated. In feature set B this happens to
``log_frequency`` and ``reynolds``, which are strongly correlated with other
inputs, so their individual coefficients should not be interpreted
physically.

Profiling runtime, memory and FLOPs
-----------------------------------

.. code-block:: console

    python src/run_profile.py

This trains each model on all cross-validation rows (not the test set) and
reports:

* the median fit and predict time over 20 repeats;
* peak memory during a fit (traced with ``tracemalloc``);
* an estimated FLOP count and the achieved FLOP rate;
* a ``cProfile`` report of where the fit time goes.

For the linear baseline, the fit needs only about 10\ :sup:`5` FLOPs but
takes about 3 ms. That is tens of MFLOP/s, a tiny fraction of the CPU's
capability. The profile shows the time goes to scikit-learn and pandas input
checks and scaling rather than the least-squares solve, so at this data size
the cost is fixed overhead, not arithmetic.

Exploratory plots
-----------------

.. code-block:: console

    python src/explore_plots.py

Running the tests
-----------------

The tests in ``test/`` use pytest (configured in ``pytest.ini``, which puts
``src/`` on the import path). Run them from the project root and write the
HTML test report to ``test/test_report.html``:

.. code-block:: console

    pip install pytest pytest-html
    python -m pytest --html=test/test_report.html --self-contained-html

The tests check:

* **Hand-calculated values:** physics features, metrics, band edges, RMSE
  breakdown, coefficient summary and FLOP counts against results worked out
  by hand.
* **A manufactured solution:** noise-free data from a known linear formula,
  with inputs about 10\ :sup:`8` apart in scale. The linear pipeline must
  reproduce it exactly and recover the known coefficients, which verifies
  the scaling and unscaling steps.
* **No data leakage:** setups are never split across folds, the test set is
  the same for every CV seed and is never trained on or predicted, every CV
  row is predicted exactly once per seed, and scalers are fitted on training
  rows only.
* **A regression check:** the linear baseline's recorded RMSE on set A
  (4.897 dB), so an unintended change to the pipeline is caught.
* **Input handling:** the data are within their physical ranges with no
  missing values; unknown model or feature-set names raise ``KeyError``, and
  a missing data file raises ``FileNotFoundError``.

Building these docs
-------------------

.. code-block:: console

    pip install sphinx
    sphinx-build -b html doc doc/_build/html
