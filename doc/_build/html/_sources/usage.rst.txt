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

The script reports validation RMSE (the main metric), MAE and R² as mean ±
standard deviation over all 15 folds (5 folds x 3 seeds), so the spread
shows how much the score changes from one batch of unseen setups to the
next. It also reports the mean fit and predict time per fold.
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

Tuning the MLP with nested cross-validation
-------------------------------------------

Choosing hyperparameters and scoring the model on the same folds would make
the score optimistic. **Nested cross-validation** keeps the two jobs apart.
For each outer fold, an inner grouped 5-fold cross-validation, run only on
that fold's training setups, picks the network size and ``alpha``. The chosen
model is then scored on the outer fold, which played no part in choosing it.

.. code-block:: console

    python src/run_tuning.py                               # all 3 seeds, sets A and B
    python src/run_tuning.py --seeds 42 --feature-sets A   # quick one-seed check

* ``--seeds``: outer CV seeds to run, any of 42, 43 and 44 (default: all).
* ``--feature-sets``: A and/or B (default: both).

The grid has 35 settings: hidden layers (32,), (64,), (32, 32), (64, 64) and
(128, 128), each with alpha 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1 and 10. A full
run is 2,625 inner fits per feature set, plus one refit per outer fold, and
takes about 44 minutes on an 8-core laptop for sets A and B together
(measured: 23.8 min for A, 19.8 min for B). One seed on one feature set
takes about 7–8 minutes. The script also re-runs the
linear baseline on the same seeds and folds, so the two models can be
compared fold by fold. Results go to ``results/tuning_*.csv``, and tables and
figures (PNG and PDF) go to ``doc/generated/``. The test hold-out is never
used.

Finding the worst-predicted setups
----------------------------------

.. code-block:: console

    python src/run_error_analysis.py                          # sets A and B, 10 worst
    python src/run_error_analysis.py --feature-sets B --n-worst 5

This ranks every cross-validation setup by its out-of-fold RMSE for the tuned
MLP, puts the linear model's RMSE alongside, and plots the worst spectra. It
needs ``results/tuning_outer_scores.csv`` from ``run_tuning.py``. Rather than
re-running the tuning, it uses the saved out-of-fold predictions (see below),
or refits each fold's chosen setting, which reproduces the tuned MLP's
predictions exactly because the model has a fixed ``random_state``. Either
way it checks every fold's RMSE against the saved value and
stops with an error if they differ. It takes about a minute.

The tuned MLP's out-of-fold predictions are saved by ``run_tuning.py`` in
``results/tuning_oof_predictions.csv``, so this script normally just loads
them. Only when the file is missing (results from before it was added) does
it refit each fold's chosen setting, and then it saves the file.

The same script also produces:

* an **error map** placing every setup on the test matrix, coloured by RMSE;
* a **Strouhal-scaling check**: whether measured peak frequencies scale
  with ``U / δ*`` (constant Strouhal number), and whether the MLP's predicted
  peak lands within one 1/3-octave step of the measured one;
* a **residual heatmap** (setups x frequencies), and the share of each setup's
  squared error that falls around its measured peak.

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
    python src/explore_plots.py --no-show   # only write the PNGs

This saves the SPL histogram and the SPL-vs-feature scatter plots for
feature sets A and B to ``doc/generated/`` (shown on :doc:`exploration`),
then opens every exploratory figure, including the per-chord-length grid.

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
  (mean per-fold RMSE 4.894 dB), so an unintended change to the pipeline is caught.
* **Input handling:** the data are within their physical ranges with no
  missing values; unknown model or feature-set names raise ``KeyError``, and
  a missing data file raises ``FileNotFoundError``.

Building these docs
-------------------

.. code-block:: console

    pip install sphinx
    sphinx-build -b html doc doc/_build/html
