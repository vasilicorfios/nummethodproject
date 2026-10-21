Results
=======

This page reports the cross-validation results for every model run so far.
The numbers are not typed in by hand: they are read from the tables in
``doc/generated/``, which ``src/run_cv.py`` and ``src/run_profile.py``
write. To update this page after changing the code:

.. code-block:: console

    python src/run_cv.py
    python src/run_profile.py
    sphinx-build -b html doc doc/_build/html

How the results were obtained
-----------------------------

* **Data used:** only the cross-validation setups, about 85% of the 106
  wind-tunnel setups. The test hold-out (seed 41) has not been used to
  train, tune or score any model.
* **Method:** grouped 5-fold cross-validation, stratified on chord length,
  so every row of a setup is either all in training or all in validation.
  It is repeated with seeds 42, 43 and 44.
* **Predictions:** every cross-validation row is predicted once per seed by a
  model that did not see it during training (an out-of-fold, or OOF,
  prediction).
* **Metrics:** RMSE (the main metric), MAE and R² are computed on each
  seed's pooled OOF predictions. Each table entry is the mean ± standard
  deviation over the three seeds.
* **Timings:** ``time.perf_counter`` around each fold's fit and predict,
  averaged over all 15 folds.

Models so far
~~~~~~~~~~~~~

* **linear:** ordinary least-squares linear regression (the baseline).
  Inputs and target are standardised with scalers fitted on the training
  folds only.

Feature sets: **A** = the five raw inputs; **B** = A + log10 frequency,
Strouhal number and chord Reynolds number.

Overall accuracy
----------------

.. csv-table:: Out-of-fold metrics (mean ± std over CV seeds)
   :file: generated/cv_summary.csv
   :header-rows: 1

Adding the physics features (set B) lowers the linear baseline's RMSE and
raises R² compared with set A. Both feature sets still leave around half of
the SPL variance unexplained, which suggests the relationship is strongly
non-linear and motivates the non-linear models planned next.

Where the errors are
--------------------

RMSE (dB) of the OOF predictions for each subgroup, averaged over seeds.
*Rows* is the number of cross-validation measurements in the group.

.. csv-table:: Out-of-fold RMSE by chord, velocity, angle band and frequency band (dB)
   :file: generated/cv_rmse_breakdown.csv
   :header-rows: 1

For the linear baseline, the largest errors are at the smallest chord, the
lowest frequency band and the highest angle band (above 15°, where the
boundary layer is likely to be separated or near stall). Free-stream velocity
has a much smaller effect on error. These are the regions where a straight
line in the inputs fits the measured spectra worst.

What the linear model learned
-----------------------------

Each cell shows an input's coefficient in **dB per standard deviation of
that input**, as mean ± std over all 15 fits (5 folds × 3 seeds). The value
in brackets is the percentage of fits whose sign agrees with the mean.
Because every input is expressed per standard deviation, effects can be
compared directly even though the inputs have very different units.

.. csv-table:: Linear coefficients (dB per input standard deviation) [sign agreement]
   :file: generated/cv_coefficients.csv
   :header-rows: 1

On set A every coefficient keeps the same sign in every fit, so the
direction of each effect is reliable. Higher frequency, longer chord, larger
angle of attack and thicker boundary layer lower the SPL, and higher velocity
raises it.

On set B, some coefficients (``log_frequency``, ``reynolds``) change sign
between fits and have a spread larger than their mean. These inputs are
strongly correlated with others (log frequency with frequency; Reynolds
number with chord and velocity), so the model cannot separate their
individual effects. Their coefficients should not be given a physical
meaning, even though the predictions themselves are stable.

Runtime, memory and FLOPs
-------------------------

Each model is fitted once on all cross-validation rows. Times are the median
of 20 repeats, and peak memory is traced with ``tracemalloc`` (a lower
bound, since it cannot see memory allocated inside compiled LAPACK code). The
FLOP count comes from :func:`functions.profiling.linear_flop_estimate`, and
the fit rate is the estimated FLOPs divided by the measured fit time.

.. csv-table:: Profile of one fit on all CV rows
   :file: generated/profile_summary.csv
   :header-rows: 1

The linear fit needs only about 10\ :sup:`5` floating-point operations, yet
it runs at tens of MFLOP/s, a tiny fraction of the CPU's capability. The
``cProfile`` reports (``results/profile_<model>_<set>.txt``) show the time
goes to scikit-learn and pandas input validation and scaling, not the
least-squares solve. At this data size the cost is fixed overhead, not
arithmetic.

What these results cannot show
------------------------------

* **Performance on unseen data:** these are cross-validation scores used
  for model comparison. The final estimate will come from the test hold-out,
  scored once on the chosen models.
* **Extrapolation:** every chord, velocity and angle seen in validation was
  also tested in training (with other setups). The results say nothing about
  geometries or flow conditions outside the tested ranges, or about other
  airfoil shapes.
* **Measurement noise:** the irreducible scatter in the wind-tunnel data is
  not separated from model error, so the lowest achievable RMSE is unknown.
