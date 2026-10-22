Results
=======

This page reports the cross-validation results for every model run so far.
The numbers are not typed in by hand: they are read from the tables and
figures in ``doc/generated/``, which ``src/run_cv.py`` and
``src/run_profile.py`` write. To update this page after changing the code:

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
  validation fold separately. Each summary entry is the mean ± sample
  standard deviation over all 15 folds (5 folds × 3 seeds). Every fold
  scores a model on a batch of setups it never saw, so the ± shows how much
  the score changes from one batch of unseen setups to the next.
* **Timings:** ``time.perf_counter`` around each fold's fit and predict,
  averaged over all 15 folds.

Models so far
~~~~~~~~~~~~~

* **linear:** ordinary least-squares linear regression (the baseline).
  Inputs and target are standardised with scalers fitted on the training
  folds only.
* **mlp (tuned):** a multi-layer perceptron whose network size and L2
  penalty ``alpha`` are chosen by nested cross-validation (see
  `Tuned MLP (nested cross-validation)`_ below). It uses the same scaling
  pipeline and the same outer folds as the linear baseline.

Feature sets: **A** = the five raw inputs; **B** = A + log10 frequency,
Strouhal number and chord Reynolds number.

Overall accuracy
----------------

.. csv-table:: Validation metrics (mean ± std over all 15 folds)
   :file: generated/cv_summary.csv
   :header-rows: 1

Adding the physics features (set B) lowers the linear baseline's RMSE and
raises R² compared with set A. Both feature sets still leave around half of
the SPL variance unexplained, which suggests the relationship is strongly
non-linear and motivates the non-linear models planned next.

.. figure:: generated/cv_predicted_vs_measured.png
   :width: 100%
   :alt: Scatter of out-of-fold predicted SPL against measured SPL for each
         feature set, with a dashed y = x line.

   Out-of-fold predicted vs measured SPL for one CV seed, so every
   cross-validation row appears once. Points on the dashed line are perfect
   predictions. The clouds are flatter than the line: the models
   over-predict quiet conditions and under-predict loud ones, pulling
   extremes towards the mean. The RMSE in each title is for this one seed,
   pooled over all its folds, so it differs slightly from the mean of
   per-fold scores in the table above.

Where the errors are
--------------------

RMSE (dB) of the OOF predictions for each subgroup, averaged over seeds.
*Rows* is the number of cross-validation measurements in the group.

.. csv-table:: Out-of-fold RMSE by chord, velocity, angle band and frequency band (dB)
   :file: generated/cv_rmse_breakdown.csv
   :header-rows: 1

.. figure:: generated/cv_rmse_breakdown.png
   :width: 100%
   :alt: Grouped bar charts of out-of-fold RMSE by chord, velocity, angle
         band and frequency band, one bar per model and feature set.

   The same breakdown as the table, drawn as bars.

For the linear baseline, the largest errors are at the smallest chord, the
lowest frequency band and the highest angle band (above 15°, where the
boundary layer is likely to be separated or near stall). Free-stream velocity
has a much smaller effect on error. These are the regions where a straight
line in the inputs fits the measured spectra worst.

.. figure:: generated/cv_residual_by_frequency.png
   :width: 100%
   :alt: Line chart of mean out-of-fold residual against frequency on a log
         axis, one line per model and feature set.

   Mean residual (measured − predicted) at each measured frequency, for one
   CV seed. An unbiased model would follow the zero line. A curve that
   rises sharply at the highest frequencies means the model systematically
   under-predicts there: SPL does not change linearly with frequency, so a
   linear model cannot follow the spectrum's shape.

.. figure:: generated/cv_example_spectra.png
   :width: 100%
   :alt: Three panels of SPL against frequency for the best, median and
         worst predicted setups, comparing measured spectra with predictions.

   Measured and predicted spectra for the best, median and worst predicted
   setups, ranked by the first model's per-setup RMSE. Each prediction comes
   from a model that never saw that setup. The linear models can only
   produce smooth, monotonic curves, so they miss the peaked spectra that
   appear at smaller chords and higher angles.

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

.. figure:: generated/cv_coefficients.png
   :width: 85%
   :alt: Dot and whisker plot of each feature's coefficient with its spread
         across folds, for each feature set.

   The same coefficients, with whiskers of ± one standard deviation over
   the 15 fits. A whisker that crosses zero marks a coefficient whose sign
   is not reliably estimated.

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

Tuned MLP (nested cross-validation)
-----------------------------------

How it was tuned
~~~~~~~~~~~~~~~~

For every seed and outer fold, an **inner** grouped 5-fold cross-validation
on the outer-training setups compared 35 settings: hidden layers (32,),
(64,), (32, 32), (64, 64) and (128, 128), each with alpha 1e-5, 1e-4, 1e-3,
1e-2, 0.1, 1 and 10. The setting with the lowest inner RMSE was refitted on
all outer-training rows and scored on the **outer** fold, which played no
part in the choice. Every split keeps a setup's rows together, and the test
hold-out was not used.

The grid was widened once. In a one-seed trial with the original 20
settings (largest network (64, 64), largest alpha 0.1), the largest network
won every fold and the largest alpha won four of five, so the best settings
sat on the edge of the grid. (128, 128) and alphas 1 and 10 were added
before the full run reported here.

Accuracy and comparison with linear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Tuned MLP outer-fold results and paired comparison with linear
   :file: generated/tuning_summary.csv
   :header-rows: 1

Two spreads are reported. **SD over folds** is how much the score changes
from one batch of unseen setups to the next. **SD across seeds** is how much
the average depends on the random shuffle of setups into folds. The second
is much smaller, so the results do not hinge on one particular split.

Both models are scored on exactly the same folds, so the fair comparison is
the RMSE difference within each fold (MLP minus linear; negative means the
MLP was better). The tuned MLP was better than linear in every outer fold on
both feature sets, by roughly 2.2–2.3 dB. That is far larger than the spread
of the differences. As with the linear model, feature set B gives a lower
MLP RMSE than set A.

.. figure:: generated/tuning_pred_vs_measured_A.png
   :width: 100%
   :alt: Predicted against measured SPL for the linear baseline and the tuned
         MLP on feature set A.

   Feature set A: out-of-fold predictions for one seed, linear (left) and
   tuned MLP (right). The MLP's points lie much closer to the y = x line,
   including the quiet and loud extremes that the linear model pulls
   towards the mean.

.. figure:: generated/tuning_pred_vs_measured_B.png
   :width: 100%
   :alt: Predicted against measured SPL for the linear baseline and the tuned
         MLP on feature set B.

   Feature set B: the same comparison.

Where the MLP still struggles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: generated/tuning_rmse_breakdown_A.png
   :width: 100%
   :alt: Grouped bars of out-of-fold RMSE by chord, velocity, angle band and
         frequency band for linear and tuned MLP, feature set A.

   Feature set A: RMSE by subgroup, linear and tuned MLP.

.. figure:: generated/tuning_rmse_breakdown_B.png
   :width: 100%
   :alt: Grouped bars of out-of-fold RMSE by chord, velocity, angle band and
         frequency band for linear and tuned MLP, feature set B.

   Feature set B: RMSE by subgroup, linear and tuned MLP.

The MLP lowers the error in every subgroup, but the pattern is the same as
for linear: the smallest chord, angles above 15° and the lowest frequency
band remain the hardest conditions to predict. One new pattern appears:
for the MLP, the highest free-stream velocity (71.3 m/s) has a noticeably
larger error than the other velocities, which was not the case for linear.

Which settings were chosen
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Settings chosen by the inner CV (settings that never won are omitted)
   :file: generated/tuning_selected_params.csv
   :header-rows: 1

.. figure:: generated/tuning_inner_grid_heatmap.png
   :width: 100%
   :alt: Heatmaps of mean inner-CV RMSE over network size and alpha for
         feature sets A and B, with the best cell outlined.

   Mean inner-CV RMSE for every setting, averaged over all outer folds and
   seeds; the outlined cell is the best. Alpha = 10 is clearly too strong a
   penalty for every network, and the best alphas are 0.1–1, so the alpha
   range now brackets the optimum. Larger networks do better, but the gain
   shrinks: on set B, (128, 128) and (64, 64) are almost tied at alpha 1.

**Edge-of-grid warning.** The largest network, (128, 128), was chosen in 15 of
15 outer folds on set A and in 9 of 15 on set B. A still larger network might
be slightly better, especially on set A. On set B, the heatmap shows the
improvement from (64, 64) to (128, 128) is already very small.

Early stopping
~~~~~~~~~~~~~~

.. figure:: generated/tuning_loss_curves_A.png
   :width: 100%
   :alt: Training loss and early-stopping R² against epoch for the selected
         MLP in three outer folds, feature set A.

   Feature set A: training loss (top) and early-stopping R² (bottom) per
   epoch for the selected model in three outer folds. The dashed line marks
   the best epoch; the final model keeps those weights.

.. figure:: generated/tuning_loss_curves_B.png
   :width: 100%
   :alt: Training loss and early-stopping R² against epoch for the selected
         MLP in three outer folds, feature set B.

   Feature set B: the same curves.

No fit, inner or refit, reached the ``max_iter`` cap of 2000 epochs: early
stopping ended every one.

Limitations of the tuning
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Early stopping holds back random rows, not whole setups.**
  ``MLPRegressor`` decides when to stop using a random 10% of the rows it is
  trained on. Rows from one setup are strongly correlated, so these
  held-back rows usually have near-identical neighbours from the same setup
  in the training data. The stopping check is therefore mildly optimistic,
  and training may stop a little late. This does not affect the reported
  scores, which come from grouped outer folds that early stopping never sees.
* **Inner folds are grouped but not stratified on chord.**
* **The largest network was often chosen**, so the grid may not contain the
  best size (see above).
* **One weight initialisation.** Every MLP uses ``random_state=0``. Results
  for other initialisations were not tested.

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
