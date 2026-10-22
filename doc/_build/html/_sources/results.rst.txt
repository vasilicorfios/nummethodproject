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

Worst-predicted setups
----------------------

Each wind-tunnel setup (one chord, velocity and angle) gives one measured
noise spectrum. To see *which* conditions the models fail on, the
out-of-fold RMSE of every one of the 90 cross-validation setups was
computed. In each CV repeat a setup is predicted once, by a model that never
saw it, and its RMSE is averaged over the 3 seeds. Setups are ranked by the
tuned MLP's RMSE, because the better model's failures are the informative
ones. The table and figure are for feature set B, the best combination.

.. csv-table:: The 10 setups with the highest tuned-MLP RMSE, feature set B (bias = mean of measured − predicted; positive means the model is too low)
   :file: generated/worst_setups_B.csv
   :header-rows: 1

.. figure:: generated/worst_setups_spectra_B.png
   :width: 100%
   :alt: Ten panels of SPL against frequency for the worst-predicted setups,
         comparing the measured spectrum with linear and tuned-MLP predictions.

   Measured spectra (black) and out-of-fold predictions (averaged over the 3
   seeds) for the 10 worst setups, worst first. Each panel has its own SPL
   axis.

What the worst setups have in common:

* **They cluster at the smallest chord and the most extreme conditions.**
  6 of the 10 have the smallest chord (0.0254 m), 6 are at an angle of attack
  of 17.4° or more, and 5 are at the highest velocity (71.3 m/s). This
  matches the RMSE breakdown above.
* **Several have a sharp, narrow tonal peak** standing 18–21 dB above the
  neighbouring troughs (ranks 1, 4, 5 and 9: all 0.0254 m chord at
  17.4–22.2°). Both models flatten the peak. For three of these four, the
  frequencies around the peak hold 64–84% of the MLP's squared error, so
  the miss is concentrated there. Rank 5 is the exception: its error is
  spread along the spectrum (see `Where along the spectrum the errors
  are`_). The smallest chord also gives the lowest chord
  Reynolds numbers in the dataset (about 5×10⁴ to 1.2×10⁵). Narrow peaks
  like these are consistent with tonal noise mechanisms that appear at low
  Reynolds number, such as laminar boundary-layer vortex shedding. The
  data cannot confirm the mechanism, but it explains why a model trained
  mostly on smooth broadband spectra struggles here.
* **The problem setups are the same for both feature sets.** 9 of the 10
  worst setups for set A are also among the 10 worst for set B
  (``results/worst_setups_A.csv``).
* **The MLP is not better everywhere.** It beats linear on 8 of these 10
  setups, but is worse on ranks 3 and 8 (0.0508 m chord at 39.6 m/s), where
  it puts a peak at the wrong frequency.

Very few training setups resemble these conditions. When one of them is held
out, the model sees at most a few setups with the same chord at nearby
angles, so it has to extrapolate. These setups are therefore where the model
is least trustworthy, and the main limit on its use near stall and at small
chords.

Error map over the test conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: generated/error_map_B.png
   :width: 100%
   :alt: Panels for each free-stream velocity placing every setup by angle of
         attack and chord, coloured by its RMSE, for linear and tuned MLP.

   Every cross-validation setup placed by its test conditions (one panel per
   velocity, angle across, chord up) and coloured by RMSE on one shared
   scale; top row linear, bottom row tuned MLP (feature set B). Numbers mark
   the 10 worst MLP setups. Hollow grey circles are held-out test setups,
   shown only to make gaps in the test matrix visible (they are not scored).

The map shows that the tested conditions do not form a full grid: each chord
was tested at its own set of angles, and some combinations are missing.
Linear's errors are large across the whole matrix. The MLP's are small in
most places and concentrate at the smallest chord at high angle, and at the
highest velocity. Those are the edges of the tested region, where a held-out
setup has the fewest similar neighbours to learn from.

Do the peaks follow Strouhal scaling?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A noise source that sheds at a fixed **Strouhal number** ``St = f L / U``
(``L`` a length scale of the flow) has a peak frequency proportional to
``U / L``. On log axes that is a straight line of slope 1. This is a physical
check on the data itself, made with the measured spectra of the
cross-validation setups only. A peak is the most prominent interior maximum
of a spectrum that stands at least 5 dB above its neighbouring troughs; 64 of
the 90 setups have one. The conclusions below were the same with 3 dB and
8 dB thresholds.

.. csv-table:: Fit of log(peak frequency) against log(U / L). Constant Strouhal scaling would give slope 1 and a much smaller spread of St than of f
   :file: generated/peak_scaling_B.csv
   :header-rows: 1

.. figure:: generated/peak_scaling_B.png
   :width: 100%
   :alt: Left, measured peak frequency against U over displacement thickness
         on log axes with fitted and slope-1 lines. Right, tuned-MLP predicted
         peak frequency against measured peak frequency.

   Left: measured peak frequency against ``U / δ*``. The dashed line is
   constant-Strouhal scaling (slope 1) through the median Strouhal number;
   the solid line is the fitted slope. The horizontal bands are the discrete
   1/3-octave frequencies. Right: frequency of the tuned MLP's predicted peak
   against the measured peak; marker area counts setups, and dotted lines
   are one 1/3-octave step either side of perfect agreement.

* **No single Strouhal number describes the peaks.** The fitted slope
  against ``U / δ*`` is about 0.5, with a 95% interval well below 1, and the
  Strouhal number at the peak varies about as much as the peak frequency
  itself. The chord is a worse length scale still (lower R²). This fits the
  peaks coming from different noise mechanisms under different conditions,
  for example turbulent trailing-edge noise, tonal shedding at low Reynolds
  number and separated flow at high angle, each with its own scaling. The
  dataset also gives only the *suction-side* displacement thickness. Some
  mechanisms scale with other lengths, such as the pressure-side boundary
  layer.
* **The coarse frequency spacing limits this test.** Peaks are known only to
  the nearest 1/3-octave step (about ±12%), and each geometry was tested at
  only 3–4 velocities.
* **The MLP puts its peak in the right place.** For 92% of the peaked
  setups, its predicted spectrum peaks within one 1/3-octave step of the
  measured peak. The linear model never does: its predicted spectra only
  slope one way. So the MLP's remaining errors are mostly about the
  *height* of the spectrum, not the *location* of its peak.

Where along the spectrum the errors are
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: generated/residual_heatmap_B.png
   :width: 100%
   :alt: Heatmaps of residual for every setup (rows, grouped by chord) and
         frequency (columns) for linear and tuned MLP, with measured peaks
         marked by dots.

   Residual (measured − predicted, averaged over seeds) for every
   cross-validation setup and frequency. Red means the model predicts too
   low, blue too high, and grey means that frequency was not measured.
   Rows are grouped by chord and ordered by angle, then velocity; dots mark
   measured peaks.

.. csv-table:: Share of each setup's squared error in the peak window (the peak ± one 1/3-octave step), median over setups. If errors were spread evenly, the two columns would match
   :file: generated/peak_error_share_B.csv
   :header-rows: 1

* **Linear has a systematic tilt.** For most chords it predicts too low at
  low frequencies and too high at high frequencies, and misses peaks
  broadly. A straight line in the inputs cannot follow the spectral shape.
* **The MLP's residuals are much smaller and mostly unstructured.** For a
  typical peaked setup, the peak window holds *less* of the MLP's error than
  its share of the frequencies, so ordinary peaks are not where the MLP
  fails.
* **The exception is the strongly tonal setups** (peaks of 15 dB or more,
  all at the smallest chord and high angle). There the peak window, 20% of
  the frequencies, holds most of the MLP's squared error. These narrow spikes
  are the clearest remaining failure. With only four such setups, and each
  held out in turn, the model has almost no examples to learn their height
  from.

Runtime, memory and FLOPs
-------------------------

The final model is frozen from the nested-CV results (CV data only; the test
hold-out was not used): **(128, 128), alpha = 1.0, on feature set B**. It won
7 of the 15 outer folds and had the lowest mean inner-CV RMSE (2.50 dB). A
cheaper **runner-up**, (64, 64) with alpha = 0.1 (2.54 dB, about 4× fewer
weights), is profiled alongside it. The settings live in
``functions.models.FINAL_MLP`` and ``RUNNER_UP_MLP``.

How it was measured
~~~~~~~~~~~~~~~~~~~

* **Training data:** each model is trained on all 1,276 cross-validation
  rows, never the test hold-out.
* **Time:** fit and predict are each repeated 5 times; the table gives the
  median and the fastest (min) run.
* **Memory:** peak memory during a fit is traced with ``tracemalloc``. This is
  a lower bound, since it cannot see memory allocated inside compiled maths
  libraries.
* **Session:** all four models were profiled in the same session on the
  same machine (below).
* **FLOPs are estimates from formulas, not measurements:**

  * linear: ``2np² + 11p³`` for the SVD least-squares solve plus scaling
    (:func:`functions.profiling.linear_flop_estimate`);
  * MLP: about 6 FLOPs per weight per training row per epoch (2 forward,
    about 4 backward), plus 2 per weight for the early-stopping rows each
    epoch (:func:`functions.profiling.mlp_flop_estimate`).

  The fit rate is the estimated FLOPs divided by the median fit time.

.. csv-table:: Profile of one fit on all CV rows (FLOPs are estimates)
   :file: generated/profile_summary.csv
   :header-rows: 1

.. csv-table:: Cost ratios between models
   :file: generated/profile_comparison.csv
   :header-rows: 1

.. csv-table:: Machine and settings used for the timings
   :file: generated/profile_machine.csv
   :header-rows: 1

What the numbers show
~~~~~~~~~~~~~~~~~~~~~

* **Training cost is the price of the MLP's accuracy.** Training the final
  MLP takes about 1.4 s, roughly 400 times longer than the linear fit. It
  has about 2,000 times as many weights and needs about 90,000 times as many
  (estimated) FLOPs. Even so, 1.4 s is small in absolute terms; the cost
  only matters when training is repeated thousands of times, as in nested
  cross-validation.
* **Prediction is cheap for both.** The final MLP predicts all 1,276 rows in
  about 3 ms (a few microseconds per row), only about 3 times slower than
  linear. Fixed overhead, not arithmetic, dominates prediction at this size.
* **Memory is not a constraint.** Peak training memory is about 2 MB. The
  final MLP's 17,793 weights take 142 kB. The pickled file is about 3 times
  larger (438 kB), because scikit-learn also saves the Adam optimiser's two
  running averages per weight, which are only needed to continue training.
* **The bigger network costs no clear extra time.** The final MLP has 3.7×
  the runner-up's weights and needs 2.9× the estimated FLOPs. Yet across
  three profiling runs its fit time was 0.9–1.2× the runner-up's, a
  difference within run-to-run noise. Two reasons: early stopping ended it
  sooner (177 vs 226 epochs), and larger matrix products use the CPU more
  efficiently (about 15 GFLOP/s, against 5–6 GFLOP/s for the runner-up).
  The noise comes from multi-threaded maths on small matrices and the
  laptop's changing clock speed, which is why the table also gives the
  fastest of the 5 repeats.
* **Linear is limited by overhead, not arithmetic.** Its fit needs only
  about 10\ :sup:`5` FLOPs but takes about 3 ms (tens of MFLOP/s). The
  cProfile reports show the time goes to scikit-learn and pandas input
  checks and scaling, not the least-squares solve.

Where the final MLP spends its training time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From ``results/profile_mlp_final_B.txt``: 177 epochs × 6 mini-batches (about
200 rows each) = 1,062 back-propagation steps. These take about three
quarters of the training time. Within that, about a third of the total goes
on matrix multiplications and about a fifth on the ReLU derivative, a simple
element-by-element step. The Adam weight updates take about 12% and the
early-stopping checks about 9%.

The ReLU step does very little arithmetic per number, yet takes over half as
long as the matrix multiplications. That shows the training is limited
mainly by moving data through memory and per-call overhead on small
mini-batches, not by arithmetic. The achieved 15 GFLOP/s is roughly 10–15%
of this processor's theoretical double-precision peak at base clock (4 cores
× 16 FLOPs per cycle × 1.8 GHz ≈ 115 GFLOP/s).

**Threads.** ``OMP_NUM_THREADS`` and the other thread variables were unset,
so OpenBLAS used 8 threads. The MLP's matrix products were therefore
multi-threaded, not single-threaded. Timings on another machine, or with a
different thread setting, will differ; the machine table above records the
setup used.

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
