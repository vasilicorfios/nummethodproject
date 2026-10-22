Data exploration
================

These figures show the full dataset (all 1503 rows, including the test
hold-out) before any model is fitted. They are written to ``doc/generated/``
by ``src/explore_plots.py``. To update this page:

.. code-block:: console

    python src/explore_plots.py --no-show
    sphinx-build -b html doc doc/_build/html

Distribution of the target
--------------------------

.. figure:: generated/explore_spl_distribution.png
   :width: 85%
   :alt: Histogram of scaled sound pressure level in 1 dB bins, with dashed
         and dotted lines at the mean and median.

   Number of observations in each 1 dB band of scaled sound pressure level.

SPL ranges from about 103 to 141 dB, with a mean of 124.8 dB and a standard
deviation of 6.9 dB. The distribution is skewed to the left (skewness about
−0.4): most measurements lie between 120 and 135 dB, with a longer tail of
quiet conditions. The mean is slightly below the median (125.7 dB) for the
same reason. Few rows lie in either tail, so models have little data to
learn from there. This helps explain why the linear baseline pulls its
predictions towards the mean (see :doc:`results`).

Feature set A: the raw inputs
-----------------------------

.. figure:: generated/explore_spl_vs_features.png
   :width: 100%
   :alt: Five scatter plots of SPL against frequency, angle of attack, chord
         length, free-stream velocity and displacement thickness.

   SPL against each of the five raw inputs. Frequency is on a log axis.

Chord length and velocity take only six and four distinct values, and the
angle of attack 27, so the data form vertical bands rather than a
continuous cloud. Within every band SPL spreads over 20 to 30 dB, so no
single input explains much of the variance on its own. The average SPL
rises slightly up to about 800 Hz and then falls at higher frequencies. This hump is a
non-linear shape that a straight line in frequency cannot follow.

Feature set B: the physics-derived features
-------------------------------------------

.. figure:: generated/explore_spl_vs_physics_features.png
   :width: 100%
   :alt: Three scatter plots of SPL against log frequency, Strouhal number
         and chord Reynolds number.

   SPL against the three features that set B adds to set A: log10
   frequency, Strouhal number and chord Reynolds number. Strouhal and
   Reynolds numbers are on log axes.

The Strouhal number, which combines frequency, displacement thickness and
velocity, collapses the data far more tightly than any raw input. Most of
the loudest points lie at Strouhal numbers of roughly 0.03 to 0.2, and SPL
falls steadily at higher values. This suggests
the spectra share a similar shape once frequency is scaled this way. The
relationship is still curved, so even with this feature a linear model
cannot fit it fully. Reynolds number, like chord and velocity, shows
vertical bands with a wide SPL spread in each.
