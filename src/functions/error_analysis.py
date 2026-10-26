"""Where and why the models fail: spectral peaks, Strouhal scaling, residuals.

Three analyses of the out-of-fold predictions, all on the cross-validation
setups only (the test hold-out is never looked at):

* :func:`spectral_peaks` finds the most prominent peak of each measured
  spectrum and the frequency at which a model's predicted spectrum peaks.
* :func:`peak_scaling_fit` checks a physical law: if a peak is produced at a
  fixed Strouhal number ``St = f L / U`` (``L`` a length scale), its
  frequency is proportional to ``U / L``. On log axes that is a straight
  line of **slope 1**. A fitted slope well away from 1 means a single
  Strouhal number does not describe the peaks.
* :func:`residual_matrix` and :func:`peak_error_share` show where along the
  spectrum the errors sit, and how much of each setup's error comes from the
  frequencies around its measured peak.

Example:
    >>> from functions.error_analysis import spectral_peaks, peak_scaling_fit
    >>> peaks = spectral_peaks(data, mlp_result)      # doctest: +SKIP
    >>> peak_scaling_fit(peaks, "disp_thickness_m")   # doctest: +SKIP
"""

# Postponed annotations: the docs build mocks pandas, and evaluating
# ``CVResult | None`` at import time is safest avoided.
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress

from functions.evaluate import CVResult, angle_band
from functions.make_folds import TEST_FOLD, assign_folds

#: A peak must stand this many dB above the lower of its two neighbouring
#: troughs (its "prominence") to count. Below about 5 dB a peak is hard to
#: tell from measurement scatter; the scaling conclusions were checked to be
#: the same at 3 and 8 dB.
PEAK_PROMINENCE_DB = 5.0

#: Peaks at least this prominent are treated as strongly tonal in
#: :func:`peak_error_summary` (narrow spikes well above the broadband level).
TONAL_PROMINENCE_DB = 15.0

#: Frequencies within this many 1/3-octave steps of the measured peak count
#: as "at the peak" in :func:`peak_error_share`.
PEAK_WINDOW_STEPS = 1

#: Columns describing a setup, carried into the peak table.
_SETUP_INFO = ["setup_id", "chord_length_m", "velocity_mps", "attack_angle_deg", "disp_thickness_m"]


def _cv_rows(data: pd.DataFrame) -> pd.DataFrame:
    """The cross-validation rows with ``setup_id`` (test hold-out removed)."""
    folded = assign_folds(data)
    return folded[folded["fold"] != TEST_FOLD]


def _mean_prediction(result: CVResult) -> pd.Series:
    """Out-of-fold prediction per data row, averaged over the CV seeds."""
    return result.oof.groupby("row")["y_pred"].mean()


def spectral_peaks(data: pd.DataFrame, result: CVResult | None = None,
                   prominence_db: float = PEAK_PROMINENCE_DB) -> pd.DataFrame:
    """Find each setup's most prominent measured peak (and predicted peak).

    Peaks are found with :func:`scipy.signal.find_peaks`, which only accepts
    interior points, so a spectrum that simply rises or falls to its end has
    no peak. Frequencies are 1/3-octave centres, so a peak frequency is only
    known to within one 1/3-octave step (about ±12%).

    Args:
        data: Dataset with physics features.
        result: Optional model result; adds the frequency at which the
            model's seed-averaged predicted spectrum is highest.
        prominence_db: Minimum prominence for a measured peak.

    Returns:
        One row per cross-validation setup with the setup columns,
        ``has_peak``, ``peak_hz``, ``prominence_db``, ``peak_index`` (position
        of the peak in the setup's sorted frequencies), ``st_peak``
        (``peak_hz * disp_thickness_m / velocity_mps``) and, with ``result``,
        ``pred_peak_hz`` and ``pred_peak_steps_off`` (1/3-octave steps from
        the measured peak).
    """
    cv = _cv_rows(data)
    predicted = _mean_prediction(result) if result is not None else None
    rows = []
    for setup_id, spectrum in cv.groupby("setup_id"):
        spectrum = spectrum.sort_values("frequency_hz")
        freqs = spectrum["frequency_hz"].to_numpy()
        index, props = find_peaks(spectrum["spl_db"].to_numpy(), prominence=prominence_db)
        row = spectrum.iloc[0][_SETUP_INFO].to_dict()
        row["setup_id"] = int(setup_id)
        if len(index):
            k = index[np.argmax(props["prominences"])]
            row.update(has_peak=True, peak_hz=freqs[k], peak_index=int(k),
                       prominence_db=float(props["prominences"].max()))
        else:
            row.update(has_peak=False, peak_hz=np.nan, peak_index=-1, prominence_db=np.nan)
        if predicted is not None:
            k_pred = int(np.argmax(predicted.loc[spectrum.index].to_numpy()))
            row["pred_peak_hz"] = freqs[k_pred]
            # Adjacent measured frequencies are one 1/3-octave step apart.
            row["pred_peak_steps_off"] = k_pred - row["peak_index"] if row["has_peak"] else np.nan
        rows.append(row)
    peaks = pd.DataFrame(rows)
    peaks["st_peak"] = peaks["peak_hz"] * peaks["disp_thickness_m"] / peaks["velocity_mps"]
    return peaks


def peak_scaling_fit(peaks: pd.DataFrame, length_column: str) -> dict[str, float]:
    """Test whether peak frequencies scale with ``U / L`` (constant Strouhal).

    Fits ``log10(peak_hz) = slope * log10(U / L) + intercept`` over the setups
    that have a peak. Constant-Strouhal scaling predicts ``slope = 1``.

    Args:
        peaks: Output of :func:`spectral_peaks`.
        length_column: Column used as the length scale ``L``, e.g.
            ``"disp_thickness_m"`` or ``"chord_length_m"``.

    Returns:
        ``length_scale``, ``n`` (setups with a peak), ``slope``,
        ``slope_ci95`` (half-width of the 95% confidence interval), ``r2``,
        ``median_st`` (median of ``peak_hz * L / U``) and the spread factors
        ``peak_hz_spread`` and ``st_spread`` (10 to the power of the SD of
        their log10). If Strouhal scaling held, ``st_spread`` would be much
        smaller than ``peak_hz_spread``.
    """
    p = peaks[peaks["has_peak"]]
    x = np.log10(p["velocity_mps"] / p[length_column])
    y = np.log10(p["peak_hz"])
    fit = linregress(x, y)
    st = p["peak_hz"] * p[length_column] / p["velocity_mps"]
    return {
        "length_scale": length_column,
        "n": len(p),
        "slope": fit.slope,
        "slope_ci95": 1.96 * fit.stderr,
        "intercept": fit.intercept,
        "r2": fit.rvalue ** 2,
        "median_st": float(st.median()),
        "peak_hz_spread": float(10 ** np.log10(p["peak_hz"]).std()),
        "st_spread": float(10 ** np.log10(st).std()),
    }


def residual_matrix(result: CVResult, data: pd.DataFrame) -> pd.DataFrame:
    """Mean residual (measured − predicted) for every setup and frequency.

    Args:
        result: Model result.
        data: The dataset the result was computed on.

    Returns:
        A table indexed by ``setup_id`` with one column per measured
        frequency (Hz, ascending). Values are in dB, averaged over seeds;
        NaN where that frequency was not measured for the setup. Rows are
        ordered by chord, then angle, then velocity.
    """
    cv = _cv_rows(data)
    residual = (cv["spl_db"] - _mean_prediction(result).reindex(cv.index)).rename("residual")
    table = pd.concat([cv[["setup_id", "frequency_hz"]], residual], axis=1)
    matrix = table.pivot_table(index="setup_id", columns="frequency_hz", values="residual")
    order = (cv.drop_duplicates("setup_id")
             .sort_values(["chord_length_m", "attack_angle_deg", "velocity_mps"])["setup_id"])
    return matrix.reindex(order)


def peak_error_share(result: CVResult, data: pd.DataFrame, peaks: pd.DataFrame,
                     window: int = PEAK_WINDOW_STEPS) -> pd.DataFrame:
    """How much of each peaked setup's squared error sits around its peak.

    For each setup with a measured peak, the frequencies within ``window``
    1/3-octave steps of the peak form the "peak window". If errors were
    spread evenly along the spectrum, the window's share of the squared
    error would equal its share of the frequencies.

    Args:
        result: Model result.
        data: The dataset the result was computed on.
        peaks: Output of :func:`spectral_peaks`.
        window: Half-width of the peak window in 1/3-octave steps.

    Returns:
        One row per peaked setup: ``setup_id``, ``rmse`` (from the
        seed-averaged prediction), ``error_share`` (fraction of squared
        error in the window) and ``frequency_share`` (fraction of the
        setup's frequencies in the window).
    """
    cv = _cv_rows(data)
    predicted = _mean_prediction(result)
    rows = []
    for _, p in peaks[peaks["has_peak"]].iterrows():
        spectrum = cv[cv["setup_id"] == p["setup_id"]].sort_values("frequency_hz")
        sq = (spectrum["spl_db"] - predicted.loc[spectrum.index]).to_numpy() ** 2
        positions = np.arange(len(spectrum))
        in_window = np.abs(positions - p["peak_index"]) <= window
        rows.append({
            "setup_id": int(p["setup_id"]),
            "rmse": float(np.sqrt(sq.mean())),
            "error_share": float(sq[in_window].sum() / sq.sum()),
            "frequency_share": float(in_window.mean()),
        })
    return pd.DataFrame(rows)


def peak_error_summary(shares: pd.DataFrame, peaks: pd.DataFrame,
                       tonal_db: float = TONAL_PROMINENCE_DB) -> pd.DataFrame:
    """Median peak-window error share for all peaked and strongly tonal setups.

    Args:
        shares: :func:`peak_error_share` output for one model.
        peaks: :func:`spectral_peaks` output.
        tonal_db: Prominence threshold for "strongly tonal".

    Returns:
        Two rows (all peaks, strongly tonal) with ``group``, ``n``,
        ``median_error_share`` and ``median_frequency_share``.
    """
    merged = shares.merge(peaks[["setup_id", "prominence_db"]], on="setup_id")
    groups = {
        f"all peaks (>= {PEAK_PROMINENCE_DB:g} dB)": merged,
        f"strongly tonal (>= {tonal_db:g} dB)": merged[merged["prominence_db"] >= tonal_db],
    }
    return pd.DataFrame([
        {"group": name, "n": len(g),
         "median_error_share": g["error_share"].median(),
         "median_frequency_share": g["frequency_share"].median()}
        for name, g in groups.items()
    ])


def tonal_table(data: pd.DataFrame, include_holdout: bool = False) -> pd.DataFrame:
    """Peak prominence and chord Reynolds number of every setup's spectrum.

    The prominence of a peak is how far it stands above its surroundings:
    its height above the higher of the lowest points on either side before a
    taller peak (:func:`scipy.signal.find_peaks`). For each setup the most
    prominent interior peak is used, with no minimum, so every setup appears.
    A spectrum with no interior peak (it only rises or only falls) gets 0 dB.

    Args:
        data: Dataset with physics features (for the Reynolds number).
        include_holdout: Include the test hold-out setups. Leave False for
            any analysis of model results; True is only for describing the
            whole dataset (exploration).

    Returns:
        One row per setup: ``setup_id``, chord, velocity, angle,
        ``angle_band``, ``reynolds``, ``log10_reynolds``, ``prominence_db``
        and ``peak_hz`` (NaN when there is no interior peak).
    """
    folded = assign_folds(data)
    rows_used = folded if include_holdout else folded[folded["fold"] != TEST_FOLD]
    rows = []
    for setup_id, spectrum in rows_used.groupby("setup_id"):
        spectrum = spectrum.sort_values("frequency_hz")
        index, props = find_peaks(spectrum["spl_db"].to_numpy(), prominence=0)
        first = spectrum.iloc[0]
        has_peak = len(index) > 0
        k = index[np.argmax(props["prominences"])] if has_peak else None
        rows.append({
            "setup_id": int(setup_id),
            "chord_length_m": first["chord_length_m"],
            "velocity_mps": first["velocity_mps"],
            "attack_angle_deg": first["attack_angle_deg"],
            "reynolds": first["reynolds"],
            "prominence_db": float(props["prominences"].max()) if has_peak else 0.0,
            "peak_hz": float(spectrum["frequency_hz"].iloc[k]) if has_peak else np.nan,
        })
    table = pd.DataFrame(rows)
    table["log10_reynolds"] = np.log10(table["reynolds"])
    table["angle_band"] = angle_band(table["attack_angle_deg"])
    return table
