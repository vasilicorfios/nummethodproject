"""Entry point that finds the setups the models predict worst.

Each wind-tunnel setup (one chord, velocity and angle) gives one noise
spectrum. This script computes the out-of-fold RMSE of every setup for the
linear baseline and the tuned MLP, ranks setups by the MLP's RMSE (the
better model, so its failures are the interesting ones) and plots the worst
spectra::

    python src/run_error_analysis.py                     # sets A and B, 10 worst
    python src/run_error_analysis.py --feature-sets B --n-worst 5

The tuned MLP's out-of-fold predictions are read from
``results/tuning_oof_predictions.csv``, which ``src/run_tuning.py`` writes.
If that file is missing (results from before it was added), they are
rebuilt by refitting each fold's chosen setting
(:func:`functions.tuning.refit_outer_predictions`, about 30 s per feature
set) and the file is written, so later runs can load it directly. Either
way every fold is checked against ``tuning_outer_scores.csv``. The test
hold-out is never used.

It then runs three further analyses (:mod:`functions.error_analysis`):

* an **error map** placing every setup on the test matrix, coloured by RMSE;
* a **Strouhal check**: do measured peak frequencies scale with ``U / δ*``
  (constant Strouhal number), and does the MLP put its peak in the right
  place?
* a **residual heatmap** (setups x frequencies) and the share of each
  setup's error that sits around its measured peak.

Writes, for each feature set ``<fs>``:

* ``results/setup_rmse_<fs>.csv``: every setup with both models' RMSE and bias.
* ``results/worst_setups_<fs>.csv``: the worst setups by MLP RMSE.
* ``doc/generated/worst_setups_<fs>.csv``: a rounded copy for the docs.
* ``doc/generated/worst_setups_spectra_<fs>.png`` and ``.pdf``: their spectra.
* ``results/spectral_peaks_<fs>.csv``: measured and predicted peak per setup.
* ``results/peak_scaling_<fs>.csv``: Strouhal-scaling fits for δ* and chord.
* ``results/peak_error_share_<fs>.csv``: per-setup share of error at the peak.
* ``doc/generated/peak_scaling_<fs>.csv`` and ``peak_error_share_<fs>.csv``:
  rounded tables for the docs.
* ``doc/generated/error_map_<fs>``, ``peak_scaling_<fs>`` and
  ``residual_heatmap_<fs>`` (``.png`` and ``.pdf``): the figures.
"""

import argparse

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.evaluate import cross_validate, per_setup_rmse
from functions.physics_features import add_physics_features
from functions.error_analysis import (
    peak_error_share,
    peak_error_summary,
    peak_scaling_fit,
    residual_matrix,
    spectral_peaks,
)
from functions.plot_error_analysis import plot_error_map, plot_peak_scaling, plot_residual_heatmap
from functions.plot_results import plot_setup_spectra
from functions.plot_tuning import save_png_pdf
from functions.evaluate import CVResult
from functions.tuning import OOF_FILENAME, refit_outer_predictions, tuned_result_from_saved
from run_cv import DOC_TABLES_DIR, RESULTS_DIR

#: Default number of worst setups to report.
N_WORST = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``--feature-sets`` and ``--n-worst`` command-line options.

    Args:
        argv: Arguments to parse; ``None`` reads the command line.

    Returns:
        Namespace with ``feature_sets`` and ``n_worst``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-sets", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--n-worst", type=int, default=N_WORST)
    return parser.parse_args(argv)


def load_tuned_mlp(data: pd.DataFrame, outer_scores: pd.DataFrame,
                   feature_set: str) -> tuple[CVResult, bool]:
    """Get the tuned MLP's out-of-fold predictions, refitting only if needed.

    Args:
        data: Dataset with physics features.
        outer_scores: Contents of ``results/tuning_outer_scores.csv``.
        feature_set: Feature set to load.

    Returns:
        The tuned-MLP result, and ``True`` if it had to be refitted (so it
        should be saved).
    """
    path = RESULTS_DIR / OOF_FILENAME
    if path.exists():
        saved = pd.read_csv(path)
        if feature_set in set(saved["feature_set"]):
            return tuned_result_from_saved(saved, outer_scores, feature_set), False
    # Older results without saved predictions: rebuild them exactly by refitting.
    return refit_outer_predictions(data, outer_scores, feature_set), True


def save_recovered_predictions(recovered: list[CVResult]) -> None:
    """Add refitted predictions to :data:`~functions.tuning.OOF_FILENAME`.

    Rows for other feature sets already in the file are kept.

    Args:
        recovered: Tuned-MLP results rebuilt by refitting.
    """
    path = RESULTS_DIR / OOF_FILENAME
    new = pd.concat([r.oof.assign(feature_set=r.feature_set) for r in recovered], ignore_index=True)
    new.insert(0, "feature_set", new.pop("feature_set"))
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["feature_set"].isin(new["feature_set"])], new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"Saved recovered tuned-MLP predictions to {path}")


def setup_comparison(mlp_setups: pd.DataFrame, linear_setups: pd.DataFrame) -> pd.DataFrame:
    """Put both models' per-setup errors side by side, worst MLP first.

    Args:
        mlp_setups: :func:`functions.evaluate.per_setup_rmse` for the MLP.
        linear_setups: The same for the linear baseline.

    Returns:
        One row per setup with the setup columns, ``n_rows``, ``rmse_mlp``,
        ``bias_mlp``, ``rmse_linear``, ``bias_linear`` and ``rank`` (1 =
        worst for the MLP).
    """
    keys = ["setup_id", "chord_length_m", "velocity_mps", "attack_angle_deg", "n_rows"]
    merged = mlp_setups[keys + ["rmse", "bias"]].merge(
        linear_setups[["setup_id", "rmse", "bias"]], on="setup_id", suffixes=("_mlp", "_linear")
    )
    merged = merged.sort_values("rmse_mlp", ascending=False, kind="stable").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged


def write_doc_table(worst: pd.DataFrame, feature_set: str) -> None:
    """Write a rounded worst-setups table for the HTML results page.

    Args:
        worst: The top rows of :func:`setup_comparison`.
        feature_set: Feature set name, used in the file name.
    """
    DOC_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Rank": worst["rank"],
        "Setup": worst["setup_id"],
        "Chord (m)": worst["chord_length_m"],
        "Velocity (m/s)": worst["velocity_mps"],
        "Angle (deg)": worst["attack_angle_deg"],
        "Frequencies": worst["n_rows"],
        "MLP RMSE (dB)": worst["rmse_mlp"].round(2),
        "MLP bias (dB)": worst["bias_mlp"].round(2),
        "Linear RMSE (dB)": worst["rmse_linear"].round(2),
    }).to_csv(DOC_TABLES_DIR / f"worst_setups_{feature_set}.csv", index=False)


def _percent(values: pd.Series) -> pd.Series:
    """Format fractions as whole percentages, e.g. 0.123 -> "12%"."""
    return (values * 100).round().astype(int).astype(str) + "%"


def run_extra_analyses(data: pd.DataFrame, linear: CVResult, mlp: CVResult,
                       linear_setups: pd.DataFrame, mlp_setups: pd.DataFrame,
                       worst: pd.DataFrame) -> None:
    """Error map, Strouhal check and residual heatmap for one feature set.

    Args:
        data: Dataset with physics features.
        linear: Linear baseline result.
        mlp: Tuned-MLP result.
        linear_setups: :func:`functions.evaluate.per_setup_rmse` for linear.
        mlp_setups: The same for the MLP.
        worst: The worst-setups table (for rank labels on the error map).
    """
    fs = mlp.feature_set
    mlp_label, linear_label = f"mlp (tuned) / {fs}", f"linear / {fs}"

    # 1. Error map: where on the test matrix are the badly predicted setups?
    fig = plot_error_map([(linear_label, linear_setups), (mlp_label, mlp_setups)], data,
                         label_ranks=worst[["setup_id", "rank"]])
    save_png_pdf(fig, DOC_TABLES_DIR, f"error_map_{fs}")

    # 2. Strouhal check on the measured peaks (and the MLP's predicted peaks).
    peaks = spectral_peaks(data, mlp)
    fits = pd.DataFrame([peak_scaling_fit(peaks, L) for L in ("disp_thickness_m", "chord_length_m")])
    peaks.to_csv(RESULTS_DIR / f"spectral_peaks_{fs}.csv", index=False)
    fits.to_csv(RESULTS_DIR / f"peak_scaling_{fs}.csv", index=False)
    pd.DataFrame({
        "Length scale L": fits["length_scale"].map({"disp_thickness_m": "δ* (displacement thickness)",
                                                     "chord_length_m": "c (chord)"}),
        "Setups with a peak": fits["n"],
        "Fitted slope (constant St gives 1)": fits["slope"].round(2).astype(str) + " ± "
                                               + fits["slope_ci95"].round(2).astype(str),
        "R²": fits["r2"].round(2),
        "Median St = f L / U": fits["median_st"].map("{:.3g}".format),
        "Spread of f (×)": fits["peak_hz_spread"].round(2),
        "Spread of St (×)": fits["st_spread"].round(2),
    }).to_csv(DOC_TABLES_DIR / f"peak_scaling_{fs}.csv", index=False)
    save_png_pdf(plot_peak_scaling(peaks, fits.iloc[0].to_dict(), mlp_label), DOC_TABLES_DIR,
                 f"peak_scaling_{fs}")
    n_close = int((peaks["pred_peak_steps_off"].abs() <= 1).sum())
    print(f"\nStrouhal check ({fs}): slope vs U/delta* = {fits['slope'].iat[0]:.2f} +/- "
          f"{fits['slope_ci95'].iat[0]:.2f} (1 = constant Strouhal), R2 {fits['r2'].iat[0]:.2f}; "
          f"MLP peak within one 1/3-octave step in {n_close} of {int(peaks['has_peak'].sum())} peaked setups")

    # 3. Where along the spectrum is the error, and how much is at the peak?
    shares, summaries = [], []
    for label, result in ((linear_label, linear), (mlp_label, mlp)):
        share = peak_error_share(result, data, peaks).assign(model=label)
        shares.append(share)
        summaries.append(peak_error_summary(share, peaks).assign(model=label))
    pd.concat(shares, ignore_index=True).to_csv(RESULTS_DIR / f"peak_error_share_{fs}.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    pd.DataFrame({
        "Model": summary["model"],
        "Setups": summary["group"] + ", n = " + summary["n"].astype(str),
        "Frequencies in peak window (median)": _percent(summary["median_frequency_share"]),
        "Squared error in peak window (median)": _percent(summary["median_error_share"]),
    }).to_csv(DOC_TABLES_DIR / f"peak_error_share_{fs}.csv", index=False)
    print(summary.round(2).to_string(index=False))
    fig = plot_residual_heatmap([(linear_label, residual_matrix(linear, data)),
                                 (mlp_label, residual_matrix(mlp, data))], data, peaks)
    save_png_pdf(fig, DOC_TABLES_DIR, f"residual_heatmap_{fs}")


def main(argv: list[str] | None = None) -> None:
    """Rank setups by error, print the worst, and save tables and figures."""
    args = parse_args(argv)
    data = add_physics_features(load_airfoil_data())
    outer_scores = pd.read_csv(RESULTS_DIR / "tuning_outer_scores.csv")

    recovered = []
    for feature_set in args.feature_sets:
        linear = cross_validate(data, "linear", feature_set)
        # Load (or, for older results, rebuild) the MLP's predictions; both
        # paths check them against the saved tuning scores fold by fold.
        mlp, refitted = load_tuned_mlp(data, outer_scores, feature_set)
        if refitted:
            recovered.append(mlp)
        mlp_setups, linear_setups = per_setup_rmse(mlp, data), per_setup_rmse(linear, data)
        table = setup_comparison(mlp_setups, linear_setups)
        worst = table.head(args.n_worst)

        print(f"\nFeature set {feature_set}: {args.n_worst} worst setups for the tuned MLP "
              f"(of {len(table)}; RMSE averaged over 3 seeds, dB)")
        print(worst.drop(columns="n_rows").round(3).to_string(index=False))

        RESULTS_DIR.mkdir(exist_ok=True)
        table.to_csv(RESULTS_DIR / f"setup_rmse_{feature_set}.csv", index=False)
        worst.to_csv(RESULTS_DIR / f"worst_setups_{feature_set}.csv", index=False)
        write_doc_table(worst, feature_set)

        titles = [
            f"#{int(r['rank'])}  c={r['chord_length_m']:.4f} m, U={r['velocity_mps']:.1f} m/s, "
            f"α={r['attack_angle_deg']:.1f}°\nMLP {r['rmse_mlp']:.2f} dB, linear {r['rmse_linear']:.2f} dB"
            for _, r in worst.iterrows()
        ]
        # Linear first (blue), tuned MLP second (orange), as in every comparison.
        fig = plot_setup_spectra([linear, mlp], data, worst["setup_id"].tolist(), titles=titles)
        save_png_pdf(fig, DOC_TABLES_DIR, f"worst_setups_spectra_{feature_set}")
        run_extra_analyses(data, linear, mlp, linear_setups, mlp_setups, worst)

    if recovered:
        save_recovered_predictions(recovered)
    print(f"\nCSV files written to {RESULTS_DIR}")
    print(f"Documentation tables and figures written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
