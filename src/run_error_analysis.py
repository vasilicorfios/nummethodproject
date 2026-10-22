"""Entry point that finds the setups the models predict worst.

Each wind-tunnel setup (one chord, velocity and angle) gives one noise
spectrum. This script computes the out-of-fold RMSE of every setup for the
linear baseline and the tuned MLP, ranks setups by the MLP's RMSE (the
better model, so its failures are the interesting ones) and plots the worst
spectra::

    python src/run_error_analysis.py                     # sets A and B, 10 worst
    python src/run_error_analysis.py --feature-sets B --n-worst 5

The MLP's predictions are rebuilt from ``results/tuning_outer_scores.csv``
(run ``src/run_tuning.py`` first) by refitting each fold's chosen setting;
see :func:`functions.tuning.refit_outer_predictions`. This takes about 30 s
per feature set. The test hold-out is never used.

Writes, for each feature set ``<fs>``:

* ``results/setup_rmse_<fs>.csv``: every setup with both models' RMSE and bias.
* ``results/worst_setups_<fs>.csv``: the worst setups by MLP RMSE.
* ``doc/generated/worst_setups_<fs>.csv``: a rounded copy for the docs.
* ``doc/generated/worst_setups_spectra_<fs>.png`` and ``.pdf``: their spectra.
"""

import argparse

import pandas as pd

from functions.data_loader import load_airfoil_data
from functions.evaluate import cross_validate, per_setup_rmse
from functions.physics_features import add_physics_features
from functions.plot_results import plot_setup_spectra
from functions.plot_tuning import save_png_pdf
from functions.tuning import refit_outer_predictions
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


def main(argv: list[str] | None = None) -> None:
    """Rank setups by error, print the worst, and save tables and figures."""
    args = parse_args(argv)
    data = add_physics_features(load_airfoil_data())
    outer_scores = pd.read_csv(RESULTS_DIR / "tuning_outer_scores.csv")

    for feature_set in args.feature_sets:
        linear = cross_validate(data, "linear", feature_set)
        # Rebuild the MLP's predictions; this also checks they match the
        # saved tuning results fold by fold.
        mlp = refit_outer_predictions(data, outer_scores, feature_set)
        table = setup_comparison(per_setup_rmse(mlp, data), per_setup_rmse(linear, data))
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

    print(f"\nCSV files written to {RESULTS_DIR}")
    print(f"Documentation tables and figures written to {DOC_TABLES_DIR}")


if __name__ == "__main__":
    main()
