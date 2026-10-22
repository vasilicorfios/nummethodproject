# Airfoil self-noise prediction

MMA3001 Numerical Methods and Machine Learning project.

This project predicts the noise an airfoil makes from its test conditions. The data is the NASA/UCI **Airfoil Self-Noise** dataset: 1,503 wind-tunnel measurements of NACA 0012 airfoils (Brooks, Pope & Marcolini, NASA RP-1218, 1989).

- **Inputs:** frequency (Hz), angle of attack (deg), chord length (m), free-stream velocity (m/s) and suction-side displacement thickness (m). Feature set **B** adds log-frequency, Strouhal number and chord Reynolds number.
- **Output:** scaled sound pressure level (SPL, dB).

A linear least-squares baseline is compared with a neural network (MLP) tuned by nested cross-validation. The final model is checked by:
- robustness shift tests;
- a physics check;
- acceptance criteria fixed in advance;
- a single evaluation on a held-out test set.

## Results

All splits keep each wind-tunnel setup (one chord, velocity and angle, i.e. one noise spectrum) entirely in training or entirely in testing. The test hold-out (16 setups, 227 rows; about 15% of setups, stratified on chord) was used once, at the very end.

| Model (feature set B) | Cross-validation RMSE | Hold-out RMSE | Hold-out R² |
|---|---|---|---|
| Linear least squares (baseline) | 4.58 ± 0.25 dB | 4.76 dB | 0.51 |
| **Final MLP**: (128, 128), alpha = 1 | 2.36 ± 0.23 dB (nested CV) | **2.63 dB** | **0.85** |
| Runner-up MLP: (64, 64), alpha = 0.1 | 2.54 ± 0.12 dB (inner CV) | 2.43 dB | 0.87 |

Cross-validation figures are mean ± SD over 15 folds (5 folds × 3 seeds).

- **The final MLP passes all acceptance criteria**, which were committed before the hold-out run:
  - it beats linear in 15 of 15 CV folds and on the hold-out;
  - its hold-out RMSE (2.63 dB) is within CV mean + 2 SD (2.82 dB) and under the 3.0 dB usefulness limit;
  - it predicts that faster flow is louder for 97% of speed pairs.
- **The runner-up scored slightly better on the hold-out.** This is reported, but the selected model was not changed afterwards: choosing by hold-out score would make the test dishonest, and 16 setups cannot settle a 0.2 dB difference.
- **The hardest conditions** are small chords at high angles of attack, where narrow tonal peaks occur. Leaving the smallest chord out of training entirely roughly doubles the MLP's error.

Full results, figures and the method are in the HTML documentation (`doc/_build/html/index.html`).

## Repository structure

| Path | Contents |
|---|---|
| `src/run_*.py`, `src/explore_plots.py` | Entry-point scripts, one per analysis (see the table below) |
| `src/functions/` | Reusable modules: data loading and features (`data_loader`, `physics_features`, `feature_sets`), splitting (`make_folds`), models (`models`), evaluation (`evaluate`, `tuning`, `final`), validation (`shift_tests`, `physics_checks`, `acceptance`, `error_analysis`), profiling (`profiling`) and plotting (`plot_*`) |
| `data/airfoil_self_noise.xlsx` | The dataset (included; about 75 kB) |
| `results/` | Full-precision CSV outputs, cProfile reports, hold-out run info and the saved final models (`results/final_models/`) |
| `doc/` | Sphinx sources (`*.rst`), generated tables and figures (`doc/generated/`), built HTML (`doc/_build/html/`) and the AI use log (`doc/ai_log.md`) |
| `test/` | pytest tests and the HTML test report (`test/test_report.html`) |
| `requirements.txt` | Exact package versions used |
| `LICENSE` | GNU GPL v3 |

## Setup

Python 3.10 was used. From the project root:

```
pip install -r requirements.txt
```

## Running the analysis

Run every command from the project root. Later scripts read results written by earlier ones, so on a fresh clone run them in this order.

| Command | What it does | Runtime |
|---|---|---|
| `python src/explore_plots.py` | Exploratory figures of the dataset | seconds |
| `python src/run_cv.py` | Linear baseline: repeated grouped 5-fold CV on feature sets A and B | seconds |
| `python src/run_tuning.py` | Tunes the MLP by nested CV and compares it with linear on the same folds | about 45 min (see below) |
| `python src/run_error_analysis.py` | Ranks wind-tunnel setups by prediction error, plots the 10 worst spectra, and adds an error map, a Strouhal-scaling check on spectral peaks and a residual heatmap | about 1 min |
| `python src/run_profile.py` | Timing, memory, size and FLOP profile of linear (A, B), the final MLP and the runner-up MLP (B), with machine and thread settings | about 25 s |
| `python src/run_shift_tests.py` | Robustness: leave one chord / one velocity out, final MLP (3 seeds) vs linear | about 1 min |
| `python src/run_physics_check.py` | Physics check (faster flow is louder) on out-of-fold predictions, then the acceptance criteria table | about 5 s |
| `python src/run_final.py` | **Run once.** Final hold-out evaluation of linear, runner-up and final MLP, and the acceptance criteria (refuses to overwrite without `--force`) | about 10 s |
| `python -m pytest --html=test/test_report.html --self-contained-html` | Runs the tests and writes the HTML test report | about 50 s |

Results are written as CSV files to `results/`. Rounded tables and figures go to `doc/generated/`. Rebuild the HTML docs with `sphinx-build -b html doc doc/_build/html`.

### Tuning the MLP (nested cross-validation)

Choosing hyperparameters and scoring the model on the same folds would flatter the result. Nested cross-validation separates the two jobs. For each outer fold, an inner cross-validation, run only on that fold's training setups, picks the network size and `alpha`. The chosen model is then scored on the outer fold, which played no part in choosing it. Every split, inner and outer, keeps all rows of a wind-tunnel setup together, and the test hold-out is never used.

```
python src/run_tuning.py                                # all 3 seeds, feature sets A and B
python src/run_tuning.py --seeds 42 --feature-sets A    # one seed, one set: quick check
```

- `--seeds`: outer CV seeds to run, any of `42 43 44` (default: all three).
- `--feature-sets`: `A` and/or `B` (default: both).

**Runtime** (8-core laptop, 35-setting grid): the full run is 35 settings × 5 inner folds × 5 outer folds × 3 seeds = 2,625 fits per feature set, plus refits, and takes about **44 minutes** for sets A and B together (measured: 23.8 min for A, 19.8 min for B). A single seed on one feature set takes about 7–8 minutes.

## AI use

AI (Claude Code) was used to help write code, tests and documentation. The tool, what it was used for, how its output was checked and the errors it made are recorded in [`doc/ai_log.md`](doc/ai_log.md). Commits made with AI help carry a `Co-Authored-By: Claude` line.
