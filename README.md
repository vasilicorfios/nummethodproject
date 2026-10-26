# Airfoil self-noise prediction

MMA3001 Numerical Methods and Machine Learning project.

This project predicts the noise an airfoil makes from its test conditions. The data is the NASA/UCI **Airfoil Self-Noise** dataset: 1,503 wind-tunnel measurements of NACA 0012 airfoils (Brooks, Pope & Marcolini, NASA RP-1218, 1989).

A linear least-squares baseline is compared with a neural network (MLP) tuned by nested cross-validation. The final model is checked by:
- robustness shift tests;
- a physics check;
- acceptance criteria fixed in advance;
- a single evaluation on a held-out test set.

## Data

Each row is one measurement: the sound pressure level at one frequency for one wind-tunnel setup (one chord, velocity and angle). There are 106 setups, each giving one noise spectrum of 8–19 frequencies.

### Inputs and output (feature set A)

| Quantity | Column | Unit | Range in the dataset | Distinct values | Role |
|---|---|---|---|---|---|
| Frequency | `frequency_hz` | Hz | 200 – 20,000 | 21 (1/3-octave band centres) | input |
| Angle of attack | `attack_angle_deg` | deg | 0 – 22.2 | 27 | input |
| Chord length | `chord_length_m` | m | 0.0254 – 0.3048 | 6 | input |
| Free-stream velocity | `velocity_mps` | m/s | 31.7 – 71.3 | 4 | input |
| Suction-side displacement thickness | `disp_thickness_m` | m | 0.000401 – 0.0584 | 105 | input |
| Scaled sound pressure level (SPL) | `spl_db` | dB | 103.4 – 141.0 | continuous | **output** |

### Physics-informed features (added in feature set B)

Feature set **B** is the five inputs above plus three features computed from them (8 inputs in total). They are calculated row by row in `src/functions/physics_features.py`, so they never mix information between rows.

| Feature | Column | Formula | Unit | Range in the dataset |
|---|---|---|---|---|
| Log frequency | `log_frequency` | log₁₀(f) | – (f in Hz) | 2.30 – 4.30 |
| Strouhal number | `strouhal` | f · δ\* / U | dimensionless | 0.0083 – 9.29 |
| Chord Reynolds number | `reynolds` | U · c / ν, with ν = 1.5 × 10⁻⁵ m²/s (air at about 20 °C) | dimensionless | 53,700 – 1,449,000 |

Here f is the frequency, δ\* the displacement thickness, U the free-stream velocity and c the chord.

The input ranges above are also the ranges the final model was trained on. Predictions outside them are refused (see [Predicting for new conditions](#predicting-for-new-conditions)).

## Results

All splits keep each wind-tunnel setup (one chord, velocity and angle, i.e. one noise spectrum) entirely in training or entirely in testing. The test hold-out (16 setups, 227 rows; about 15% of setups, stratified on chord) was used once, at the very end.

| Model (feature set B) | Cross-validation RMSE | Hold-out RMSE | Hold-out R² |
|---|---|---|---|
| Linear least squares (baseline) | 4.58 ± 0.25 dB | 4.76 dB | 0.51 |
| **Final MLP**: (128, 128), alpha = 1 | 2.36 ± 0.23 dB (nested CV) | **2.63 dB** | **0.85** |
| Runner-up MLP: (64, 64), alpha = 0.1 | 2.54 ± 0.12 dB (inner CV) | 2.43 dB | 0.87 |

Cross-validation figures are mean ± SD over 15 folds (5 folds × 3 seeds).

Acceptance criteria (committed before the hold-out run and before the physics check was written):

| Criterion | Threshold | Final MLP | Result |
|---|---|---|---|
| C1: beats linear on the hold-out | MLP RMSE < linear RMSE | 2.63 vs 4.76 dB | PASS |
| C2: hold-out RMSE within CV mean + 2 SD | ≤ 2.82 dB | 2.63 dB | PASS |
| C3: hold-out RMSE under the usefulness limit | ≤ 3.0 dB | 2.63 dB | PASS |
| P1: faster flow predicted louder | ≥ 90% of speed pairs | 97% | PASS |

A fifth criterion, C1a ("beats linear in at least 13 of 15 CV folds"), was removed because its outcome was already known when it was set. The final model was not changed after the hold-out run.

Full results and figures are in the HTML documentation (`doc/_build/html/index.html`).

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

## Predicting for new conditions

```python
import pandas as pd
from functions.predict import predict_spl   # run from src/, or add src/ to the path

predict_spl(pd.DataFrame({"frequency_hz": [1000], "attack_angle_deg": [5.0],
                          "chord_length_m": [0.1524], "velocity_mps": [55.5],
                          "disp_thickness_m": [0.0028]}))   # about 131.5 dB
```

Inputs are checked first. Missing values, physically impossible values (zero or negative frequency, chord, velocity or thickness; negative angles) and values outside the trained range (frequency 200–20,000 Hz, angle 0–22.2°, chord 0.0254–0.3048 m, velocity 31.7–71.3 m/s) stop with an error naming the column and row. Nothing is imputed. See "Valid inputs" in the HTML docs (Usage page).

## Running the analysis

Run every command from the project root. Later scripts read results written by earlier ones, so on a fresh clone run them in this order.

| Command | What it does | Runtime |
|---|---|---|
| `python src/explore_plots.py` | Exploratory figures of the dataset | seconds |
| `python src/run_cv.py` | Linear baseline: repeated grouped 5-fold CV on feature sets A and B | seconds |
| `python src/run_tuning.py` | Tunes the MLP by nested CV and compares it with linear on the same folds | about 45 min (see below) |
| `python src/run_error_analysis.py` | Ranks wind-tunnel setups by prediction error, plots the 10 worst spectra, and adds an error map, the share of error at each spectral peak and a tonal-peak plot | about 1 min |
| `python src/run_profile.py` | Timing, memory, size and FLOP profile of linear (A, B), the final MLP and the runner-up MLP (B), with machine and thread settings | about 25 s |
| `python src/run_shift_tests.py` | Robustness: leave one chord / one velocity out, final MLP (3 seeds) vs linear | about 1 min |
| `python src/run_physics_check.py` | Physics check (faster flow is louder) on out-of-fold predictions, then the acceptance criteria table | about 5 s |
| `python src/run_final.py` | **Run once.** Final hold-out evaluation of linear, runner-up and final MLP, and the acceptance criteria (refuses to overwrite without `--force`) | about 10 s |
| `python src/plot_holdout_residuals.py` | Residual vs measured SPL plot for the three models, drawn from the saved hold-out predictions (no re-run) | a few seconds |
| `python -m pytest --html=test/test_report.html --self-contained-html` | Runs the tests and writes the HTML test report | about 50 s |

Results are written as CSV files to `results/`. Rounded tables and figures go to `doc/generated/`. Rebuild the HTML docs with `sphinx-build -b html doc doc/_build/html`.

### Tuning the MLP (nested cross-validation)

For each outer fold, an inner cross-validation, run only on that fold's training setups, picks the network size and `alpha`. The chosen model is then scored on the outer fold, which played no part in choosing it. Every split, inner and outer, keeps all rows of a wind-tunnel setup together, and the test hold-out is never used.

```
python src/run_tuning.py                                # all 3 seeds, feature sets A and B
python src/run_tuning.py --seeds 42 --feature-sets A    # one seed, one set: quick check
```

- `--seeds`: outer CV seeds to run, any of `42 43 44` (default: all three).
- `--feature-sets`: `A` and/or `B` (default: both).

**Runtime** (8-core laptop, 35-setting grid): the full run is 35 settings × 5 inner folds × 5 outer folds × 3 seeds = 2,625 fits per feature set, plus refits, and takes about **44 minutes** for sets A and B together (measured: 23.8 min for A, 19.8 min for B). A single seed on one feature set takes about 7–8 minutes.

## AI use

AI (Claude Code) was used to help write code, tests and documentation. The tool, what it was used for, how its output was checked and the errors it made are recorded in [`doc/ai_log.md`](doc/ai_log.md). Commits made with AI help carry a `Co-Authored-By: Claude` line.
