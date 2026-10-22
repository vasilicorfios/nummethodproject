# nummethodproject

MMA3001 Numerical Methods and Machine Learning project.

## Structure

- `src/` — source code
- `data/` — datasets (large datasets should be hosted locally or on Google Drive instead of committed here)
- `doc/` — documentation
- `test/` — test functions
- `LICENSE` — project license

## Running the analysis

Run every command from the project root.

| Command | What it does | Runtime |
|---|---|---|
| `python src/run_cv.py` | Linear baseline: repeated grouped 5-fold CV on feature sets A and B | seconds |
| `python src/run_profile.py` | Timing, memory, size and FLOP profile of linear (A, B), the final MLP and the runner-up MLP (B), with machine and thread settings | about 25 s |
| `python src/run_tuning.py` | Tunes the MLP by nested CV and compares it with linear on the same folds | about an hour (see below) |
| `python src/run_error_analysis.py` | Ranks wind-tunnel setups by prediction error, plots the 10 worst spectra, and adds an error map, a Strouhal-scaling check on spectral peaks and a residual heatmap (needs `run_tuning.py` results) | about 1 min |
| `python -m pytest --html=test/test_report.html --self-contained-html` | Runs the tests and writes the HTML test report | about 10 s |

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
