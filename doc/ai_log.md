# AI use log

This log records how AI was used in this project, as required by the brief's
"AI use and reflection" section. Sections 1–5 are a factual record of the AI
sessions. Sections 6 and 7 are for the student's own assessment and
reflection, and are to be written by the student.

## Tool

| Item | Detail |
|---|---|
| Tool | Claude Code (Anthropic), run inside VS Code |
| Model | Claude Opus 5.5 |
| Sessions covered | 21–22 October 2026 (baseline linear model onwards) |
| Traceability | Every commit made with AI help carries a `Co-Authored-By: Claude` line, so `git log` shows which commits involved AI |

Work done before these sessions (data loading, the grouped train/test split,
physics features, exploratory plots and the exploration page) is not covered
by this record.

> **✍️ WRITE THIS YOURSELF (1 of 3):** say whether AI was used for that
> earlier work, and if so, for what. Delete this box when done.

## 1. What the AI was used for

Each task was set by the student in a written prompt; the AI wrote code,
tests and documentation, ran them, and reported results.

| Stage | Student's request | What the AI did | PR |
|---|---|---|---|
| Linear baseline | Linear regression in a scaling pipeline; swappable models and feature sets; RMSE/MAE/R²; breakdowns by chord, speed, angle and frequency band; timing | Wrote `models.py`, `feature_sets.py`, `evaluate.py`, `run_cv.py`; chose the frequency-band edges (half-decades) | #4 |
| Brief cross-check | Check the work against the project brief | Read the brief and listed gaps (tests, validation write-up, README, AI log) | – |
| Performance and interpretation | Profiling and coefficients | Wrote `profiling.py` (timing, memory, FLOP estimates, cProfile) and coefficient effects in dB per SD | #4 |
| Tests | Test functions and report | Wrote pytest tests, including a manufactured-solution test | #4 |
| Results docs | Put results and plots in the HTML docs | Generated tables and figures, wrote `doc/results.rst` | #4 |
| Summary statistic | Report SD over all 15 folds, not over 3 seeds | Changed `summarise()` | – |
| MLP and nested CV | Tuned MLP with grouped nested CV, outputs, figures, tests | Wrote `tuning.py`, `run_tuning.py`, `plot_tuning.py`; ran a one-seed timing trial, then the full run | #6 |
| Error analysis | Worst setups and further ideas | Wrote per-setup RMSE, worst-setup spectra, error map, Strouhal peak check, residual heatmap | #7 |
| Final settings and profiling | Freeze final MLP; profile it against linear | Added `FINAL_MLP`, `RUNNER_UP_MLP`; extended the profiler (weights, epochs, FLOPs, machine and thread info) | #8 |
| Validation | Shift tests, physics check, acceptance criteria | Wrote `shift_tests.py`, `physics_checks.py`, `acceptance.py` (committed on its own before any results) | #9 |
| Hold-out | Single final evaluation | Wrote `final.py`, `run_final.py`; ran it once after committing the code | (hold-out branch) |
| Housekeeping | requirements, AI log, README | This file, `requirements.txt`, README | (this branch) |

## 2. How AI output was checked

- **Automated tests:** 123 pytest tests, all passing, with an HTML report in
  `test/test_report.html`. They include:
  - values worked out by hand (metrics, band edges, FLOP counts, weight
    counts);
  - a manufactured solution (data from a known linear formula, recovered
    exactly);
  - leakage checks: no setup split across folds, no hold-out row reaching
    any training or tuning step, and outer-validation rows never seen by the
    inner search.
- **Reproduction checks:**
  - refitting each fold's chosen MLP reproduced the saved outer-fold RMSE to
    10 decimal places;
  - linear results were checked to be unchanged after each refactor.
- **Numerical checks against the data**, e.g. condition numbers, peak
  prominences and the counts quoted in the docs, which were recomputed before
  being written down.
- **Every figure was viewed** before being committed. Several were corrected
  (see section 4).
- **Documentation:** the Sphinx docs were rebuilt after every change and
  required to build with no warnings.
- **The project brief** was read and work was checked against it from the
  second session on.

## 3. Decisions that stayed with the student

- **Scope and methods:** the models compared, the feature sets, the metrics
  and the breakdowns, the grouped splitting rules, the seeds, and the
  "never touch the hold-out" rule.
- **Summary statistic:** the student asked for the spread over all 15 folds
  rather than over the 3 seeds, so that it answers "how much does the score
  change from one batch of unseen setups to the next?".
- **Tuning grid:** the student asked which settings had won before deciding.
  They chose to add alphas 1 and 10, and then, after asking for a timing
  comparison, the (128, 128) network.
- **Runner-up model:** the AI recommended (64, 64) with alpha 1.0. The
  student kept their original choice, alpha 0.1.
- **Final model:** (128, 128) with alpha 1.0 on feature set B, chosen from the
  nested-CV results.
- **Acceptance criteria and thresholds:** C1–C3 and P1, set by the student
  before the hold-out run.
- **Documentation scope:** the student removed a "Method" (formulas) page the
  AI had added.
- **Hold-out policy:** run once; the selected model does not change, even
  though the runner-up scored better on the hold-out.
- **Removing a non-blind criterion:** after the hold-out run, the student
  removed acceptance criterion C1a ("beats linear in at least 13 of 15 CV
  folds") because its outcome was already known when it was set. The student
  also checked in the git history that P1's threshold was committed before
  the physics check existed.
- **Interpretation of the results:** in the student's words: "Early versions
  of the HTML results pages included AI-drafted interpretation of the
  figures. I reviewed it, replaced it with factual captions, and wrote my own
  interpretation in the report." The earlier versions remain in the git
  history.

## 4. AI errors found and corrected

| What went wrong | How it was found | Fix |
|---|---|---|
| Did not check work against the project brief at first | Student asked | Brief read; requirements recorded |
| Used a pandas call that is being deprecated | Warning when run | Rewritten |
| Type hints (`X \| None`) broke the docs build because pandas/matplotlib are mocked there (happened twice) | Sphinx import errors | Postponed annotations |
| Test tolerance (1e-9) too tight for floating-point round-off | Test failure | Loosened to 1e-6 with explanation |
| Claimed the set-B input matrix was "close to singular" | AI computed the condition number (≈9) before publishing | Claim corrected (that page was later removed) |
| Timing script passed layer sizes positionally (`MLPRegressor`'s first argument is `loss`) | Error | Keyword arguments everywhere |
| Underestimated nested-CV runtime by about 2× | One-seed timing trial | Estimate revised before the full run |
| Docstring said early stopping ends "20 epochs after the best epoch" | Loss-curve figure contradicted it | Corrected and checked against scikit-learn source |
| Shortcut function would miscount fits with non-default inner folds | Code review | Replaced by storing the value |
| Shell escaping put a hidden control character into a test file | Syntax error | Rewritten |
| `regressor__` in a comment broke the docs (read as a link) | Sphinx error | Quoted as code |
| Worst-setup ranks shown as "#1.0" | Viewing the figure | Fixed |
| Overstated tonal-peak finding (wrong peak height; "all four" when it was three) | Checked against computed data | Corrected on the results page |
| "Final MLP trains 1.1× slower than runner-up" | A third profiling run contradicted it | Reworded to the observed 0.9–1.2× range |
| Profiler test failed (local function cannot be pickled) | Test failure | Module-level recording class |
| Physics-check figure labelled the nested-CV MLP as "final MLP" | Viewing the figure | Label corrected |
| Figure legend covered a bar; cramped titles; unreadable axis labels | Viewing the figures | Layout fixed |
| Created an unnecessary branch when the student had already made one | Noticed straight away | Deleted (no commits) |
| Hold-out scatter plot looked up the wrong seed | End-to-end test with a stand-in fold, **before** the real run | Fixed; the real hold-out ran once without error |
| A text replacement broke a line of `run_final.py` | Read back before committing | Fixed |

## 5. Limitations of the AI's help worth noting

- **Confident first drafts:** several first-draft claims were stated more
  confidently than the evidence supported (section 4). They were only caught
  because results were rechecked against the data.
- **Runtime estimates** were optimistic until measured.
- **Unasked-for additions:** the AI sometimes added things not requested (a
  formulas page, extra labelling). The student removed or accepted these
  case by case.

## 6. Approximate level of AI contribution

> **✍️ WRITE THIS YOURSELF (2 of 3):** your own estimate of how much of the
> work was AI-generated, and why you used AI for those parts. Delete this box
> when done.

For reference, from the record above: most of the code, tests and
documentation from the linear baseline onwards was drafted by the AI to the
student's written specifications. The student set the tasks, the rules and
the thresholds, and made the modelling decisions in section 3.

## 7. Reflection

> **✍️ WRITE THIS YOURSELF (3 of 3):** your reflection, in your own words.
> Delete this box when done.

The brief asks for:

- why AI was used for these tasks;
- how AI use affected your understanding and workflow;
- what you would do differently;
- which parts you can now explain without AI (the interview is without AI).
