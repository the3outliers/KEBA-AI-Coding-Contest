# KEBA AI Coding Contest 2026 · Level 2

**How much will be charged?** Forecast hourly EV charging energy for 15 anonymized chargers. The final submission combines historical charging profiles with a probability × amount model, then adjusts for recent activity and the contest calendar.

[Open the Level 2 notebook](level_2.ipynb)

## Reproduce the submission

1. Use Python 3.12 and install the dependencies from this directory.

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Place the two original Level 2 contest files in `data/`.

   | File | Purpose |
   | --- | --- |
   | `level_2_train.csv` | 164,160 labeled hourly observations from January 2024 through March 2025 |
   | `level_2_test_submission.csv` | 21,960 forecast rows for April and May 2025 |

3. Start JupyterLab and run all cells in `level_2.ipynb`.

   ```bash
   python -m jupyterlab
   ```

The notebook writes `outputs/level_2_submission.csv` with the columns `unique_id;ds;y_hat` in the original template order. If this folder sits alongside the original `level2_input/` directory, the notebook can read the contest files from there when `data/` is absent.

## Evaluation

About 94% of the training hours have zero charging energy. The notebook shows this sparsity and compares a probability × amount baseline with an always-zero forecast on a February–March 2025 holdout.

The Level 2 pitch reported an MSE of **385,028.9067**, which also appears as the estimated optimum in the archived calibration. Public leaderboard feedback informed the contest-specific blend and smoothing weights. The hidden test labels are needed to independently evaluate the final MSE.

The public release contains the notebook, dependency list and this guide. Contest data, generated predictions and trial submissions are kept separately.
