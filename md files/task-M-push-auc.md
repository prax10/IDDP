# Task M: pushing overall AUC (honestly)

**Goal:** raise overall test AUC (0.866) as far as possible with legitimate methods: nothing from
after the prediction moment, and nothing chosen on the test set.

**Result: 0.866 → 0.872** (re-tuned XGBoost, now deployed as **v6**). Further routes added ≤ 0.0006.
An overall AUC of 0.90 is not reachable honestly: the early checkpoints (M1 0.81) hold the average
down, and at 10% of an issue's life that information mostly doesn't exist yet.

## Final comparison (test, scored once; 5 seeds; paired bootstrap vs v5b)

| Model | Overall AUC | M1 | M3 | M5 | M8 | M10 | M1–M3 | M5–M10 | Hardest subset* | Δ vs v5b [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| Equal average: re-tuned + comment text + MLP + GRU | 0.8725 | 0.819 | 0.843 | 0.867 | 0.899 | 0.913 | 0.831 | 0.893 | 0.831 | +0.0069 [+0.0064, +0.0073] |
| Re-tuned + comment text | 0.8723 | 0.815 | 0.841 | 0.868 | 0.900 | 0.914 | 0.829 | 0.895 | 0.830 | +0.0067 [+0.0064, +0.0070] |
| **Re-tuned XGBoost (v6, deployed)** | **0.8719** | 0.814 | 0.841 | 0.868 | **0.900** | **0.915** | 0.828 | **0.895** | 0.830 | **+0.0063 [+0.0060, +0.0066]** |
| v5b (previous) | 0.8656 | 0.803 | 0.833 | 0.862 | 0.897 | 0.911 | 0.818 | 0.891 | 0.823 | — |
| Stacking, weights fitted on validation | 0.8601 | 0.795 | 0.828 | 0.857 | 0.890 | 0.903 | 0.812 | 0.885 | 0.820 | −0.0055 |

\* censoring-safe rows whose outcome is not already decided (not yet past the project median).

Accuracy at 0.5: 0.802 → **0.810**. Balanced accuracy: 0.716 → **0.743**. Brier: 0.137 → **0.131**.

## What each step contributed

**1. Re-tuning with rolling-origin validation: +0.0063. This is the whole gain.**
(`task_m1_retune.py`, `reports/m1_rolling_retune.csv`)
- v5b's 43 trees came from early stopping on one validation window that is unusually hard. Every
  model scores lower on it than on test, as noted since Phase 8.
- Tuning was redone over three successive windows of train+val. Seven configurations were tried
  (depth 6/8/10, leaf regularisation, column sampling, recency half-life 2 vs 4 years).
- The landscape is flat: 0.831–0.832 mean window AUC across all of them. What matters is the
  tree count. The two earlier windows want 140–190 trees; the odd last window wants 36.
  Selected: depth 8, min_child_weight 50, **109 trees**.
- Honesty note: on the official validation window alone, the new config is very slightly worse
  (0.784 vs 0.785). On test it is clearly better (+0.0063). That is the window-shift problem the
  rolling validation was meant to address, and selection never looked at test.

**2. Comment text: +0.0004, negligible, not deployed.**
(`task_m2_comment_text.py`)
- 1.52M comments exported from TAWOS_DB.
- TF-IDF + SVD(24) fitted only on comments written before the validation period starts.
- Per checkpoint, using only comments strictly before T: the average and the latest comment's
  representation, plus keyword counts (patch / PR / merged, blocked / waiting, can't reproduce,
  workaround, questions, fixed) and lengths.
- Cross-check: whether an earlier comment exists agrees with the existing comment-count feature on 99.99% of rows.
- **Finding:** what people write adds almost nothing beyond *that* they comment, who, and when. That is
  a useful negative result: the activity signal already carries the comment information.

**3. Ensembles: +0.0006 at most; stacking hurts.**
- An equal-weight average of re-tuned XGB, MLP and GRU gives 0.8725 overall and helps early (M1
  0.819 vs 0.814), but costs a little late.
- Fitting stacking weights on the validation period makes things *worse* (0.860). The
  validation period behaves differently from the test period, so weights fitted there don't transfer.
- Not deployed: the single XGBoost keeps exact explanations (TreeSHAP) and is within 0.0006.

## The numbers that are now in the 90s (v6)

| | v5b | **v6** |
|---|---|---|
| AUC at M8 | 0.897 | **0.900** |
| AUC at M10 | 0.911 | **0.915** |
| AUC, final stretch M8–M10 (ensemble) | 0.904 | **0.908** |
| AUC, late stage M5–M10 | 0.891 | **0.895** |

Triage numbers (not-yet-overdue issues, deployed 5-seed ensemble; `reports/m4_v6_headline.csv`):

| | v5b | **v6** |
|---|---|---|
| Overall AUC of the 5-seed ensemble (what the app shows) | 0.866 | **0.873** |
| "Confident: late" precision | 96.4% | **97.5%** |
| Share of eventually-late issues it is confident about | 27.6% | 23.4% |
| Top-10% flag precision | 98.4% | **98.6%** |

Calibration re-done for v6 (train-only copies, validation calibration, test coverage): 90% sets
cover **91.8%**. With the deployed models, coverage is **92.6%** (late 89.9%, on time 98.9%).
Scrum ranges are unchanged (96–98%). (The seed-mean AUC is 0.8719; the ensemble of the 5 seeds
scores 0.8726. Quote 0.872 or 0.873, and say which.)

## Deployment

- `models/xgb_dynamic_live_v6_*`: 5 seeds; meta with prior-match thresholds and categories.
- The app loads v6 automatically (falling back to v5b if v6 is absent).
- Confidence sets recalibrated for v6: `models/conformal_calibration_v6.json`, coverage in
  `reports/l5_conformal_coverage_v6.csv` and `reports/l5b_deployed_coverage_v6.csv`.

## Paper / viva lines

- **Headline:** "Overall AUC **0.872**; **0.90–0.915** from 80% of expected duration onward."
- **Why not 0.9 overall:** the early checkpoints cap it. Every route we know to a 0.9 overall uses
  information from after the prediction moment.
- **Method point:** single-window validation under-fitted the model. Rolling-origin validation
  fixed it (+0.006), with selection blind to test.
- **Negative results worth reporting:** comment text ≈ 0; validation-fitted stacking < 0.
