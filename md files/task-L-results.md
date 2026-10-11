# Task L — Future-work items done properly

Branch `model-improvements`. Scripts `task_l*.py`, `l4_*.py`, `recommender.py`; reports `reports/l*`.
All numbers are on the untouched chronological test period (262,433 checkpoint rows, 34,398 issues)
unless stated. Nothing below was tuned on test.

| # | Item | One-line result |
|---|---|---|
| L1 | Full-scale DTW clustering | The "93% in one cluster" result was an artifact of complete-linkage clustering on a 4k sample. Full-population k-medoids gives balanced clusters. |
| L1b | Patterns re-tested | Balanced *shape* clusters do help the old feature set (+0.011 AUC at M3–M5), but add **nothing** on top of v5b activity features. |
| L2 | Regression ("how late") | SA 59% overall (54% at M1 → 63% at M10); beats project-median and creation-time regressors; calibrated 80% intervals. |
| L3 | Stronger baselines | XGBoost v5b stays best (AUC 0.866). MLP 0.864, GRU 0.863, logistic regression 0.856, RF 0.850, literature features 0.782. **The features matter more than the learner.** An untuned XGB+MLP+GRU average adds +0.005 (+0.013 at M1). |
| L4 | External dataset | __L4__ |
| L5 | Uncertainty | 90% conformal sets cover 92% on test; CQR intervals cover 84% / 93% (80% / 90% targets); sprint spill-count ranges cover 88–92% where a naive interval covers 69–77%. |
| L6 | "What would help" | Counterfactual re-scoring for 4 actions. In the test-period history every action points the same way, but status-move effects are overstated about 2×. Shown in the app with that caveat. |

---

## L1 — Full-scale DTW clustering (future-work #6)

**Why.** The paper's explanation for weak pattern transfer was that 93% of issues fell into one
cluster. That came from complete-linkage hierarchical clustering on a stratified 4,000-issue
sample (full pairwise DTW over 115k trajectories was called infeasible).

**What was done.**
- DTW (the dtaidistance definition) re-implemented in numba. It matches dtaidistance on 200 random
  pairs, and makes DTW to a handful of medoids over all 115,176 training trajectories cheap.
- **A.** The published 4 medoids, assigned to the full training population: 89.2 / 9.3 / 0.2 / 1.3%.
  So the published clustering is genuinely unbalanced, not just in the sample.
- **B.** Complete linkage re-run on 5 different samples (4k–12k): the largest cluster is always
  93–97%. **The imbalance comes from the method.** Complete linkage isolates a few extreme
  trajectories as their own clusters and leaves everything else in one.
- **C.** CLARA k-medoids (the k-medoids objective, scored on the *full* population). Two variants:
  - **level:** log1p stall as-is. Elbow at K=4, sizes **65 / 27 / 7 / 1%**.
  - **shape:** z-normalised, the trajectory's shape only. Elbow at K=3, sizes **76 / 12 / 12%**.

  The clusters differ in outcome:

  | variant | cluster | share | late rate |
  |---|---|---|---|
  | shape | rising stall (steady growth) | 76% | 71% |
  | shape | rise then reset (status moved) | 12% | 44% |
  | shape | flat / single point | 12% | 6% |
  | level | high, growing stall | 27% | 72% |
  | level | low stall | 65% | 54% |

**Correction for the paper.** Drop "93% single-cluster dominance" as the *explanation*. It is
true of the published clustering, but it is a property of complete-linkage clustering, not of
issue-level data. Full-scale k-medoids finds balanced clusters with clearly different delay
rates (6% vs 71%).

## L1b — Do the better clusters help? (pattern contribution re-tested)

Same protocol as Phase 8: pattern label added as a feature, 5 seeds, paired bootstrap.

| feature set | pattern variant | Δ AUC all | Δ M3 | Δ M5 | Δ M10 |
|---|---|---|---|---|---|
| baseline (Phase 8 features) | published medoids | −0.003 | 0.000 | −0.001 | −0.001 |
| baseline | full-scale level (K=4) | −0.002 | 0.000 | −0.002 | 0.000 |
| baseline | **full-scale shape (K=3)** | **+0.004** | **+0.012** | **+0.011** | 0.000 |
| v5b live features | any of the four | 0.000 | 0.000 | 0.000 | 0.000 |

(CIs in `reports/l1b_pattern_contribution.csv`; every v5b row is "equivalent, CI within ±0.005".)

**The new explanation (stronger than the old one).**
- Properly built patterns *do* carry signal. They help the Phase-8 feature set early on, which is
  the same direction as Kula et al.'s finding.
- But the v5b activity features (time since last status change, whether work started or reached
  review, comments, reassignments) already describe the same thing directly.
- So the pattern is redundant, not uninformative. That is a positive, testable claim, where the
  old one ("the clusters are degenerate") was a methodological artifact.

**Also fixed:** the published pipeline only assigned patterns from M3 onward (it needs 3 points).
Assigning from M1 with the available prefix changes nothing (rows 4–7 and 20–23).

## L2 — Regression: how late, not just whether (future-work #8)

- **Target:** `rrd = log2(resolution time / project median)`, with the train+val median, so
  `rrd > 0` is exactly the binary label (checked: 99.9% agreement). The predicted remaining days
  are `median × 2^rrd − elapsed`.
- **Model:** v5b live features, absolute-error XGBoost, config chosen on val, 3 seeds.
- **Metric:** MAE plus Standardised Accuracy, SA = 1 − MAE / MAE_random-guess (Shepperd & MacDonell), the metric Kula et al. report.

| scope | SA v5b | SA creation-time regressor | SA "project median" | median abs. error, remaining days | 80% interval coverage |
|---|---|---|---|---|---|
| all | **58.9%** | 49.8% | 35.5% | 19.9 | 80.1% |
| M1 | 53.9% | 42.6% | 39.0% | 16.4 | 81.9% |
| M5 | 59.7% | 51.6% | 36.5% | 20.0 | 80.0% |
| M10 | **63.2%** | 54.3% | 27.0% | 23.6 | 78.7% |

- The sign of the regression output alone ranks late vs on-time issues at AUC 0.846, close to
  the dedicated classifier (0.866).
- **Comparison caveat:** Kula et al. report SA 66–92% on *epic-level* BRE at one company. That
  is a different unit and target, so the numbers are not comparable; only the metric is shared.
- **Honest limit:** the median error of ~20 days against a median of 27 remaining days means
  "how late" is only roughly predictable. That is why the app shows a range rather than a date.

## L3 — Stronger baselines

Every model sees the same v5b live features (computed strictly before each checkpoint), is
trained on train+val, and is scored on the identical 262,433 test rows. The confidence
intervals are a paired bootstrap of the AUC difference against XGBoost (500 resamples).

| model | AUC | Δ vs XGB [95% CI] | AUC M1 | AUC M10 | balanced acc. | Brier | train time |
|---|---|---|---|---|---|---|---|
| **XGBoost v5b** (5 seeds) | **0.866** | — | 0.804 | **0.912** | 0.717 | 0.137 | — |
| MLP, PyTorch 2×256 (2 seeds) | 0.864 | −0.003 [−0.003, −0.002] | **0.812** | 0.904 | **0.735** | **0.135** | 2.4 min |
| GRU over the checkpoint sequence (2 seeds) | 0.863 | −0.004 [−0.004, −0.003] | 0.810 | 0.902 | 0.717 | 0.137 | 2.8 min |
| Logistic regression | 0.856 | −0.011 [−0.011, −0.010] | 0.811 | 0.888 | 0.733 | 0.138 | 25 s |
| Random forest (300 trees) | 0.850 | −0.017 [−0.017, −0.016] | 0.787 | 0.890 | 0.682 | 0.144 | 5.7 min |
| Literature feature set + RF (Choetkiertikul-style) | 0.782 | −0.085 [−0.086, −0.083] | 0.726 | 0.790 | 0.653 | 0.167 | 1.6 min |
| *XGB + MLP + GRU, equal-weight average* | *0.871* | *+0.005 [+0.004, +0.005]* | *0.817* | *0.912* | | | |

**What this shows.**
1. **XGBoost is the right choice, but by a small margin.** Logistic regression is only 0.011
   behind. Nearly all of the gain since Phase 8 (0.80 → 0.87) comes from the *features*
   (activity, workflow, known-at-T track records), not from model complexity. That is the
   honest headline, and it is robust.
2. **Literature feature set.** Our re-implementation of the classic delay-risk factors
   (discussion, priority and fix-version churn, links, workload, track record, elapsed time) loses
   0.085 AUC. The workflow-state and time-since-activity features are the new contribution.
   (Caveat: this is the *feature idea* on TAWOS, not their exact pipeline or data.)
3. **The sequence model does not beat the tabular model.** A GRU reading M1..Mt is no better than
   XGBoost on the snapshot at Mt, because the snapshot features already summarise the history
   (counts and times since events). This matches the L1b finding: the trajectory shape adds
   nothing once the activity features are in.
4. **Neural nets are slightly better early (M1).** They are better calibrated and generalise more
   smoothly with little evidence. An equal-weight average of XGB, MLP and GRU, with no weights fitted
   on anything, gains +0.005 overall and +0.013 at M1. It is a real but small gain; the app keeps
   the single XGBoost (simpler, and it has TreeSHAP explanations), and the ensemble is reported as an option.
5. Both neural nets overfit after one epoch (validation AUC peaks at epoch 1). The validation
   period is harder than test for every model, as noted since Phase 8.

## L4 — External dataset validation

__L4_DETAIL__

## L5 — Uncertainty with coverage guarantees

**Protocol:** models trained on TRAIN only, calibrated on VAL, coverage measured on TEST. The
test period is later, so drift is included.

**Flow mode: 90% Mondrian (class-conditional) conformal sets.** Each issue gets {late},
{on time}, or {late, on time} = "uncertain".

| scope | coverage | late covered | on time covered | confident calls | accuracy of confident calls |
|---|---|---|---|---|---|
| all | **92.1%** | 90.0% | 97.0% | 51% | **84.6%** |
| M1 | 88.1% | 79.1% | 99.4% | 44% | 72.7% |
| M5 | 92.4% | 90.9% | 96.4% | 50% | 84.9% |
| M10 | 95.8% | 96.9% | 89.9% | 67% | 93.8% |

- The model is honest about when it doesn't know: half the time it commits, and those calls are
  right 85% of the time (94% late in an issue's life).
- M1 under-covers late issues (79%). That is drift between the validation and test periods early
  in the lifecycle, and is reported as is.

**How late: conformalised quantile regression (CQR) on rrd.**

| target | coverage before conformal step | coverage after |
|---|---|---|
| 80% | 80.1% | **84.2%** |
| 90% | 90.8% | **92.7%** |

The intervals are wide (median 80% range for remaining days: 4–186). That is the honest
uncertainty of single-issue durations.

**Scrum mode: 90% interval on how many committed issues will spill.** Split-conformal on
validation-sprint residuals, by sprint progress.

| progress | coverage (conformal) | coverage (naive Poisson-binomial) | median width (issues) | test sprints |
|---|---|---|---|---|
| 0% | **92.0%** | 70.7% | 8.6 | 475 |
| 25% | 88.4% | 69.3% | 7.5 | 450 |
| 50% | 91.0% | 71.2% | 7.1 | 420 |
| 75% | 92.0% | 76.8% | 6.5 | 401 |

The naive interval treats issues as independent and badly under-covers. Spills within a sprint
are correlated (a bad sprint goes badly for many issues at once), and the conformal interval
accounts for that from data.

**Does the calibration survive deployment?** (`task_l5b_deployed_check.py`) The app's models
are refit on train+val, and the constants above were calibrated with train-only copies.
Re-measured exactly as the app computes them:
- **Flow sets:** 92.5% coverage (late 90.1%, on time 98.0%), essentially unchanged.
- **Scrum ranges:** 96–98% coverage. That is above the 90% target, i.e. conservative: rounding
  to whole issues widens the range, and the refit model fits better. The median width is 5–7
  issues, and the point estimate is off by 1.2–1.6 issues on average.

**A real "90s" number from this (honest).** Among issues *not yet overdue* (overdue ones are late
by definition, so they are excluded):

| scope | share flagged "confident: late" | precision of those flags | share of eventually-late issues caught | base late rate |
|---|---|---|---|---|
| all | 18.7% | **96.4%** | 27.6% | 65.5% |
| M1 | 4.3% | **97.4%** | 7.6% | 55.5% |
| M5 | 24.7% | **96.9%** | 34.5% | 69.3% |
| M10 | 37.9% | **95.2%** | 48.1% | 75.1% |

When the model says it is *confident* an issue will be late, it is right 95–97% of the time at
every checkpoint. The trade-off is coverage: it is that sure about a quarter of open issues. The
"confident: on time" calls are 71–74% precise, so the asymmetry is real. Lateness is easier to
be sure of than punctuality. Source: `reports/l5b_confident_calls_not_overdue.csv`.

**In the app.**
- Flow portfolio: a "confidence" column (confident late / confident on time / uncertain), a
  "likely done in" 80% range, and a KPI counting confident late calls.
- Issue detail: the same values as metrics.
- Scrum board: "Predicted to spill 4.3, 90% range 2–8".

## L6 — "What would help?"

`recommender.py`, in the app's flow-mode Issue detail under "What would help?". For one issue
at moment T, each action is applied to the issue's feature row and re-scored with the live
model:

| action | what changes in the feature row |
|---|---|
| start work | status → in progress, work started, status just moved |
| get it into review | status → review, review reached, status just moved |
| unblock / discuss | a comment now (counts +1, last activity = now) |
| reassign to colleague X | X's *known-at-T* track record, experience and open workload replace the current assignee's; candidates are people with ≥ 5 resolved issues in the project who were active in the last 180 days |

**These are model estimates, not causal effects.** So before showing them, they were checked
against what actually happened (`task_l6_recommendations.py`, `task_l6b_strict_check.py`).
- **Sample:** 20,000 test rows at M2–M6, where a PM can still act.
- **Comparison:** did issues that *actually* took the action in the next 7 days end up late less
  often than issues at the same predicted-risk decile that did not?
- **Strict version:** only issues still open 7 days later, which removes "it reached review
  because it was finishing anyway".

| action | model estimate (mean risk change) | observed, all | observed, strict | direction agrees |
|---|---|---|---|---|
| start work | −0.22 | −0.29 | **−0.15** | ✓ |
| get into review | −0.31 | −0.34 | **−0.16** | ✓ |
| discuss / comment | −0.04 | −0.31 | **−0.12** | ✓ |
| reassign (best colleague) | −0.08 | −0.13 | **−0.08** | ✓ |

**Reading.**
- Every recommendation points the right way in real outcomes.
- The size of the effect is right for reassignment, about 2× too large for status moves, and too
  small for commenting.
- Even the strict observational numbers are still confounded (whoever picks an issue up may pick
  the easy ones), so neither column is a causal estimate. The app says so in one line under the
  table.
- A randomised trial or a natural experiment would be needed for causal claims. That is future work.

This also replaces the old Task G "reassignment" demo logic: the recommender now lists all four
levers and *declines* (shows "little effect") when none moves the risk by ≥ 0.02.
