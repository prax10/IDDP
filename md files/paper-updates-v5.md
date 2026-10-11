# Paper updates after Tasks H, I and J (v5)

This doc lists what to change in `paper-package.md` and `results-summary.md`, section by section. Each changed number points to the report file it comes from. Old documents are left as they were.

---

## Contribution and RQs (paper Part 1)

**New one-sentence statement:**

> We adapt Kula et al.'s dynamic delay-prediction framework to issue-level, multi-project open-source data. Re-forecasting an issue from what is observably happening to it (workflow moves, review, comments, reassignments, edits) beats a creation-time baseline from the very first checkpoint (AUC 0.803 vs 0.728 at 10% of expected duration, rising to 0.911 vs 0.739 at 100%). Kula's trajectory patterns and stall ratio add nothing once that activity is observed. The approach extends to a Scrum setting, predicting sprint spillover at AUC 0.75.

**Claims that change**

| Old claim | New claim | Evidence |
|---|---|---|
| "Dynamic overtakes static at M3" | Dynamic wins from **M1**, with the CI excluding zero at every checkpoint | `i4_dynamic_v4_vs_static.csv`; v5b per-checkpoint values in `j4_metrics_panel.csv` |
| Patterns "real but marginal" | Same verdict, now backed four ways: frequentist CIs (Phase 8), SHAP (#12 → near zero), Bayesian LOO (+20 ELPD in-sample), and zero test-AUC change (B1/B2, live model) | `j7_*`, `j6_live_vs_full_v5b.csv` |
| Text is the #1 static group | Still true for static. For dynamic, text falls to 0.034 summed \|SHAP\|, the 3rd smallest group | `j2_shap_groups_v5b.csv` |
| "Late advantage comes primarily from survival" | Early on, workflow status history leads. Late, elapsed time / position vs the label threshold dominates (0.42 → 0.99 summed \|SHAP\|) | `j2_shap_group_shift_v5b.csv` |
| Per-checkpoint ≈ pooled | Pooled ≥ per-checkpoint at all 10 checkpoints (equal at M1, +0.004 to +0.015 at M2–M10) | `h1_percheckpoint_all10_tuned.csv` |

**Add RQ5 (Scrum):** can the same framework predict whether a committed issue will finish within its sprint?

---

## Method (paper Parts 3–5): what to add

1. **5 training seeds.** XGBoost subsampling moves single-run AUC by up to about 0.02 at M10. All estimates are seed means; CIs are paired bootstraps of seed-mean differences (`h_common.paired_bootstrap_seeds`).
2. **Activity features at checkpoint time T.** Status-history counts and current workflow stage (8 buckets); time since the last status change, comment and any activity; comment counts; workload at T. Source: `task_i1_activity_features.py`.
   - **Fixed leak:** Phase 8's `current_status` falls back to the FINAL status for issues with no transitions. The new `status_bucket` uses the status before the first transition instead.
3. **Change-history features at T.** Reassignment, sprint membership and carry-overs, story points at T, field churn (priority, rank, links, fix version, estimate, label, component), and known-at-T track records.
   - Track records are smoothed rates over project, assignee and reporter issues already resolved before T, with m = 10 pseudo-issues.
   - Source: `task_j4_change_history_features.py` and `feature_engine.py`.
4. **Position vs the label threshold.** `log_elapsed_over_median` and `past_median` are knowable at T, because the threshold is the fixed train+val project median. Rows with `past_median = 1` (8.9% of rows) are positive by construction, so **report AUC on undetermined rows too**.
5. **Recency weighting.** Training rows are weighted by 0.5^(age / 4 years). The half-life was chosen on validation (`i2_recency_halflife_val.csv`).
6. **Thresholds.** Per-checkpoint "prior-match" cut-offs, set so the predicted-delayed share equals the historical delay share at that checkpoint (train+val only).
7. **Final model, v5b.** XGBoost, lr 0.05, depth 8, 43 trees, 94 features, 5-seed ensemble. A "live" variant drops `log1p_stall` and `pattern_label` (ΔAUC −0.00005) and powers the app.
8. **Scrum mode** (`task_j5_scrum_mode.py`):
   - Committed sprint = the first sprint the issue was in while that sprint was running.
   - Label = not resolved by the sprint's real close (min of planned end and actual completion) + 1 day.
   - Checkpoints at 0/25/50/75% of the planned sprint, placed strictly before the real close.
   - Chronological split by sprint start, 70/10/20.

---

## Results (paper Part 6): replacement tables

**6.2 Dynamic vs static: replace the table.** Test AUC, 5-seed mean, per checkpoint. Static from `h1_metrics_panel.csv`; v5b from `j4_metrics_panel.csv` (subset = all).

| Checkpoint | Static | Phase 8 dynamic (old) | **v5b dynamic** |
|---|---|---|---|
| M1 | 0.728 | 0.706 | **0.803** |
| M2 | 0.725 | 0.723 | **0.818** |
| M3 | 0.724 | 0.745 | **0.833** |
| M4 | 0.726 | 0.768 | **0.848** |
| M5 | 0.728 | 0.787 | **0.862** |
| M6 | 0.733 | 0.811 | **0.881** |
| M7 | 0.734 | 0.827 | **0.890** |
| M8 | 0.735 | 0.837 | **0.897** |
| M9 | 0.739 | 0.851 | **0.904** |
| M10 | 0.739 | 0.862 | **0.911** |

Old dynamic values are the 5-seed re-estimates from `h1_metrics_panel.csv` (pooled); they are within seed noise of the published single run.

**Overall (all checkpoints), v5b:**

| AUC | Accuracy @0.5 | Balanced accuracy @0.5 | Balanced accuracy, prior-match cut-offs | F1 | Brier | PR-AUC |
|---|---|---|---|---|---|---|
| 0.866 | 0.802 | 0.716 | 0.764 | 0.868 | 0.137 | 0.939 |

**Hard-subset AUC (add as a new table).** Seed-averaged ensemble predictions:

| Rows | Old Phase 8 dynamic | **v5b** |
|---|---|---|
| All | 0.798 | 0.866 |
| Outcome still open | 0.752 | 0.836 |
| Censoring-safe | 0.789 | 0.858 |
| **Censoring-safe and outcome still open** | **0.738** | **0.823** |

**Deployment-realistic (add):** with monthly retraining on outcomes known by each month (`j9_rolling_vs_frozen_v5b.csv`), AUC is **0.857** overall and 0.859 on censoring-safe rows.

**Scrum mode (new subsection):** test set of 16,744 rows, 4,851 issues, 747 sprints (`j5_scrum_metrics.csv`).

| Sprint progress | AUC | Accuracy | Majority accuracy | Balanced accuracy |
|---|---|---|---|---|
| All | **0.746** | 0.718 | 0.671 | 0.653 |
| Start (0%) | 0.721 | 0.677 | 0.602 | 0.649 |
| 25% | 0.727 | 0.697 | 0.645 | 0.650 |
| 50% | 0.736 | 0.728 | 0.693 | 0.649 |
| 75% | 0.764 | 0.789 | 0.775 | 0.629 |

Team-history baseline: AUC 0.590. Difference +0.156 [+0.148, +0.165].

**6.5 Generalisation: replace with the J3 results** (see `task-J-overnight-summary.md` §J3 and `j3_summary.txt`).

**6.7 SHAP: replace with `j2_shap_v5b.png`** and the group table in the overnight summary.

---

## Threats to validity (paper Part 7): what to add or change

- **Right-censoring (new, important).** Only resolved issues are modelled, so recently created long-running issues are missing.
  - 40% of test issues are exposed; delay rate 0.425 vs 0.501 for safe issues (`j1_exposure.csv`).
  - Most of Task H's "+0.13 from adding val data" lives in the exposed rows (M10: +0.24 vs +0.03).
  - All headline comparisons hold on censoring-safe rows (`j1_contrasts.csv`). Report both.
- **Creation-date split look-ahead (new).** The frozen model trains on validation issues that resolve during the test period. The strictly causal rolling protocol scores 0.009 lower overall (0.857 vs 0.866), and the gap sits entirely in censored rows.
- **Remove:** "tuned vs untuned confounds hyperparameters with training data" (resolved in H3: the effect is data, not tuning). Also remove "per-checkpoint tuned for only 3 of 10" (H1 covers all 10).
- **Keep and reword:** test-period separability (now explained largely by censoring), and the clustering subsample (still not run at full scale).
- **Final-value fields.** `Story_Point`, title and description length are final values in TAWOS. The time-correct `sp_at_T` is now included. Text-length features contribute about 0.003 AUC.

---

## Methodological narrative (paper Part 10): add these items

7. **Seed noise found and controlled.** Single runs varied by up to 0.02 AUC at M10, so everything was moved to 5-seed estimates.
8. **The retracted per-checkpoint advantage was re-explained.** It came from training-data differences (train vs train+val), not hyperparameters (H3).
9. **Censoring diagnosed.** The apparent value of the most recent training data was mostly an artifact of resolved-only sampling near the snapshot end (J1).
10. **Two Scrum-label leaks caught before reporting.** Carry-over events after early sprint completion, and pre-sprint planning moves counted as commitments. Logs: `j5_run_log_v1_*`, `j5_run_log_v2_*`.
11. **The Phase 8 status-fallback leak fixed** in the new activity features.
