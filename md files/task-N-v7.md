# Task N: v7, an honest push on overall AUC (and the leak it uncovered)

Branch `v7-honest`. Scripts `task_n1_v7_features.py` … `task_n5_report.py`. Reports `reports/n2_val_ablation.csv`,
`n4_rolling_retune_v7.csv`, `n5_results.csv`, `n5_by_checkpoint.csv`, `n5_vs_v6.csv`, `n3_clearcut_v7.csv`.
Test period scored once, after all choices were made on train/validation. 5 seeds, paired bootstrap (500).

## Headline

**v7 test AUC 0.867** (5-seed ensemble 0.868). v6 was 0.872, but about 0.007 of that came from features that
describe an issue's *final* state rather than its state at the checkpoint. v7 removes those, adds three new
known-at-T feature groups, and recovers part of the gap (+0.002). 0.90 overall is not reachable honestly.

| Test (scored once) | v6 (as deployed) | v6 cleaned | **v7** |
|---|---|---|---|
| AUC overall (seed mean) | 0.872 | 0.865 | **0.867** |
| AUC overall (5-seed ensemble) | 0.873 | 0.866 | **0.868** |
| Early warning, M1–M3 | 0.828 | 0.817 | **0.821** |
| Late stage, M5–M10 | 0.895 | 0.890 | **0.891** |
| Final stretch, M8–M10 | 0.908 | 0.904 | **0.904** |
| M1 / M10 | 0.814 / 0.915 | 0.801 / 0.911 | **0.806 / 0.911** |
| Outcome still open | 0.843 | 0.834 | **0.837** |
| Censoring-safe + outcome open | 0.830 | 0.822 | **0.826** |
| Top-10% flag precision, outcome open | 0.986 | 0.981 | **0.982** |
| Top-10% flag precision at M1 | 0.956 | 0.923 | **0.940** |
| Accuracy / balanced accuracy @0.5 | 0.810 / 0.743 | 0.807 / 0.734 | **0.808 / 0.735** |

Paired bootstrap: v6 cleaned − v6 = −0.0069 [−0.0074, −0.0063]; v7 − v6 = −0.0050 [−0.0055, −0.0044];
**v7 − v6 cleaned = +0.0019 [+0.0017, +0.0022]**. v6 reproduced exactly (0.8719) before anything changed.

## 1. The leak: final-state features

| Feature group (in v6) | Problem | Fix in v7 |
|---|---|---|
| `assignee_prior_delay_rate`, `assignee_prior_resolved_count`, `assignee_concurrent_open`, `is_unassigned`, `assignee_open_at_T`, `assignee_rate_known`, `assignee_n_known` | Use the issue's **final** assignee. Rebuilt from the assignee change log, the assignee at T differs from the final one on **46%** of checkpoint rows, and **37%** of rows are unassigned at T (15% using the final assignee). The reconstruction agrees with the final assignee on 94% of issues at their last event, so it is sound. | Dropped; replaced by `is_unassigned_T`, `assignee_rate_known_T`, `assignee_n_known_T` computed on the assignee at T |
| `n_links`, `n_blocking_links` | Count the **final** link list. Long-running issues accumulate links, so the count partly encodes duration. | Dropped; replaced by links at T (below) |
| `Story_Point` | Final value (the time-correct `sp_at_T` was already in) | Dropped (no measurable effect) |

Validation ablation (fit on train, score on validation, 1 seed):

| Variant | Val AUC | M1 | M10 |
|---|---|---|---|
| v6 | 0.782 | 0.749 | 0.792 |
| − final-assignee features (+ assignee at T) | 0.773 | 0.737 | 0.783 |
| − final link counts | 0.768 | 0.730 | 0.777 |
| − final story points (= v6 cleaned) | 0.767 | 0.728 | 0.778 |
| cleaned + links at T | 0.768 | 0.729 | 0.778 |
| cleaned + fix-version progress | 0.769 | 0.730 | 0.777 |
| cleaned + recent assignee form | 0.768 | 0.730 | 0.775 |
| **v7 = cleaned + all three** | **0.770** | **0.733** | 0.775 |
| v6 + all three, not cleaned (info only) | 0.786 | 0.755 | 0.795 |

The leak costs more on validation (−0.015) than on test (−0.007). Paper line: *"An audit found that seven
assignee features and two link counts were computed from an issue's final state; rebuilding them as of each
checkpoint lowers test AUC from 0.872 to 0.865."*

## 2. New features (all strictly before T; leakage test passes, max |corr| 0.275)

- **Links at T** (8 features) from 448k Link change-log events. Jira keys mapped to issue IDs via Issue_Link
  (91,406 keys; 45% of events resolved to a target), so a blocker's own status at T is known: open blockers,
  blockers resolved in the last 7 days, time since the last link change. 25% of rows have a link at T; 1.9%
  have an open blocker.
- **Fix version at T** (8 features), the closest thing to a deadline in TAWOS: the version assigned at T, how
  many of its issues were assigned and done before T, its age, resolutions in the last 14 days, and a
  velocity-based "days left" estimate (open issues ÷ recent resolution rate). 24% of rows have a fix version.
  TAWOS release dates were deliberately not used: a version's release date is set when it ships, which is
  after T.
- **Recent assignee form** (5 features) on the assignee at T: late rate over 90 days, resolutions in 30 / 90
  days, late share of their last 5, days since their last resolution.

Each group adds about +0.001 to +0.002 on its own. Together +0.0025 on validation and +0.0019 on test.

## 3. Re-tune (rolling origin, 3 windows of train+val; test untouched)

| Config | Best trees | Mean window AUC |
|---|---|---|
| depth 8, min_child_weight 50 (v6 config) | 154 | 0.8192 |
| depth 10, min_child_weight 50 | 155 | 0.8194 |
| **depth 8, min_child_weight 50, colsample 0.6** | **130** | **0.8202** |

Flat again: the model is information-limited, not tuning-limited.

## 4. Secondary result: clear-cut delays (not a replacement headline)

Late = more than 1.5× the project median, on time = under median ÷ 1.5, everything in between dropped (test
keeps 207k of 262k rows). Same v7 features and config.

| Scope | Rows | AUC | M1 | M5 | M10 |
|---|---|---|---|---|---|
| All clear-cut rows | 207,214 | **0.924** | 0.855 | 0.925 | 0.981 |
| Undecided rows only (elapsed < median ÷ 1.5) | 151,413 | **0.881** | 0.848 | 0.873 | 0.912 |

Read it carefully: an open issue older than median ÷ 1.5 can only be clearly late (a clearly-on-time issue
would already be resolved), so part of the 0.924 is decided by construction. **0.881 is the honest
clear-cut number.** Paper line: *"Separating clearly late from clearly on-time issues, with borderline issues
excluded, reaches AUC 0.92 (0.88 on rows whose outcome is not yet decided)."*

## 5. What changes elsewhere

- **Paper / viva:** use v7 numbers. Add the final-state audit to the methodological narrative (it is a
  strong viva answer: "how did you know the assignee at that time?" now has a real answer).
- **Secondary results** (baselines, regression, external validation, Scrum, conformal, recommendations) were
  run on v5b/v6 features, which include the final-assignee features. Expect small downward revisions if
  re-run on the v7 feature set. The external portable model also used them.
- **App:** still loads v6. Switching to v7 needs the app's feature engine to compute links-at-T, fix-version
  and assignee-at-T features for "now". Models saved as `models/xgb_v7_seed{0..4}.json` + `xgb_v7_meta.json`.
- **Data:** `phase8_activity_features.csv`, `phase8_change_history_features.csv` and `phase8_v7_features.csv`
  are not on the Mac (too large to copy back). Regenerate with `task_i1_activity_features.py`,
  `task_j4_change_history_features.py` and `task_n1_v7_features.py` (under a minute each).
