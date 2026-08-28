# Task: Phase 9 cross-project generalization + one verification check

## Part A — Verify the `elapsed_minutes` interpretation (quick, 5 min, do first)

Claim to test: `elapsed_minutes` is not an independent "progress" signal —
by construction it equals `(i/10) × expected_duration_proxy`, so within a
single checkpoint it should be near-perfectly correlated with
`expected_duration_proxy`, and across the pooled data it mainly encodes
which checkpoint a row came from.

Check and report:
1. Within checkpoint M5 (and M8) separately, the Pearson correlation
   between `elapsed_minutes` and `expected_duration_proxy` on test rows.
   Expected ≈ 1.0 — confirm or refute.
2. Correlation between `elapsed_minutes` and the checkpoint index across
   all pooled rows.
3. If (1) confirms ≈1.0: retrain the dynamic with-pattern model **without**
   `elapsed_minutes` (keeping `log1p_stall`, `pattern_label`, and the
   static features) and report per-checkpoint AUC. This isolates how much
   of the dynamic model's performance came from genuine issue-specific
   progress signal vs. the checkpoint/group-duration encoding. Report it
   as an extra line in the metrics table — this is a strong ablation for
   the paper, not a correction.

## Part B — Cross-project generalization (Phase 9 core)

**Scope: 10 projects, not all 39** (full 39-project LOPO is the stretch
version; 10 is defensible and much faster). Choose them to span the size
range — include the small ones flagged in EDA (projects 10, 6, 9, 2, 35 —
all under ~700 resolved issues) plus ~5 large ones, so small-project noise
is visible rather than hidden.

For each chosen project P:
1. Train the **static** model (Phase 7 hyperparameters: n_estimators=400,
   max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
   on all resolved issues from the other 38 projects.
2. Test on project P's resolved issues.
3. Record: project ID, N (test issues), delay rate in P, AUC, F1.

**Important — keep the label definition honest:** `delayed` is defined
relative to each issue's *own* project median, which is already computed
per-project in the Phase 6 table. Do not recompute the median using the
held-out project's test data in any way that the training projects
wouldn't have had access to — just use the existing label column.

Also compute, for comparison, **within-project** performance for those
same 10 projects: train and test on the same project using its own
chronological 80/20 split (skip any project with too few issues for this
to be meaningful — report which were skipped and why).

## Part C — Report

- Per-project table: ID, N, delay rate, cross-project AUC/F1,
  within-project AUC/F1, and the gap between them.
- Macro-mean AND median/IQR across the 10 projects — do not report only
  the mean, since large projects would dominate it.
- Explicitly flag which results come from small-N projects and are
  therefore noisy.
- Save: `reports/phase9_generalization.csv` and a plot
  (`reports/phase9_generalization.png`) showing per-project AUC with N
  annotated, cross-project vs. within-project.

## Report back

The per-project table, the macro-mean/median/IQR, the size of the
generalization gap, and Part A's correlation results + the
no-`elapsed_minutes` ablation AUCs.
