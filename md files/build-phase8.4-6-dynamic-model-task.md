# Task: Phase 8.4-8.6 — Dynamic pattern assignment, dynamic model, milestone evaluation

**Stop after Step 7 and report back before touching Phase 9/10.**

Inputs: `data/features/phase8_checkpoints.csv` (all issues, not just train),
`data/features/phase8_medoids.csv` (the 4 medoids from 8.3, already in
log1p space, computed only on the 4,000-issue training subsample),
`data/features/phase8_split.csv`, `data/features/phase6_feature_table.csv`.

## Step 1 — Apply log1p everywhere

Apply `log1p(stall_ratio)` to **every** row of `phase8_checkpoints.csv`
(train, validation, and test alike) — this is a fixed mathematical
transform with no fitted parameters, so unlike `expected_duration_proxy`
it's safe to apply universally with no leakage risk.

## Step 2 — Dynamic pattern assignment (train + validation + test)

For every issue and every checkpoint *t* from M3 onward: take that issue's
**partial trajectory** — its log1p(stall_ratio) values from M1 through Mt
only, even if the table already contains later checkpoints for that issue
— and compute DTW distance to each of the 4 saved medoids (compare the
partial trajectory directly against the full medoid; DTW handles the
length mismatch via alignment, no truncation needed). Assign
`pattern_label = argmin(distance)`. For M1-M2, set
`pattern_label = "insufficient_history"` (fixed, no computation).

**This is what makes it "dynamic":** the same issue can have a different
`pattern_label` at M5 than at M8, and the label at M5 must be computable
using only M1-M5 data — never M6 onward, even though it exists in the
table for issues that progressed further.

## Step 3 — Required verification (do not skip)

- Confirm the 4 medoids are exactly those saved from 8.3 (the 4,000-issue
  training subsample) — validation/test data must never have touched them.
- Spot-check 2-3 specific issues: manually confirm their `pattern_label`
  at some checkpoint *t* was computed using only checkpoints 1..t.
- Report how often `pattern_label` changes across an issue's own
  checkpoints, and roughly what checkpoint it tends to stabilize by (quick
  descriptive stat, not a full experiment — this is a nice bonus for the
  paper, not a blocker).

## Step 4 — Train two dynamic models (the required ablation)

Both trained on **training-split checkpoint rows**, using Phase 6's static
per-issue features plus per-checkpoint stall_ratio and elapsed time:
- **(a) Dynamic XGBoost, no pattern:** same features, no `pattern_label`.
- **(b) Dynamic XGBoost, with pattern:** same features plus `pattern_label`
  (categorical, including the `insufficient_history` value for M1-M2 rows).

Same hyperparameters as Phase 7 unless there's a clear reason to change
them.

## Step 5 — Milestone-wise evaluation (test split only)

For each checkpoint M1 through M10: evaluate both models (a) and (b) on
**only the test-split issues that have a snapshot at that specific
checkpoint** (the evaluated population shrinks as checkpoints increase,
since issues resolve and drop out — that's expected, not a bug). Report
AUC, F1, precision, recall, **and N (issue count at that checkpoint)** —
don't report a metric without its N, since the population composition
changes checkpoint to checkpoint.

If it's cheap to add: also evaluate the Phase 7 static baseline model's
predictions on the same per-checkpoint test populations, as a third
reference line. Skip this if it adds meaningful time — it's a nice-to-have
comparison, not required.

## Step 6 — Save

- `reports/phase8_checkpoint_accuracy.png`: AUC (or F1) vs. checkpoint
  index, one line per model variant (with-pattern, without-pattern, and
  static baseline if included). This is likely your best single chart for
  the presentation — make it clean and readable.
- `reports/phase8_dynamic_metrics.csv` or `.json`: the full per-checkpoint
  metrics table, including N.

## Step 7 — Report back

Does `pattern_label` help — at which checkpoints, by how much? The
leakage-verification results from Step 3. The checkpoint-accuracy plot.
Anything that looks off (e.g. a metric that's suspiciously flat, or N
dropping to near-zero at late checkpoints making those numbers unreliable
— flag it rather than reporting a metric with N=12 as if it were solid).
