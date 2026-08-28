# Task B4 — Validation-based tuning (the skipped Part 0), then settle the headline

This is the last outstanding methodological gap. It sits directly under
the current headline claim, so it needs closing before that claim is
final.

**Why:** the pooled dynamic model currently runs Phase 7's hyperparameters
(lr=0.05, 400 rounds) on 943,095 checkpoint rows — ~5× the data Phase 7
was tuned for, and far more heterogeneous. There is direct evidence it is
undertuned: in the earlier v1 run at lr=0.1 / 300 rounds it reached 0.780
AUC at M10, versus 0.734-0.739 at lr=0.05 / 400. So the current finding
("pooled never beats static") may be partly an artifact of settings
borrowed from a different problem.

Use the **validation split** (issues between the 70th and 80th percentile
of `Creation_Date` in `data/features/phase8_split.csv`) — currently
unused. Never tune on test.

## Step 1 — Tune all three configurations independently

For each of:
- (a) static model
- (b) pooled dynamic (with pattern)
- (c) per-checkpoint dynamic (with pattern) — tune once on pooled
  validation data and apply the chosen config to all ten models, rather
  than tuning ten times; note this simplification in the report

run a small grid — 9-12 configs, not exhaustive:

```
learning_rate: [0.05, 0.1, 0.2]
max_depth:     [4, 6, 8]
n_estimators:  [300, 600, 1000]   with early stopping on validation
subsample / colsample_bytree: fixed at 0.8
```

Select by **validation AUC**. Then retrain each on train+validation with
its selected config and evaluate once on test.

Report for each: selected config, validation AUC, test AUC.

## Step 2 — Regenerate the three-line comparison

Rebuild the per-checkpoint table (static / pooled / per-checkpoint) on
identical test populations with all three tuned, using the v3 (train+val
median) labels and v2 (with-text) features.

Report per checkpoint: N, %delayed, AUC for all three lines, and
**majority-class accuracy** (not majority F1 — with the majority class now
"not delayed," all-zeros prediction yields F1 = 0.000, which is
mathematically correct but meaningless as a baseline; use accuracy, or a
stratified-random baseline).

## Step 3 — Settle the headline question

State plainly:

1. **Does the tuned pooled model beat static at any checkpoint?** If yes,
   at which, and by how much. If no, the "pooling destroys the dynamic
   benefit" finding survives proper tuning and can be stated strongly.
2. **Does per-checkpoint still beat pooled** once both are tuned, and by
   how much?
3. **Where is the per-checkpoint crossover** against static under tuned
   settings (M3 under both previous labelings)?

Save `reports/phase8_final_tuned_comparison.csv` and
`reports/phase8_final_tuned_comparison.png`.

## Step 4 — Add a tuned row to the robustness table

Extend `reports/label_definition_robustness.csv` with a third column for
the tuned results, so the paper can show that conclusions hold across
both label definitions *and* under proper per-model tuning.

## Report back

Selected configs and validation AUCs for all three; the final tuned
three-line table; direct answers to the three questions in Step 3; and
whether any previously-stated conclusion changes as a result.
