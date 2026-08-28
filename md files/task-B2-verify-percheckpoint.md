# Task B2 — Verify the per-checkpoint result before it becomes the headline

Short task. Three checks, then a decision. Do not skip check 2 — the
+0.147 AUC gain at M10 is large enough that it needs a mechanism.

## Check 1 — Full per-checkpoint table (needed to locate the new crossover)

Report the complete M1-M10 comparison, all three lines on identical
per-checkpoint test populations:
- Static (v2, with text)
- Pooled dynamic (with pattern)
- Per-checkpoint dynamic (with pattern)

Plus N per checkpoint and the majority-class baseline.

**Specifically: identify the exact checkpoint where per-checkpoint dynamic
first overtakes static.** With pooled models it was M6. From the partial
numbers already reported (M1: static 0.719 vs 0.706; M5: static 0.709 vs
0.763), it now appears to be somewhere in M2-M4. Pin it down.

## Check 2 — Why do per-checkpoint models win? (the important one)

Hypothesis to test: at checkpoint M_i, `elapsed_minutes` is exactly
`(i/10) × expected_duration_proxy`, so within a single checkpoint it is a
constant multiple of `expected_duration_proxy`. A per-checkpoint model may
therefore be exploiting the checkpoint definition — effectively learning
the ratio between an issue's group-level expected duration and its
project's median — rather than learning richer stage-specific dynamics.

Retrain the per-checkpoint models **without `elapsed_minutes`** (keep
`log1p_stall`, `pattern_label`, static features, text) and report
per-checkpoint AUC alongside the with-`elapsed_minutes` numbers.

Interpretation to report:
- If AUC collapses toward the pooled model's level (especially at late
  checkpoints), the per-checkpoint advantage is largely the
  checkpoint-definition mechanism. Still a legitimate result, but it must
  be described that way — not as "the model learns better stage-specific
  dynamics."
- If AUC stays high without `elapsed_minutes`, the specialization is
  genuine and the per-checkpoint design is learning real stage-specific
  structure. Say so.

Either outcome is publishable; the point is to know which one is true.

## Check 3 — Label-definition sensitivity (5 minutes, closes a methodological gap)

The label uses `Resolution_Time_Minutes > per-project median`, where the
median was computed via `groupby('Project_ID').transform('median')` over
the **entire** resolved population — including test issues. For a
forecasting setup, the strictly correct version computes each project's
median from **training issues only**.

Don't relabel and re-run everything. Just quantify the exposure:
1. Compute each project's median from training-split issues only.
2. Recompute test-set labels using those train-only medians.
3. Report **what percentage of test labels change.**

If it's under ~2%, state that in the limitations as a quantified,
negligible effect. If it's materially higher, flag it — that would be
worth a re-run.

## Report back

The full three-line M1-M10 table with the exact crossover checkpoint; the
per-checkpoint with/without-`elapsed_minutes` comparison and which
hypothesis it supports; and the percentage of test labels affected by the
train-only median definition.
