# Task: Fair three-way comparison + threshold tuning + Phase 10 SHAP

Three parts, all cheap (no retraining except optional). Do them in order.

## Part A — Fair static-vs-dynamic comparison (fixes the headline chart)

**Problem:** Phase 7's static AUC (0.724) is measured on all 40,658 test
issues (~50% delayed), while Phase 8's per-checkpoint AUCs are measured on
shrinking, increasingly delay-skewed surviving populations (57.7% → 87.8%
delayed). Comparing them directly is apples-to-oranges — the current
"static beats dynamic until M7" reading may be a population artifact.

**Fix:** load `models/xgb_static_baseline.joblib` and score it on **each
per-checkpoint test population** — i.e. for checkpoint M_i, take exactly
the test issues evaluated at M_i in `reports/phase8_dynamic_metrics.csv`,
and predict on their Phase 6 static features (creation-time only; the
static model has no stall/elapsed inputs by design). Report AUC and F1 per
checkpoint.

This gives three comparable lines on identical populations:
1. Static baseline (creation-time info only)
2. Dynamic, no pattern
3. Dynamic, with pattern

Add these columns to `reports/phase8_dynamic_metrics.csv` and regenerate
`reports/phase8_checkpoint_accuracy.png` with all three lines plus the
majority-baseline reference. **Use AUC as the primary panel** — F1 is
distorted by the shifting base rate, as already established.

State plainly in the report which model wins at which checkpoints, now
that the populations match.

## Part B — Threshold sensitivity for Phase 7 (quick, fixes bad optics)

Phase 7's F1 at the default 0.5 threshold (0.641) is below the
majority-class baseline (0.656) — but that's a threshold artifact, not a
ranking failure (AUC 0.724). Compute Phase 7's precision/recall/F1 at the
threshold that maximizes F1 on the **training** set (not the test set —
picking a threshold on test would be leakage), then report test-set
metrics at that tuned threshold alongside the 0.5-threshold numbers.
Add to `reports/phase7_metrics.json`.

## Part C — Phase 10 SHAP (do this even if time is short)

Using `models/xgb_static_baseline.joblib` and the dynamic with-pattern
model:

1. **Global (static model):** `shap.TreeExplainer` → summary plot over a
   sample of the test set (5,000-10,000 rows is plenty, don't run all
   40k). Save to `reports/phase10_shap_static_summary.png`.
2. **Global (dynamic with-pattern model):** same, on a sample of test
   checkpoint rows. Save to `reports/phase10_shap_dynamic_summary.png`.
   Specifically report where `pattern_label`, `log1p_stall`, and
   `elapsed_minutes` rank among feature importances — this is the direct
   evidence for whether the discovered patterns contribute anything.
3. **Importance shift across checkpoints (high presentation value):**
   compute mean |SHAP| per feature separately for early checkpoints
   (M1-M3) vs. late (M8-M10) on the dynamic model, and report which
   features gain/lose importance as issues progress. Save to
   `reports/phase10_shap_importance_shift.png`. Expected story to verify
   (or refute — report what's actually there): static features matter
   early, progress features matter later.
4. **Local example:** one waterfall plot for a single correctly-predicted
   delayed test issue, to `reports/phase10_shap_local_example.png`.

## Report back

Part A's three-way per-checkpoint table and which model actually wins
where. Part B's tuned-threshold numbers. Part C's top-10 features for
both models, where `pattern_label` ranks, and the early-vs-late
importance shift.
