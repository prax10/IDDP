# Task I — Making the dynamic model better (results)

Run on 2026-10-08. Same chronological split, label (`delayed_v2`), test rows (262,433 checkpoint rows) and evaluation protocol as Task H: validation-selected configs, 5 training seeds, paired bootstrap CIs (1,000 test resamples).

Scripts:
- `task_i1_activity_features.py` builds the new features.
- `task_i2_model_upgrades.py` runs the model comparison.
- `task_i3_sanity_checks.py` runs ablations and robustness checks.
- `task_i4_finalize_model.py` saves the final model and runs the static comparison.

## Headline: yes, the model is significantly better

| Pooled dynamic model | AUC all | M1 | M3 | M5 | M10 | Brier (all) |
|---|---|---|---|---|---|---|
| Before (Task H baseline) | 0.797 | 0.706 | 0.745 | 0.787 | 0.862 | 0.169 |
| **After (v4: + activity features + recency weighting)** | **0.857** | **0.794** | **0.823** | **0.853** | **0.896** | **0.143** |
| Gain [95% CI] | **+0.060** [+0.059, +0.062] | +0.087 | +0.078 | +0.066 | +0.034 | −0.026 |

Accuracy and other scores, averaged over M1–M10:

| | Accuracy | Balanced accuracy | F1 |
|---|---|---|---|
| Before, threshold 0.5 | 0.743 | 0.562 | 0.840 |
| **After, threshold 0.5** | **0.804** | 0.687 | **0.868** |
| **After, per-checkpoint threshold ("prior match")** | 0.800 | **0.752** | 0.851 |

- Majority-class accuracy is 0.556 at M1 and 0.839 at M10. At M1 the new model reaches 0.711–0.720 accuracy (was 0.617).
- The per-checkpoint threshold sets the predicted-delayed share equal to the historical delay share at that checkpoint. It uses train+val only.
- That threshold fixes the old problem where the model said "delayed" for almost everything late on: balanced accuracy at M10 goes from 0.555 to 0.765.

## Static vs dynamic: the crossover moves from M3 to M1

The new dynamic model beats the static model at **every** checkpoint, with every CI excluding zero:
- M1: +0.065 [+0.060, +0.070]
- M5: +0.125
- M10: +0.157 [+0.148, +0.166]

Before, static won at M1 and the crossover was M3. **This changes headline finding 1.** Once you observe what is actually happening on an issue (status moves, review, comments), re-forecasting is worth it from the very first checkpoint.

## What was added (all measured strictly before the checkpoint moment T)

- **Status history:** number of transitions, reopens, backward moves, ever in progress, ever in review or testing, current workflow stage (8 buckets), and time since the last status change.
- **Comments:** count, distinct authors, comments in the last 7 days, and time since the last comment.
- **Activity:** time since the last activity of any kind.
- **Workload at T:** open issues in the project, issues the project resolved in the last 30 days, and open issues on the assignee.
- **Text stats:** title length, description length, and whether the description contains code or a stack trace.
- **Recency weighting:** older training issues count less, with a 4-year half-life chosen on validation. Worth +0.005 AUC [+0.005, +0.006].

## Which parts matter (ablation, test AUC, all checkpoints)

| Added to baseline | AUC |
|---|---|
| nothing | 0.797 |
| + status history | **0.848** |
| + last-activity time | 0.820 |
| + comments | 0.818 |
| + text stats | 0.800 |
| + workload at T | 0.799 |
| + all | 0.852 |

- By gain importance, the top features are time since last status change (30%), time since any activity (19%), ever in review (13%) and current stage (7%).
- Workload and text stats add almost nothing. They could be dropped for simplicity.

## Checks that this is real and not leakage

1. **Timing.** Every event counted is strictly before T, and the script asserts this. Checkpoint rows only exist while the issue is still unresolved, so the resolution itself is never visible.
2. **Fixed leak in old Phase 8 status.** Phase 8's `current_status` falls back to the issue's *final* status when it has no transitions. That would leak, so the new `status_bucket` uses the pre-first-transition status instead.
3. **Correlations.** The highest correlation of any new feature with the label is 0.325 (time since last status change). That is similar to the existing `assignee_prior_delay_rate` (0.328), and nowhere near the 0.95 leakage threshold.
4. **Not a recency artifact.** Trained on TRAIN only (no validation data), the activity features still add **+0.064 AUC** (0.739 → 0.803) and +0.056 at M10. So the gain is separate from the Task H finding that recent data matters.
5. **Data fix.** 14 comment timestamps used a 2-digit year ("14-02-05" = 2014-02-05) and were corrected.

## Model comparison

- XGBoost with activity features: 0.852.
- **+ recency weighting: 0.857 (chosen).**
- LightGBM with the same features: 0.846, so worse and not used.
- As in Task H, hyperparameters barely matter. The gain comes from the information, not the algorithm.

## Saved model

- `models/xgb_dynamic_pooled_v4_activity_seed{0..4}.joblib` is a 5-model ensemble: average the `predict_proba` outputs.
- `models/xgb_dynamic_pooled_v4_activity_meta.json` holds the feature list, config and per-checkpoint thresholds (0.61 at M1 → 0.67 at M10).
- The Streamlit app still uses v3. To switch, the app needs the activity features computed at "now".

## Caveats to state in the paper

- Text length and code flags come from the *final* description text in TAWOS. That text can be edited after creation, so these features are only approximately creation-time. They contribute ~0.003 AUC, so dropping them changes nothing material.
- The Task H recency caveat still applies to absolute AUC levels: test-period separability is higher than validation-period separability.
- Assignee is the final assignee. TAWOS's assignee-change history is not in the raw cache, because only STATUS changes were extracted.

## Next options

1. **Scrum mode.** `Sprint_ID` is set on only 9.5% of issues, so this would be a sub-population model. It needs the Sprint table exported from MySQL.
2. **Assignee-change history.** Extract `Change_Log` rows for the assignee field from the DB, for reassignment counts at T.
3. **Update the app** to v4 with live activity features.
4. **Re-run SHAP** on v4 for the explainability section. The story shifts from "elapsed time dominates" to "workflow activity dominates".
