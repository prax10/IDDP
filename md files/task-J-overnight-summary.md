# Task J — overnight run summary

Branch: `model-improvements` (`main` untouched). Every number below comes from a file in `reports/` (named in brackets). All point estimates are means over 5 training seeds, and all CIs are paired bootstraps over 1,000 test resamples, unless the line says otherwise.

## 1. Headline: how much better is the model now?

The pooled dynamic model on the same 262,433 test checkpoint rows:

| | Before (Task H baseline) | v4 (Task I) | **v5b (now)** |
|---|---|---|---|
| AUC, all checkpoints | 0.797 | 0.857 | **0.866** |
| AUC at M1 | 0.706 | 0.794 | **0.803** |
| AUC at M10 | 0.862 | 0.896 | **0.911** |
| Accuracy @0.5 | 0.730 | 0.794 | **0.802** |
| Balanced accuracy @0.5 | 0.568 | 0.697 | **0.716** |
| Brier (lower is better) | 0.169 | 0.143 | **0.137** |
| PR-AUC | 0.907 | 0.934 | **0.939** |

[`i2_metrics_panel.csv`, `j4_metrics_panel.csv`]

**With per-checkpoint cut-offs.** The app uses "prior-match" cut-offs: at each checkpoint, the cut-off is set so the share flagged as delayed matches the historical delay share. With these, v5b reaches accuracy 0.799, **balanced accuracy 0.764** and F1 0.856 (ensemble predictions). The baseline at 0.5 had balanced accuracy 0.568.

**The fairest test is the hard rows.** These are issues that are not yet past the project's typical duration (so the outcome is still open) and that are not exposed to the dataset-end censoring found in J1. Seed-averaged ensemble predictions, so slightly above the seed-mean figures [`j4_test_predictions.npz`, `i2_test_predictions.npz`]:

| Subset (rows) | Before | v4 | **v5b** | Gain |
|---|---|---|---|---|
| All (262k) | 0.798 | 0.858 | **0.866** | +0.069 |
| Outcome still open (228k) | 0.752 | 0.826 | **0.836** | +0.084 |
| Censoring-safe (159k) | 0.789 | 0.850 | **0.858** | +0.069 |
| **Censoring-safe and outcome still open (136k)** | **0.738** | 0.814 | **0.823** | **+0.086** |

**Static vs dynamic.** Dynamic now beats static from the very first checkpoint. At M1 static is 0.728 vs v5b 0.803, and the gap grows to +0.17 by M10 [`h1_metrics_panel.csv`, `i4_dynamic_v4_vs_static.csv`]. The old "crossover at M3" no longer holds. It was a property of the weaker feature set.

**Strictly causal deployment number: 0.857.** This comes from retraining every month on only the outcomes known at that point, J9 below. Quote it next to 0.866.

**Claims in the paper that changed tonight** (the exact edits are in `paper-updates-v5.md`):

| Topic | Before | Now |
|---|---|---|
| Crossover | M3 | **M1**. Dynamic wins everywhere, even against a static model given every creation-time feature (J10). |
| Cross-project | "beats within-project on 30/38" | **Not supported** once both arms use the same test set. Cross ≈ within; pooling all projects' history is best (29/38, dynamic) (J3). |
| Patterns | "real but marginal" | Still marginal. Kula's stall and pattern features add exactly nothing once activity is observed (J6), and the Bayesian LOO effect does not transfer to test (J7). |
| Text | top static group | Still true for static. For dynamic it is the 3rd-smallest group (J2). |
| New threats | — | Right-censoring at the dataset end (J1), and creation-date-split look-ahead (J9). Both are quantified, and the headline numbers hold on safe rows. |

## 2. What each stage found

### J1 — Right-censoring check [`j1_*`]
- TAWOS keeps only resolved issues. An issue created shortly before a project's data snapshot shows up only if it was resolved quickly.
- **40% of test issues are "at risk"**: created within the project's 90th-percentile resolution time of the snapshot end. Their delay rate is 0.425, against 0.501 for safe issues.
- **The Task H "+0.13 from adding validation data" is mostly this artifact.** At M10 the effect is +0.237 on at-risk rows but only +0.026 on safe rows.
- **The headline findings survive.** v4 vs static is +0.127 on safe rows and +0.132 on at-risk rows. Dynamic vs static and the activity-feature gains hold on censoring-safe issues.

### J4 — Change-history and "known at T" features → v5 [`j4_*`]
New features, all strictly before T:
- reassignment history;
- sprint membership and carry-overs;
- story points at T (the base `Story_Point` is the final value);
- field churn: priority, rank, links, fix version, estimate, label and component edits;
- known-at-T track records for project, assignee and reporter, using only outcomes already resolved by T.

| Variant | Gain vs v4 | 95% CI |
|---|---|---|
| v5a (the features above) | +0.0095 | [+0.009, +0.010] |
| **v5b (+ age relative to the label threshold)** | **+0.0147** | [+0.014, +0.015] |

- v5b adds `log_elapsed_over_median` and `past_median`. They are knowable at T: the threshold is the fixed project median from the label definition. On rows where the outcome is still open, the gain is +0.017.
- Deeper trees (depth 10/12) were tried and lost on validation [`j4_run_log_partial_depthgrid.txt`].

### J9 — Rolling monthly retraining [`j9_rolling_vs_frozen_v5b.csv`]
Each test month, the model is retrained on every issue whose outcome was known by the start of that month.

| Test rows | Rolling minus frozen | 95% CI |
|---|---|---|
| Safe | +0.002 | [+0.002, +0.003] |
| At risk | −0.020 | [−0.021, −0.019] |

- The frozen model is trained on train+val. It "knows" validation issues that only resolved during the test period, which is mild look-ahead under a creation-date split. That look-ahead helps exactly the censored rows.
- **Strictly causal AUC: 0.857 overall, 0.859 on safe rows.** This is the honest deployment figure.

### J6 — Live model and app [`j6_live_vs_full_v5b.csv`, `app/app_v5.py`]
- **Kula's stall and pattern features add nothing once activity is known.** Dropping `log1p_stall` and `pattern_label` changes AUC by −0.00005 [−0.0002, +0.0001].
- That is useful: the app can score any issue at any date, because every remaining feature is computed by `feature_engine.py` from raw events.
- `app/app_v5.py` is a new Streamlit app; `app/app.py` is untouched. It has four tabs:
  - **Portfolio:** pick a project and a date, and every open issue is scored with its top reasons.
  - **Issue detail:** risk history plus SHAP factors.
  - **Scrum mode:** running sprints and spill risk per issue.
  - **Model card.**
- Screenshots: `reports/j6_app_*.png`. Titles are blank in the screenshots only because `issue.csv` wasn't in the cloud workspace; on your Mac they show.
- Run with `streamlit run app/app_v5.py`. The first load takes about a minute.

### J5 — Scrum mode [`j5_*`]
- **Task:** predict whether an issue will be finished by the end of the sprint it was committed to.
- **Data:** 33,291 committed issues in 3,732 sprints, with checkpoints at 0/25/50/75% of the sprint.
- **Label:** not resolved by the sprint's real close + 1 day. Unresolved issues are included, so this label is not censored.

| Sprint progress | AUC | Accuracy | Majority-class accuracy |
|---|---|---|---|
| All checkpoints | **0.746** | 0.718 | 0.671 |
| Sprint start | 0.721 | | |
| 75% through | 0.764 | | |

- The team-history baseline scores 0.590 AUC; the gain over it is +0.156 [+0.148, +0.165].
- **Two leaks were found and fixed on the way. Both are logged and worth mentioning in a viva:**
  1. A third of sprints closed before their planned end, and Jira logs carry-overs at the close. The fix: checkpoints now sit before the *actual* close.
  2. Moving an issue to another sprint *before* the sprint started was being counted as committing it to the first sprint. The fix: the committed sprint is the first one the issue was in *while it was running*. The first version scored 0.752 partly through this leak; the honest version scores 0.746.

### J2 — SHAP on v5b [`j2_shap_*_v5b.*`]
Groups ranked by summed mean |SHAP|:

| Rank | Group | Summed mean \|SHAP\| |
|---|---|---|
| 1 | Elapsed time / position vs threshold | 0.66 |
| 2 | Workflow status history | 0.61 |
| 3 | People track record | 0.23 |
| 4 | Recency of activity | 0.17 |
| 5 | Links / field churn | 0.15 |
| … | | |
| | Text | 0.034 |
| | Stall + pattern (Kula) | 0.018 |

- **Early vs late:** status history and people track record matter most early. Elapsed time dominates late (0.42 → 0.99).
- **Text:** the largest group in the static model (Phase 10), but almost redundant once workflow activity is observed.

### J10 — Is the M1 win just "better features"? [`j10_static_plus_vs_v5b.csv`]
- **The test:** give the static model every feature computable at creation (track records known at creation, workload at creation, text length, story points at creation, sprint/assignee at creation). Re-select it on validation.
- **Static+ results:** issue-level AUC 0.753 (static 0.740). On the checkpoint rows it reaches only 0.739 at M1, against 0.728 for static.
- **v5b still wins at every checkpoint:** +0.064 [+0.059, +0.068] at M1, rising to +0.162 at M10. So the early-checkpoint advantage comes from **observing what happens to the issue**, not from the richer static information.

### J7 — Bayesian layer (Kula's approach, locked-plan items 8K/8L) [`j7_*`]
Fitted with PyMC NUTS on a 30k-row stratified sample (2 chains × 800 draws). Scored on all 262k test rows.

| Model | Test AUC | Brier | ECE (calibration error) | Mean 90% interval width |
|---|---|---|---|---|
| B1: hierarchical logistic, interpretable features + project intercepts | 0.832 | 0.147 | **0.014** | 0.064 |
| B1 + pattern | 0.832 | 0.147 | 0.015 | 0.071 |
| XGBoost trained on TRAIN only (raw) | 0.803 | 0.160 | 0.039 | — |
| B2: Bayesian recalibration of that XGBoost on VAL, with project intercepts | **0.854** | **0.139** | 0.020 | 0.033 |
| B2 + pattern | 0.853 | 0.140 | 0.020 | 0.036 |

- **Patterns, the Bayesian verdict.** PSIS-LOO on the fitting sample prefers the with-pattern models by about 20 ELPD (SE ≈ 7). Pattern evidence is real in-sample, and the "escalating" cluster_2 coefficient is positive with a 94% interval excluding zero. **But test AUC does not move** (+0.0002 / −0.0016). This is the same "real but marginal" conclusion as Phase 8, now with credible intervals.
- **The Bayesian layer's useful contribution is calibration and drift correction.** A simple interpretable hierarchical logistic model is the best calibrated (ECE 0.014) at AUC 0.83. Recalibrating a stale model with per-project intercepts learned from recent data adds +0.051 AUC. That matches J1/J9: projects drift, and recent outcomes matter.
- **Convergence:** max r-hat 1.01–1.03, minimum bulk ESS ≥ 100. Acceptable but not perfect; 4 chains would tighten it.
- **Coefficients worth quoting** (standardised; 94% intervals in `j7_B1_no_pattern_posterior.csv`):

| Effect | Coefficient |
|---|---|
| Assignee's prior delay rate | +0.48 |
| Log elapsed time | +0.51 |
| Log number of links | +0.33 |
| Being in review | −0.33 |
| Number of status transitions | −0.25 |
| Comments in the last 7 days | −0.16 |

### J3 — Time-respecting cross-project test (future-work #4) [`j3_lopo_static.csv`, `j3_lopo_dynamic_v4.csv`, `j3_summary.txt`]

Phase 9 had **two** asymmetries:
1. cross-project trained on all time periods;
2. cross-project was **scored on all of the held-out project's issues while within-project was scored on its last 20% only**, which are different test sets.

Now every arm is scored on the same test set: each project's chronologically last 20%. The time-respecting arms only train on issues created before that project's test cutoff. 38 projects.

| Training data | Static mean AUC | Dynamic (v4) mean AUC |
|---|---|---|
| Within: own project's first 80% | 0.657 | 0.756 |
| Cross: other projects, before the cutoff | 0.647 | 0.778 |
| **Pooled: all projects incl. own, before the cutoff** | **0.665** | **0.800** |
| Cross, all time (the original Phase 9 setup) | 0.648 | 0.771 |

| Comparison (paired, 38 projects) | Static | Dynamic |
|---|---|---|
| Cross (time-respecting) vs within | 20/38, −0.010, p = 0.98 | 19/38, +0.022, p = 0.26 |
| **Pooled vs within** | 25/38, +0.008, p = 0.09 | **29/38, +0.044, p = 7×10⁻⁷** |
| Cross all-time vs cross time-respecting | +0.0003, p = 0.30 | −0.007, p = 0.13 |

- **What changes:** the headline "cross-project beats within-project on 30/38 (0.710 vs 0.657)" **does not survive a common test set**. It came mainly from asymmetry (2). Time leakage (1) turns out to be negligible (+0.0003 static).
- **The corrected RQ3 answer:** a model trained only on other projects is about as good as one trained on the project's own history, with no significant difference. That is still a positive transfer result: no collapse on unseen projects.
- **The practical result:** the best option is pooling every project's past data, including the project's own. For the dynamic model this significantly beats own-history-only training on 29 of 38 projects.

## 3. Files
- **Scripts:**
  - `task_j1_censoring_check.py`
  - `task_j4_change_history_features.py`
  - `task_j4_v5_model.py`
  - `task_j5_scrum_mode.py`
  - `task_j6_live_model.py`
  - `task_j9_rolling_retrain.py`
  - `task_j2_shap_final.py`
  - `task_j7_bayesian_layer.py`
  - `task_j3_time_respecting_lopo.py`
  - `feature_engine.py`
  - `app_v5_backend.py`
  - `app/app_v5.py`
- **Models** (not in git, per `.gitignore`):
  - `models/xgb_dynamic_pooled_v5b_seed{0..4}.joblib` plus `_meta.json`
  - `models/xgb_dynamic_live_v5b_*`, used by the app
  - `models/xgb_scrum_spillover_*`
- **Feature tables:** `data/features/phase8_change_history_features.csv` and `data/features/scrum_rows.csv.gz`. Rebuild them on the Mac with `python task_j4_change_history_features.py` and `python task_j5_scrum_mode.py`.
