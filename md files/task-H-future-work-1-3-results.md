# Task H — Future-work items #1–#3 (results)

Run on 2026-10-08. Library versions match the project venv (xgboost 3.4.1, scikit-learn 1.9.0, pandas 3.0.5). Same data files, splits, labels (`delayed_v2`) and feature lists as B4/B5.

Scripts: `h_common.py`, `task_h1_percheckpoint_tuning.py`, `task_h2_text_pca_sweep.py`, `task_h3_confound.py` (`run_all.sh` runs all three). Outputs are the `reports/h1_*`, `reports/h2_*` and `reports/h3_*` files.

---

## 0. Seed noise (found first)

- Loading the saved `xgb_dynamic_pooled_v3_tuned.joblib` reproduces the published per-checkpoint AUCs **exactly** (M10 = 0.8586), so the data is identical.
- Retraining the **same config** with different random seeds gives M10 AUCs between **0.845 and 0.865** (sd ≈ 0.004). The cause is XGBoost's row and column subsampling (`subsample=0.8`, `colsample_bytree=0.8`).
- The mean over 5 seeds is **0.858**, so the published numbers are representative and not a lucky draw.
- However, single-run differences under ~0.01 (like the old "max gap 0.0098" equivalence claim) are within seed noise.
- **From here on, every model is trained with 5 seeds.** Point estimates are seed means. CIs are paired bootstraps (1,000 test resamples) of the seed-mean AUC difference.

---

## #3 — Tuned vs untuned confound: FIXED (2×2 factorial)

Both configs were trained on both training sets and scored on the identical 262,433 test rows.

| Test AUC (5-seed mean) | untuned / train | untuned / train+val | tuned / train | tuned / train+val |
|---|---|---|---|---|
| all checkpoints | 0.740 | 0.792 | 0.734 | **0.795** |
| M1 | 0.683 | 0.714 | 0.672 | 0.711 |
| M5 | 0.720 | 0.783 | 0.715 | 0.782 |
| M10 | 0.722 | 0.851 | 0.717 | **0.858** |

Where the original M10 gap (+0.136) comes from:

| Effect at M10 | ΔAUC | 95% CI |
|---|---|---|
| Hyperparameters, data = train | −0.006 | [−0.007, −0.004] |
| Hyperparameters, data = train+val | +0.007 | [+0.006, +0.009] |
| **Data (adding val), untuned config** | **+0.128** | [+0.120, +0.137] |
| **Data (adding val), tuned config** | **+0.141** | [+0.133, +0.150] |
| Original confounded gap | +0.136 | [+0.127, +0.145] |

Main effects: **data +0.135, hyperparameters +0.001.** The same holds over all checkpoints (data +0.057, hyperparameters −0.001).

**Plain English:** tuning did essentially nothing. The whole "tuning gain" came from training on the validation data, which is the most recent ~13% of issues and the closest in time to the test set.

**Knock-on correction:** `task_b3_phase8.py` trained the untuned **per-checkpoint** models on train+val but the untuned **pooled** model on train only. So the retracted "+0.147 per-checkpoint advantage" (headline finding 7) was the **same data confound**, not "pooled inherited hyperparameters tuned for a different problem". The retraction stands; the reason given for it needs rewriting.

Also corrects B5's "overfitting-gap mechanism": the gap was data recency, not overfitting from bad hyperparameters.

---

## #1 — All 10 per-checkpoint models individually tuned: DONE

Every model, including the pooled one, was selected the same way:
- 9-config grid
- early stopping on its own validation rows
- mean val AUC over 3 seeds
- refit on train+val with 5 seeds

| Ckpt | Static | Pooled | Per-ckpt (indiv. tuned) | Per-ckpt − pooled [95% CI] | Verdict |
|---|---|---|---|---|---|
| M1 | **0.728** | 0.706 | 0.707 | +0.001 [−0.002, +0.003] | equivalent |
| M2 | 0.725 | 0.723 | 0.714 | −0.009 [−0.011, −0.007] | pooled better |
| M3 | 0.724 | **0.745** | 0.731 | −0.015 [−0.017, −0.012] | pooled better |
| M4 | 0.726 | **0.768** | 0.760 | −0.009 [−0.011, −0.007] | pooled better |
| M5 | 0.728 | **0.787** | 0.783 | −0.004 [−0.006, −0.002] | pooled better |
| M6 | 0.733 | **0.811** | 0.808 | −0.004 [−0.006, −0.002] | pooled better |
| M7 | 0.734 | **0.827** | 0.819 | −0.008 [−0.010, −0.006] | pooled better |
| M8 | 0.735 | **0.837** | 0.832 | −0.006 [−0.008, −0.004] | pooled better |
| M9 | 0.739 | **0.851** | 0.842 | −0.009 [−0.011, −0.007] | pooled better |
| M10 | 0.739 | **0.862** | 0.854 | −0.009 [−0.011, −0.006] | pooled better |

- **Pooled is never worse.** It is equivalent at M1 and significantly better at M2–M10, by 0.004–0.015 AUC. The differences are statistically real but practically small (all but M3 fall within ±0.01).
- Paper wording: replace "equivalent (max gap 0.0098 across M1/M5/M10)" with **"pooled matches or slightly beats individually-tuned per-checkpoint models at all 10 checkpoints"**. That is a stronger justification for using one model.
- **M3 crossover re-confirmed.** Pooled beats static with the CI excluding zero from M3 onward. Static still wins at M1.
- Selected configs: 9 of 10 checkpoints pick depth 8, and 6 of 10 pick lr 0.05 with 21–30 trees. The pooled model picks lr 0.05, depth 8, 29 trees (multi-seed). The old pick (lr 0.1, depth 8, 11 trees) was within noise of this.

Other metrics are in `reports/h1_metrics_panel.csv` (accuracy, balanced accuracy, F1, PR-AUC, Brier, log loss, per checkpoint and per model).
- Pooled accuracy at 0.5 goes from 0.617 at M1 to 0.853 at M10, against a majority baseline of 0.556 to 0.839.
- Pooled Brier goes from 0.223 at M1 to 0.105 at M10; static is 0.211 to 0.225.

---

## #2 — Text PCA components (30 / 50 / 75 / 100): DONE

- PCA was refit on train-split embeddings only. The first 30 components match the existing `text_pc_*` columns exactly (|corr| = 1.000000).
- Variance captured: **42.5% at k=30, 54.4% at k=50, 65.0% at k=75, 73.1% at k=100.**
- Each k was re-tuned on validation (static and pooled), then trained with 5 seeds.

| Test AUC (5-seed mean) | k=30 | k=50 | k=75 | k=100 |
|---|---|---|---|---|
| Static (all test) | 0.740 | 0.743 | 0.742 | 0.743 |
| Pooled dynamic, all | 0.797 | 0.796 | 0.796 | 0.796 |
| Pooled M1 | 0.706 | 0.708 | 0.709 | 0.708 |
| Pooled M5 | 0.787 | 0.785 | 0.785 | 0.786 |
| Pooled M10 | 0.862 | 0.862 | 0.858 | 0.861 |

- Static: +0.003 at k=100 (CI [+0.0026, +0.0034]). Significant but negligible.
- Dynamic: −0.0005 to −0.001 overall, with a slight loss at most checkpoints.
- Accuracy is flat (static 0.676 → 0.680; dynamic 0.730 → 0.727).

**Plain English:** the variance curve was still climbing, but the extra variance is not delay-relevant. Text signal saturates at 30 components. **Keep k=30.** This strengthens finding 5 ("text is the biggest SHAP group but largely redundant").

---

## What to change in the paper / results summary

1. **Finding 7:** keep the retraction, but change the reason to the train vs train+val data confound. Also add the new all-10 result (pooled ≥ per-checkpoint everywhere).
2. **Limitations:** remove "per-checkpoint tuned for only 3 of 10" and "tuned-vs-untuned confound". Both are resolved.
3. **Future work:** remove "more PCA components" and "per-checkpoint tuning at all ten". Report them as done, with null and pro-pooled results.
4. **Methodology:** add a "5 training seeds, seed-mean estimates" line. Single-run XGBoost AUC varies by up to ~0.02 at M10.
5. **Headline numbers stay the same** within seed noise: pooled M10 0.862 vs published 0.859, M3 crossover unchanged.

## New issue raised by #3 (worth the next slot)

Adding the most recent 13% of training data (val) is worth **+0.13 AUC at M10**. Without it, the dynamic model at M10 (0.72) is no better than static. So the late-checkpoint "dynamic advantage" depends heavily on training data that is close in time to the test period.

Two possible explanations need separating:
- (a) real concept drift, meaning recent data simply matters most;
- (b) right-censoring near the dataset's snapshot end. Only *resolved* issues are kept, so long-running recent issues are missing, and that changes what "delayed at M10" looks like in the val/test period.

A quick check: re-run with a cut-off that drops issues created within the longest project-median duration of the snapshot end, and see whether the val effect shrinks. This also ties into caution 3 ("test is more separable than validation") and item #4 (time-respecting cross-project splits).

**Cheap accuracy win spotted:** at the default 0.5 threshold the pooled dynamic model's balanced accuracy is only 0.54–0.58 (static: ~0.66). It predicts "delayed" almost everywhere late on. Choosing a threshold per checkpoint on validation would raise accuracy, F1 and balanced accuracy without touching AUC.
