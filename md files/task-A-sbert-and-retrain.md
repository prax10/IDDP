# Task A — SBERT text features + hyperparameter match + full pipeline re-run

**Run this FIRST.** It changes the feature table, so every downstream
phase must be re-run afterward. Do not start Tasks B or C until this is
complete.

Commit before starting (`git add . && git commit -m "pre-SBERT baseline"`)
so the current results are recoverable if this goes wrong.

## Step 1 — Encode issue text with SBERT

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

- Pull `ID`, `Title`, `Description_Text` from `data/raw/issue.csv`, for
  the **203,290 resolved issues only** (don't encode the full 458k — waste
  of time, nothing else uses them).
- Encode `Title.fillna('') + '. ' + Description_Text.fillna('')`, with
  `batch_size=64, show_progress_bar=True`. This produces a (203290, 384)
  array. Expect 20-40 minutes on a laptop CPU.
- **Cache the raw embeddings immediately** to `data/features/sbert_raw.npy`
  before doing anything else — if a later step fails you must not have to
  re-encode.

## Step 2 — PCA to 30 dimensions (leakage-safe)

⚠️ **Fit PCA on training-split issues ONLY**, then `transform()` the
validation and test issues. Fitting PCA on all data would leak test-set
structure into the feature representation. Use the same chronological
split as everywhere else (`data/features/phase8_split.csv`).

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=30, random_state=0)
pca.fit(emb[train_mask])          # train only
emb30 = pca.transform(emb)        # all rows
```

Report the cumulative explained variance ratio of the 30 components.
Save as `text_pc_0` ... `text_pc_29` columns.

## Step 3 — Rebuild the feature table

Merge the 30 text columns into `data/features/phase6_feature_table.csv`
on `ID`. Save as `data/features/phase6_feature_table_v2.csv` (keep the
original — you'll want the with-text vs. without-text comparison).

Re-run the leakage test on the new table: max |correlation| with
`delayed` across all features, confirm no post-resolution columns.

## Step 4 — Re-run Phase 7 with text features

Same chronological split, same hyperparameters (n_estimators=400,
max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8).
Save as `models/xgb_static_v2.joblib`, metrics to
`reports/phase7_metrics_v2.json`.

**Report AUC with text vs. without text (0.724 baseline)** — this is the
direct measure of whether text carries signal.

## Step 5 — Re-run Phase 8 dynamic models with MATCHED hyperparameters

This fixes a stated limitation: the dynamic models previously used
lr=0.1 / 300 rounds while Phase 7 used lr=0.05 / 400 rounds, so they were
never a matched ablation.

Retrain all dynamic variants on the checkpoint data with **Phase 7's exact
hyperparameters**, using the v2 feature table:
- (a) dynamic, no pattern
- (b) dynamic, with pattern
- (c) dynamic, with pattern, no `elapsed_minutes` (the ablation)

Re-run the per-checkpoint evaluation (M1-M10) on the test split, scoring
the v2 static model on the same per-checkpoint populations for the
three-way comparison. Save to `reports/phase8_dynamic_metrics_v2.csv` and
regenerate `reports/phase8_checkpoint_accuracy_v2.png`.

Keep reporting N and the majority-class baseline per checkpoint.

## Step 6 — Re-run Phase 10 SHAP on the v2 models

Same as before. For the text features, **report them as a group** —
sum mean|SHAP| across all 30 `text_pc_*` columns and report that
aggregate alongside individual features, since 30 individually-small PCA
components would otherwise clutter the summary plot and understate text's
collective contribution.

Report where the text group ranks relative to
`assignee_prior_delay_rate`, `elapsed_minutes`, `log1p_stall`, and
`pattern_label`.

## Report back

1. SBERT encoding time, PCA cumulative explained variance.
2. Phase 7 AUC: with text vs. without (0.724).
3. The new per-checkpoint three-way table, and whether matched
   hyperparameters changed the M6 crossover point.
4. Where the text-feature group ranks in SHAP.
5. Leakage test results on the v2 table.
