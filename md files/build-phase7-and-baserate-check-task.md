# Task: Build Phase 7 (still missing) + add class-balance context to Phase 8's milestone table

## Part A — Phase 7 static baseline (this is missing entirely, build it now)

Check first: does `models/xgb_static_baseline.joblib` or
`reports/phase7_metrics.json` already exist? If not, build it:

1. Load `data/features/phase6_feature_table.csv`, merge in `Creation_Date`
   from `data/raw/issue.csv` on `ID` if not already present.
2. Chronological split: sort by `Creation_Date`, train = first 80%,
   test = last 20%. **This test set must be the same set of issues as
   Phase 8's test split** (`data/features/phase8_split.csv`) — verify the
   issue IDs match before proceeding; if they don't, use Phase 8's test
   split directly rather than recomputing your own, so the eventual
   static-vs-dynamic comparison is on identical held-out issues.
3. Drop `ID`, `Creation_Date`, and any leakage columns if present. Train:
   ```python
   from xgboost import XGBClassifier
   clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
   clf.fit(X_train, y_train)
   ```
4. Evaluate on test: AUC, precision/recall/F1, confusion matrix, and a
   majority-class baseline (predict the majority class of `y_train` for
   every test row) for comparison.
5. Save: `models/xgb_static_baseline.joblib`,
   `reports/phase7_metrics.json`, and a plot to `reports/`.

## Part B — Class-balance context for Phase 8's milestone table (quick, important)

The current milestone table (`reports/phase8_dynamic_metrics.csv`) shows
F1 climbing to 0.936-0.937 by M9-M10 while AUC only reaches ~0.76-0.78 —
that gap suggests the late-checkpoint evaluation population is heavily
skewed toward `delayed=1` (plausible: issues still active past 90-100%+ of
their expected duration are very likely to end up delayed), which would
make F1 look artificially high regardless of model quality.

For each checkpoint already in the milestone table (M1, M3, M5, M7, M9,
M10 — or all of M1-M10 if cheap to add): compute the actual `delayed`
class balance (% positive) among that checkpoint's evaluated test-split
issues, and what F1/accuracy a trivial "always predict the majority class
at this checkpoint" baseline would achieve. Add both as new columns to
`reports/phase8_dynamic_metrics.csv`.

## Report back

Phase 7's AUC/F1/majority-baseline numbers (the static reference point
that's been missing), and the per-checkpoint class-balance +
majority-baseline columns for Phase 8 — specifically confirm whether the
high F1 at M9-M10 is mostly a base-rate artifact or whether the model
meaningfully beats the majority-baseline even there.
