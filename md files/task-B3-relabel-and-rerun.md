# Task B3 — Relabel with train-only medians, re-run affected phases

Decision made: relabel. 3.36% exceeds the stated threshold, and the
project's core claim is leakage-freedom, so this gap gets closed rather
than documented.

**Commit first** (`git add . && git commit -m "pre-relabel results"`) —
the existing results must remain recoverable, because you will report
both label definitions side by side as a robustness check.

## What does NOT need re-running (don't waste time on these)

- **DTW clustering (8.3).** Clusters are built from `stall_ratio`
  trajectories; no labels involved. Medoids remain valid. Only the
  per-cluster *delay rates* need recomputing (trivial).
- **`expected_duration_proxy`, checkpoints, stall ratios.** Derived from
  resolution times, not labels. Unchanged.
- **SBERT embeddings and PCA.** Unchanged.
- **Pattern assignment.** Unchanged.

## Step 1 — Relabel

Compute each project's median `Resolution_Time_Minutes` using **only
non-test issues** — i.e. the train + validation portion (the first 80% by
`Creation_Date`, which is exactly the complement of the shared test split
used by both Phase 7 and Phase 8).

Apply those medians to label **all** issues (train, validation, test):

```
delayed_v2 = Resolution_Time_Minutes > train_val_project_median[Project_ID]
```

Report:
- Overall class balance under the new labels (it will no longer be exactly
  50% — that's expected and more realistic; say so rather than
  "correcting" it).
- Test-set class balance specifically.
- Any project where the train+val median is based on very few issues —
  small projects may now have noisy thresholds. Flag which, and how many
  issues they cover.

Save as a new column (keep the old label alongside for the comparison).

## Step 2 — Re-run, using the v2 (with-text) feature table

1. **Phase 7 static.** Same hyperparameters, same chronological split.
   Report AUC, F1 @ 0.5 and @ train-tuned threshold, majority baseline.
2. **Phase 8 dynamic — pooled and per-checkpoint**, with and without
   `pattern_label`. Regenerate the three-line per-checkpoint comparison
   (static / pooled / per-checkpoint) on identical test populations, with
   N, %delayed, and majority-F1 per checkpoint. **Report the crossover
   checkpoint under the new labels** (was M3 for per-checkpoint, M6 for
   pooled).
3. **Phase 9** — full 39-project LOPO plus within-project, cross-project
   AUC/F1, macro-mean/median/IQR, and the cross-beats-within count.
4. **Phase 10 SHAP** — static and per-checkpoint dynamic. Report feature
   rankings, with text as a summed group *and* noted per-component.
5. **Cluster delay rates** — recompute the four clusters' delay rates
   under the new labels (clusters themselves unchanged).

## Step 3 — Robustness comparison (this is the payoff)

Produce one table comparing headline numbers under both label
definitions:

| Result | Old label (full-population median) | New label (train+val median) |
|---|---|---|
| Phase 7 static AUC | 0.734 | ? |
| Per-checkpoint crossover | M3 | ? |
| Per-checkpoint M10 AUC | 0.881 | ? |
| Pooled M10 AUC | 0.734 | ? |
| Phase 9 cross-project mean AUC | 0.717 | ? |
| Phase 9 within-project mean AUC | 0.654 | ? |
| Cross-beats-within count | 33/38 | ? |

Save to `reports/label_definition_robustness.csv`.

**State the conclusion explicitly:** are the qualitative findings
(dynamic overtakes static mid-lifecycle; per-checkpoint beats pooled;
cross-project beats within-project; patterns contribute marginally)
stable across both label definitions? If yes, that is a robustness
result worth reporting in the paper. If any conclusion flips, that is
important and must be reported prominently rather than buried.

## Report back

New class balances, all re-run numbers, the robustness comparison table,
and a clear statement of which conclusions held and which (if any)
changed.
