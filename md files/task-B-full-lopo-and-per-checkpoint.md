# Task B — Full 39-project LOPO + per-checkpoint models

**Run AFTER Task A.** Use the v2 (with-text) feature table and v2 models
throughout, so all final numbers come from the same model generation.

## Part 1 — Full 39-project leave-one-project-out (upgrades the stretch goal)

Extend Phase 9 from 10 projects to **all 39**. For each project P: train
the static v2 model on the other 38, test on P. Record project ID, N,
delay rate, cross-project AUC/F1.

Also compute within-project AUC/F1 for every project where it's
meaningful (skip projects whose 80/20 chronological split leaves fewer
than ~30 test issues or lacks both classes — report which were skipped
and why).

Report: full 39-row table, macro-mean, median, IQR, and the count of
projects where cross-project beats within-project (previously 8 of 10 —
does that hold at full scale?).

Save `reports/phase9_generalization_full.csv` and
`reports/phase9_generalization_full.png` (per-project AUC, N annotated,
sorted by N so the small-project noise is visually obvious).

**Keep the temporal caveat in mind:** cross-project training uses all
time periods while within-project uses a chronological split. Don't
"fix" this — just make sure the report states it, since it's already
documented as a known asymmetry.

## Part 2 — Per-checkpoint models (closer to Kula's design)

Currently one pooled classifier serves all ten checkpoints. Kula refits
at each milestone. Test whether that matters.

Train **ten separate models**, one per checkpoint, each on only that
checkpoint's training rows (with pattern label, v2 features, Phase 7
hyperparameters). Evaluate each on its own checkpoint's test population.

Compare against the pooled model's per-checkpoint AUCs from Task A.
Report both lines on one plot: `reports/phase8_pooled_vs_percheckpoint.png`.

Expected considerations to report honestly:
- Late checkpoints have less training data (M10 has far fewer rows than
  M1), so per-checkpoint models may *underperform* there despite the
  cleaner design — report N per checkpoint alongside.
- If pooled wins, that's a legitimate finding (pooling shares statistical
  strength across checkpoints); if per-checkpoint wins, that's closer to
  Kula and worth adopting as the headline model.

## Report back

The full 39-project table with macro-mean/median/IQR and the
cross-beats-within count; the pooled-vs-per-checkpoint comparison with N
per checkpoint; and which model configuration you'd recommend as the
final headline model based on the numbers.
