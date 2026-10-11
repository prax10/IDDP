# Paper updates after Task L

This adds to `paper-updates-v5.md` (Tasks H–J), which still applies. It lists only what Task L
changes. Evidence and full tables are in `task-L-results.md` and `reports/l*`.

---

## Claims that change

| Old claim | New claim | Evidence |
|---|---|---|
| Patterns fail because "93% of issues fall into one cluster" | The 93% comes from **complete-linkage clustering**, not from issue-level data. Full-population k-medoids gives balanced clusters (65/27/7/1% level, 76/12/12% shape) with very different late rates (6% vs 71%). | `l1_run_summary.txt`, `l1_clara_full_population.csv` |
| Patterns add little "because they can't discriminate" | Patterns **do** carry signal: balanced shape clusters add +0.011–0.012 AUC at M3–M5 to the Phase-8 features. But they add **nothing** on top of the v5b activity features, which describe the same behaviour directly. **The pattern is redundant, not uninformative.** | `l1b_pattern_contribution.csv` |
| Pattern labels exist only from M3 (needs 3 points) | Assigning from M1 with the available prefix changes nothing. Note it as a design choice, not a limitation. | `l1b_pattern_contribution.csv` |
| Clustering at full scale "infeasible" | Done: numba DTW (verified against dtaidistance) + CLARA, over all 115,176 training trajectories. Remove from Future Work. | `task_l1_full_dtw.py` |
| Generalisation shown only across projects *within* TAWOS | It also holds across **organisations**. Zero-shot AUC on 5 unseen Jira organisations is 0.755–0.852 (mean 0.801), equal to or better than own-history training on 3 of 5 (mean 0.792). | `l4_external_validation.csv` |
| Binary prediction only (regression was future work) | A regression of how late works: SA 59% (54% at M1 → 63% at M10), with calibrated intervals. Remove from Future Work. | `l2_regression_metrics.csv` |
| XGBoost chosen without comparison | XGBoost is best, but by a small margin over strong baselines (LR −0.011, MLP −0.003, GRU −0.004). The **features** carry the result. | `l3_baselines.csv` |

## New RQ answers

- **RQ2 (patterns), rewritten:** "Delay patterns found by full-scale DTW clustering are balanced and
  predictive on their own (+0.012 AUC at M3 over the Phase-8 features). But they are fully
  redundant once workflow-activity features are observed (Δ = 0.000, CI within ±0.005). The
  mechanism behind Kula et al.'s patterns transfers; its added value does not, because
  issue-level Jira history records that behaviour directly."
- **RQ3 (generalisation), extended:** "...and a model trained only on TAWOS ranks issues in five
  unseen Jira organisations at AUC 0.76–0.85 without retraining."
- **New RQ6 (optional):** Can the model say how confident it is, and how late an issue will be?
  Yes. 90% conformal sets cover 92% on test, and confident-late calls on not-yet-overdue issues
  are 96% precise.

## Method: what to add

1. **Full-scale DTW** (L1). Numba DTW (dtaidistance definition, 200-pair check), CLARA k-medoids
   scored on the full population, level and z-normalised shape variants, elbow on full-population cost.
2. **Regression target** (L2). `rrd = log2(resolution time / train+val project median)`, so
   rrd > 0 ⇔ the binary label. Absolute-error XGBoost. SA per Shepperd & MacDonell.
3. **Baselines** (L3). The same features and rows for all models: LR, RF, literature-factor RF,
   MLP (2×256) and GRU over M1..Mt. Paired bootstrap against XGBoost.
4. **Conformal prediction** (L5). Train-only models, VAL calibration, TEST coverage:
   - Mondrian (class-conditional) sets for late / on time;
   - CQR for rrd;
   - split-conformal residual intervals for Scrum spill counts.

   Deployed-model coverage re-checked (L5b).
5. **External pipeline** (L4):
   - the mongodump archive streamed with the standard library;
   - Jira JSON → TAWOS layout;
   - TAWOS population, label and checkpoint definitions re-applied, with the self-test on a
     TAWOS project (2,210/2,210 issues, 99.95% of rows);
   - a portable feature set of 49 features (no Project_ID, text PCs, notebook statics or comments).
6. **Recommendations** (L6). Counterfactual re-scoring for four actions, checked observationally
   on test with risk-decile matching. The strict version keeps only issues still open after 7 days.

## Results: new tables

**Baselines (6.x).**

| Model | AUC | Δ vs XGB [95% CI] | M1 | M10 |
|---|---|---|---|---|
| XGBoost v5b | 0.866 | — | 0.804 | 0.912 |
| MLP | 0.864 | −0.003 [−0.003, −0.002] | 0.812 | 0.904 |
| GRU (sequence) | 0.863 | −0.004 [−0.004, −0.003] | 0.810 | 0.902 |
| Logistic regression | 0.856 | −0.011 [−0.011, −0.010] | 0.811 | 0.888 |
| Random forest | 0.850 | −0.017 [−0.017, −0.016] | 0.787 | 0.890 |
| Literature factors + RF | 0.782 | −0.085 [−0.086, −0.083] | 0.726 | 0.790 |
| XGB+MLP+GRU average (untuned) | 0.871 | +0.005 [+0.004, +0.005] | 0.817 | 0.912 |

**Regression (new subsection, next to the Kula comparison).**

| Scope | SA, v5b | SA, creation-time | SA, project median | Median abs. error (days remaining) | 80% interval coverage |
|---|---|---|---|---|---|
| All | 58.9% | 49.8% | 35.5% | 19.9 | 80.1% |
| M1 | 53.9% | 42.6% | 39.0% | 16.4 | 81.9% |
| M10 | 63.2% | 54.3% | 27.0% | 23.6 | 78.7% |

Say explicitly that Kula's SA of 66–92% is on epic-level BRE at one company: a different unit and
target, with only the metric shared.

**Uncertainty (new subsection).**
- Flow conformal sets: 92.1% coverage (deployed model: 92.5%); confident calls on 51% of rows.
- Confident-late calls on not-yet-overdue issues: **96.4%** precise (M1 97.4%, M10 95.2%), catching 28% of late issues.
- CQR: 84.2% / 92.7% coverage at the 80% / 90% targets.
- Scrum spill-count ranges: 88–92% coverage, against 69–77% for the naive Poisson-binomial interval.

**External validation (new subsection under 6.5).** Use the L4 table in `task-L-results.md`
(5 repos × A/B/C, with the A−B CI). Add the M1 → M10 rise on each repo (+0.12 to +0.19).

**Recommendations (Discussion, or a short Results paragraph).**

| Action | Model estimate | Observed (strict) |
|---|---|---|
| Start work | −0.22 | −0.15 |
| Get into review | −0.31 | −0.16 |
| Comment / discuss | −0.04 | −0.12 |
| Reassign to best colleague | −0.08 | −0.08 |

The direction agrees on all four, the magnitudes are approximate, and none of it is causal.

## Threats to validity: add

- **External repos.** No comments in the dataset (portable model: 0.858 vs 0.866 on TAWOS).
  Heavy censoring for Qt (12% safe) and IntelDAOS (0%). Decision thresholds need local recalibration.
- **Conformal coverage at M1** for late issues is 79% (target 90%), from drift between the
  validation and test periods. Overall coverage holds.
- **Recommendations are not causal.** Even the strict observational check is confounded. A
  randomised or natural-experiment study is needed.
- **Neural baselines** got light tuning only (epochs chosen on val, fixed architecture). The
  conclusion "XGB ≥ NN" is for comparable effort, not exhaustive tuning.

## Remove from Future Work

Full-scale clustering · regression formulation · external dataset · stronger baselines ·
uncertainty intervals · actionable recommendations. All are done.

## Future Work, new list

1. A causal test of the recommendations (randomised nudges, or a natural experiment such as
   reviewer availability).
2. Per-organisation threshold calibration from a small amount of local history.
3. External validation on a dataset that includes comments.
4. Deploying the XGB+MLP+GRU ensemble (+0.013 at M1) with SHAP-style explanations for the neural parts.
5. Bringing the regression ("how late") into Scrum mode as remaining days vs the sprint end.

## Methodological narrative: add

13. **A published explanation retracted by our own follow-up.** The 93% cluster came from the
    clustering method; full-scale k-medoids replaced the "degenerate clusters" story with a
    stronger "redundant with activity features" story.
14. **The external pipeline was validated before it was used.** It reproduced TAWOS's own
    population and features on a TAWOS project before any external number was read.
15. **Calibration checked on the deployed models,** not just the clean train-only protocol.
