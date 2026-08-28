# Task B6 — Accuracy reporting + statistical significance (short, ~30-40 min)

Two additions to the final results. No retraining — use the saved final
models (static: lr=0.2/depth=4/42 rounds; pooled dynamic with pattern:
lr=0.1/depth=8/11 rounds), v3 train+val-median labels, v2 features.

## Part 1 — Accuracy (never actually computed for the final models)

Report **accuracy** for:

1. The final tuned **static** model on the full test set, at both the 0.5
   threshold and the train-tuned threshold (0.405). Alongside it: the
   majority-class accuracy on the same test set (53.0%).
2. The final tuned **pooled dynamic** model at **each checkpoint M1-M10**,
   alongside that checkpoint's majority-class accuracy (predict the
   checkpoint's majority class for every row).
3. The static model's accuracy on each per-checkpoint test population, so
   all three columns are comparable.

Add these as columns to `reports/phase8_final_tuned_comparison.csv`.

**Context to state alongside every accuracy figure:** the checkpoint
populations become increasingly delay-skewed (57.7% → 87.8%), so
accuracy rises mechanically with checkpoint index regardless of model
skill. Always report the majority-class accuracy next to it — the gap
between them is the actual skill.

## Part 2 — Bootstrap confidence intervals and significance tests

Currently every comparison in the results is a bare point estimate
("0.738 beats 0.722") with no uncertainty quantification. Close that.

**A. Bootstrap 95% CIs on AUC.** For each checkpoint, bootstrap the test
set (1,000 resamples, fixed seed — report it) and compute the 95%
confidence interval for:
- static AUC
- pooled dynamic AUC
- **the difference** (dynamic − static)

The CI on the *difference* is the important one: if it excludes zero, the
difference is significant at that checkpoint. Report the exact checkpoint
where the difference CI first excludes zero in favour of dynamic — this
is the statistically-supported version of the M3 crossover claim, and
should replace the bare point-estimate version in the paper.

**B. Significance test across checkpoints.** Following Kula et al.'s
approach for comparability: Wilcoxon signed-rank test on the paired
per-checkpoint AUCs (static vs. dynamic across M1-M10), plus a
Vargha-Delaney A₁₂ effect size. Report the p-value and effect size.

**C. Phase 9 generalization.** Wilcoxon signed-rank test on the paired
per-project cross-project vs. within-project AUCs (38 projects), plus
A₁₂. This tests whether the "cross-project beats within-project on 30/38"
finding is statistically supported rather than a lucky split.

**D. Pattern contribution.** Bootstrap CI on the AUC difference between
dynamic-with-pattern and dynamic-without-pattern, per checkpoint. State
plainly whether the pattern's contribution is statistically
distinguishable from zero at any checkpoint — given the SHAP ranking
(#12 of 14), it may well not be, and that is a legitimate and useful
finding to report.

Save all of the above to `reports/statistical_tests.csv` and produce one
plot showing dynamic-minus-static AUC with its 95% CI band across
checkpoints (`reports/auc_difference_ci.png`) — the point where the band
crosses zero is a strong, self-explanatory figure for the paper.

## Report back

The accuracy table with majority-class baselines; the checkpoint where
the AUC-difference CI first excludes zero; Wilcoxon p-values and A₁₂
effect sizes for the checkpoint comparison and the Phase 9 comparison;
and whether the pattern feature's contribution is statistically
distinguishable from zero.
