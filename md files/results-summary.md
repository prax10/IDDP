# Results Summary — Intelligent Dynamic Delay Prediction (TAWOS)

Complete core chain: EDA → features → static baseline → dynamic model →
generalization → explainability. All phases built and verified.

---

## Headline findings (lead with these)

1. **Dynamic prediction works, and improves as issues progress.** AUC
   climbs from 0.659 at the first checkpoint to 0.780 at the tenth.
2. **But it only beats static prediction from the midpoint onward.** On
   identical test populations, the creation-time-only static model wins
   through M5; crossover at M6; dynamic leads clearly by M9-M10.
   Concrete claim: *progress information only starts paying off roughly
   halfway through an issue's expected lifetime.*
3. **The late-stage advantage comes from survival, not from trajectory
   shape — proven by ablation.** Removing `elapsed_minutes` costs nothing
   at M1 (0.670 vs 0.668) but 4 AUC points by M10 (0.719 vs 0.760). The
   issue-specific dynamics (`log1p_stall` #7 of 13 in SHAP,
   `pattern_label` #8) are real but modest. What predicts delay late is
   *"this issue has been open unusually long for its kind"* — not the
   fine-grained stall pattern.
4. **Delay patterns exist in TAWOS but contribute marginally at issue
   level.** K=4 chosen empirically from TAWOS's own elbow curve (the same
   K Kula et al. found, independently). Adding the pattern moves AUC
   +0.005 to +0.01 early and slightly *hurts* at M9-M10. **This is a
   legitimate negative-transfer result** — see the Kula comparison below
   for the mechanical explanation.
5. **The method generalizes across projects — and pooling actively
   helps.** Cross-project training beats within-project training on 8 of
   10 projects (mean AUC 0.708 vs 0.656). The usual "does it collapse on
   unseen projects" failure mode does not occur here.

---

## Phase 1-5 — EDA

- 458,232 issues, 39 projects. **203,290 genuinely resolved** after a
  filter requiring a real completion `Resolution` AND excluding non-work
  `Status` values (needed because some projects, e.g. project 28, reuse
  `Resolution='Done'` for Won't-Fix/Invalid tickets).
- Label: `delayed = Resolution_Time_Minutes > per-project median`.
  49.99% positive — balanced by construction.
- Cross-validated three independent ways (SQL counts, Python `repr()`
  checks, direct pandas checks) — all reconcile.

## Phase 6 — Feature engineering

- 203,290 rows × 12 columns. Priority normalized across two Jira
  vocabularies onto a 1-5 ordinal scale + separate `Unknown`; Type rare
  categories (<100 issues) bucketed to `Other`.
- Workload features computed chronologically per assignee (prior resolved
  count, prior delay rate via expanding mean, concurrent open issues).
- **Leakage tests all pass:** max |correlation| with the label is 0.328
  (`assignee_prior_delay_rate`); no post-resolution columns present;
  manual expanding-mean recompute matched exactly on spot-checked
  assignees.

## Phase 7 — Static baseline

| Metric | Value |
|---|---|
| AUC | **0.724** |
| Accuracy | 0.666 (vs. 50% chance on a balanced label) |
| F1 @ 0.5 threshold | 0.641 |
| F1 @ tuned threshold (0.385, tuned on train only) | **0.683** |
| Majority-class baseline F1 | 0.656 |

The default 0.5 threshold was a poor operating point for this class
balance — AUC 0.724 shows the ranking was sound all along. Report the
tuned number alongside the default.

## Phase 8 — Dynamic model

**Setup:** 167,174 issues had ≥1 observation checkpoint (1,342,536
checkpoint rows, ~8.03 per issue). The ~36k resolved issues with zero
checkpoints resolved faster than 10% of their group's historical median —
the variable-length design working as intended, but it means the dynamic
model's evaluation population is necessarily narrower than the static
model's.

**Data-quality fixes found during the build:** `Change_Log` rows labelled
`Change_Type='STATUS'` actually mix genuine workflow transitions with
Resolution-field changes (one side null) — ~412k such rows dropped,
keeping ~871k genuine transitions. Separately, ~9.6% of naive dwell
segments ran negative because status logging continues after the recorded
`Resolution_Date`; fixed by clipping at `Resolution_Date`.

**Pattern discovery:** DTW + hierarchical clustering (complete linkage —
average linkage produced a degenerate chaining artifact) on a stratified
4,000-issue training subsample, over `log1p(stall_ratio)` (raw signal is
heavy-tailed: median 1.3, p99 ~37,000, max ~1.4M). K=4 from a clean WSS
elbow.

| Cluster | Size | Delay rate | Shape |
|---|---|---|---|
| 1 | 3,727 (93%) | 60.5% (≈ baseline) | Steady low stall, gentle drift — the unremarkable majority |
| 2 | 227 (5.7%) | **69.2%** | Escalates continuously, never levels off |
| 3 | 8 (0.2%) | 50.0% (n too small) | Sustained high plateau, sharp crash right before resolution |
| 4 | 38 (0.95%) | 57.9% | Early-onset jump to a high plateau, stays elevated |

Subsample baseline delay rate is 60.9% (not 50%) — it covers only issues
with valid checkpoints.

**Pattern stability:** 88.9% of issues never change pattern label once
assigned; median stabilization at M3 (locks in at first opportunity).

**Fair three-way comparison** (identical per-checkpoint test populations):

| Checkpoint | N | Static | Dynamic (no pattern) | Dynamic (+pattern) | Dynamic (+pattern, no elapsed) |
|---|---|---|---|---|---|
| M1 | 34,398 | **0.714** | 0.659 | 0.668 | 0.670 |
| M3 | 29,838 | **0.712** | 0.689 | 0.693 | — |
| M5 | 26,279 | **0.717** | 0.710 | 0.712 | 0.698 |
| M6 | 24,947 | 0.723 | 0.726 | **0.730** | — |
| M8 | 22,243 | 0.733 | 0.748 | **0.748** | 0.717 |
| M9 | 21,232 | 0.742 | **0.757** | 0.750 | — |
| M10 | 20,236 | 0.748 | **0.780** | 0.760 | 0.719 |

## Phase 9 — Cross-project generalization

10 projects: 5 small (<700 resolved issues) + 5 large.

| Project | N | Cross-project AUC | Within-project AUC | Difference |
|---|---|---|---|---|
| 10 | 249 | 0.773 | 0.661 | +0.112 |
| 6 | 281 | 0.609 | 0.551 | +0.059 |
| 9 | 407 | 0.709 | 0.623 | +0.086 |
| 2 | 625 | 0.644 | 0.637 | +0.007 |
| 35 | 647 | 0.661 | 0.576 | +0.084 |
| 34 | 30,709 | 0.717 | 0.695 | +0.022 |
| 33 | 29,167 | 0.720 | 0.723 | −0.003 |
| 28 | 19,654 | 0.742 | 0.763 | −0.021 |
| 12 | 12,387 | 0.759 | 0.647 | +0.112 |
| 22 | 11,854 | 0.745 | 0.681 | +0.067 |

- **Cross-project:** mean 0.708, median 0.718, IQR [0.673, 0.744]
- **Within-project:** mean 0.656, median 0.654, IQR [0.627, 0.692]

Cross-project training wins on 8 of 10 projects; only the two largest
(33, 28) see within-project match or slightly exceed it.

## ⚠️ Interpretation cautions (state these before someone asks)

**1. F1 at late checkpoints is a base-rate artifact, not model skill.**
The surviving test population becomes progressively delay-skewed (57.7%
delayed at M1 → 87.8% at M10). A trivial "always predict delayed"
baseline scores F1 ≈ 0.935 at M10; the model scores 0.937. **Use AUC as
the primary metric** — it isn't distorted by the shifting base rate.

**2. `elapsed_minutes` is not an independent progress measurement.**
Checkpoints are defined as `M_i = (i/10) × expected_duration_proxy`, so
within any single checkpoint `elapsed_minutes` is *perfectly*
proportional to `expected_duration_proxy` (verified: correlation =
1.000000 at both M5 and M8) — i.e. it carries a static, group-level
property, not this issue's individual progress. Note its pooled
correlation with checkpoint index is only 0.288, so it is *not* simply
"checkpoint index in disguise" — most of its variance is cross-issue
differences in expected-duration scale. Neither quantity is leakage; both
are knowable at prediction time. But "elapsed time dominates" must not be
read as "progress dynamics dominate."

**3. Phase 9's two setups are not perfectly comparable on time.**
Within-project runs use a chronological 80/20 split (true forecasting:
past → future). Cross-project runs train on all 38 other projects across
*all* time periods, including issues created after the test project's
test issues. So the cross-project model has both more data *and* an
easier temporal task. Both factors plausibly contribute; training-set
size clearly dominates for the small projects (project 10's within-project
split yields ~199 training rows vs. ~203,000 available cross-project).
LOPO is routinely done this way, but state the asymmetry explicitly.

## Phase 10 — Explainability (SHAP)

- **Static model top 5:** `assignee_prior_delay_rate` > `Type_Normalized`
  > `n_links` > `Story_Point` > `Project_ID`. Who it's assigned to, and
  their track record, is the strongest creation-time signal.
- **Dynamic model:** `elapsed_minutes` #1 by a wide margin, then
  `Project_ID`, `assignee_prior_delay_rate`, `Type_Normalized`,
  `n_links`; `log1p_stall` #7, `pattern_label` #8 (of 13).
- **Early vs. late importance shift — refutes the expected story.** The
  anticipated pattern was "static features early, progress features late."
  What actually happens: `elapsed_minutes` is the *only* feature that
  gains importance late (+0.68, by far the largest mover), with
  `Project_ID` gaining slightly (+0.23). Everything else — including
  `pattern_label` (−0.051) and `log1p_stall` (−0.014) — *loses*
  importance. **The stall and pattern signals are most useful
  early-to-mid, not late.** Corroborated independently by the
  `elapsed_minutes` ablation above.

---

## Comparison to Kula et al. (ESEC/FSE '23)

**Critical: they never reported classification accuracy.** Their task was
regression over a continuous label (BRE), so accuracy is undefined for
their work. Their metrics were MAE and **Standardized Accuracy (SA)**,
where `SA = (1 − MAE/MAE_random) × 100` — the percentage by which the
model beats *random guessing*, **not** the share of correct predictions.
Do not cite their SA as accuracy, and do not let anyone compare it
against this project's accuracy.

| | Kula et al. | This project |
|---|---|---|
| Unit of analysis | 4,040 epics, 270 teams, one company (ING) | 203,290 issues, 39 open-source projects |
| Label | Continuous BRE | Binary (> per-project median) |
| Metrics | SA 66% → 92%; MAE 0.19 → 0.04 over milestones | AUC 0.659 → 0.780 over checkpoints |
| Dynamic vs. static | +12–57% SA, +16–81% MAE | +0.03 AUC at M10; static wins through M5 |
| Pattern benefit | Identical at M1-M2, then +9–20% SA from M3 on | +0.005–0.01 early, negative at M9-M10 |
| Cluster sizes | 36% / 44% / 14% / 6% | 93% / 5.7% / 0.95% / 0.2% |
| K chosen | 4 (elbow) | 4 (elbow, independently) |

**Why the pattern feature underperforms here — the mechanical
explanation.** Kula's four patterns split the population into four
meaningfully-sized groups. Ours collapse into one dominant "unremarkable"
cluster holding 93% of the subsample (65% of all checkpoint rows). A
feature that takes the same value for the overwhelming majority of rows
cannot discriminate much, regardless of model quality.

**Kula et al. explicitly called for this replication.** From their
Threats to Validity: *"our data may not be representative of software
projects in other organizations and open source settings... Replication
of our work is needed to validate the findings in other settings and
reach more general conclusions."* And: *"The patterns in other
organizations might differ from the four patterns identified at the case
company."* This project answers that call directly, and reports what was
found rather than what was hoped for.

---

## On accuracy (for the viva/defense)

Accuracy is **66.6%** for the static model against a 50% chance baseline
on a deliberately balanced label. **Above ~90% would be a red flag on
this task, not a goal** — it would indicate either data leakage (using
post-resolution information) or a lopsided label where predicting one
class always scores high while learning nothing. The ~88% accuracy
available at late checkpoints is a base-rate artifact (87.8% of surviving
issues are delayed there; "always predict delayed" scores 87.8%). AUC is
the honest measure of skill here, and it is reported throughout.

---

## Limitations to state explicitly

- The label is a relative proxy (slower than that project's own median),
  not an absolute SLA breach.
- Priority/Type normalization approximates across projects that don't
  share one Jira vocabulary; rare-category thresholds are
  population-dependent (Milestone has 29 issues in the resolved
  population and folds into `Other`).
- The dynamic model's evaluation population (167,174 issues with ≥1
  checkpoint) is narrower than the static model's (203,290), and becomes
  a survivorship-biased subset at later checkpoints by construction.
- Clustering ran on a 4,000-issue stratified subsample, not all 142,303
  training issues — a full pairwise DTW matrix is computationally
  infeasible at that scale. Medoids from that subsample classify
  everything else.
- The dynamic models are a single pooled classifier across all
  checkpoints rather than a per-checkpoint refit — a simplification
  relative to Kula et al.'s design.
- The dynamic models used improvised hyperparameters (lr 0.1, 300 rounds)
  rather than Phase 7's (lr 0.05, 400 rounds), because Phase 7 did not
  exist when they were trained. Not a matched ablation.
- Phase 9's cross-project vs. within-project comparison carries a
  temporal asymmetry (see caution 3).
- Small projects produce noisy per-project results; within-project test
  sets for the five small projects hold only 50-130 issues.
- SBERT text features (Phase 6.4) were deferred — 93.6% of issues have
  description text that no current model reads.
- The Bayesian logistic layer (Phase 8 stretch) was not implemented.

## Future work (honest next steps)

- Add the deferred SBERT text features — the largest untapped signal.
- Retrain dynamic models with Phase 7-matched hyperparameters for a
  clean ablation.
- Per-checkpoint model refits instead of one pooled classifier (closer to
  Kula's design).
- Full 39-project leave-one-project-out, with time-respecting
  cross-project splits to remove the temporal asymmetry.
- The Bayesian logistic layer for calibrated uncertainty estimates.

## Framing for the paper

"We adapt Kula et al.'s dynamic delay-prediction framework to issue-level
TAWOS data, retaining the core mechanism — discovering recurrent temporal
delay patterns via trajectory clustering and feeding the evolving pattern
into dynamic prediction — while adapting the intermediate delay
representation (stall ratio in place of delayed story points), target
(binary Relative Resolution Delay in place of continuous BRE), milestone
definition (historically-estimated expected duration in place of planned
iterations), and predictors to the structure of TAWOS. We find that
dynamic prediction improves over a static baseline only from the midpoint
of an issue's expected lifetime onward; that its advantage derives
primarily from survival to a given checkpoint rather than from
fine-grained stall dynamics or discovered delay patterns — which, unlike
at epic level in Kula et al., contribute only marginally at issue level;
and that the approach nonetheless generalizes across projects, with
cross-project training outperforming within-project training on 8 of 10
projects tested." **Do not claim to have reproduced Kula et al.**
