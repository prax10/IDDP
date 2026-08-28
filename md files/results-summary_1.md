# Results Summary — Intelligent Dynamic Delay Prediction (TAWOS)

**FINAL — settled numbers.** All phases complete, model work closed.

Final configuration:
- **Labels:** `delayed = Resolution_Time_Minutes > project median`, where
  the median is computed from **train+validation issues only** (never
  test).
- **Features:** issue attributes + normalized priority/type +
  chronological assignee workload + dependency counts + 30 PCA components
  of SBERT text embeddings (PCA fitted on train only).
- **Static model:** XGBoost, lr=0.2, depth=4, 42 rounds.
- **Dynamic model:** pooled XGBoost with pattern label, lr=0.1, depth=8,
  11 rounds. Hyperparameters selected on a held-out validation split.

---

## Headline findings

**1. Dynamic prediction overtakes static prediction at checkpoint M3 —
30% of an issue's expected duration.**
This is the single most robust result in the project. It held identically
across two label definitions (full-population median, train+val median),
two model architectures (pooled, per-checkpoint), and both untuned and
validation-tuned hyperparameters. Four independent stress tests, same
answer.

**2. Before M3, creation-time information is better than progress
information.** At M1 the static model wins (0.727 vs 0.713). Early in an
issue's life there is barely any progress to observe, and the extra
signal is noise. Practically: don't bother re-forecasting an issue in its
first 20-30% of expected duration.

**3. The advantage grows substantially with elapsed lifetime** — from
+0.016 AUC at M3 to +0.125 at M10 (0.859 vs 0.734).

**4. Discovered delay patterns contribute marginally at issue level.**
K=4 emerged from TAWOS's own elbow curve (independently matching Kula's
K=4), and the "escalating, never recovers" pattern does carry the highest
delay rate. But 93% of the clustering subsample falls into one
unremarkable cluster, and `pattern_label` ranks #12 of 14 features by
SHAP. **This is a negative-transfer result:** Kula et al. found patterns
gave 9-20% SA improvement on epic-level data at one company; that does
not carry over to issue-level open-source data.

**5. Issue text is the strongest static feature group but largely
redundant.** SBERT text components collectively rank #1 in the static
model (summed mean|SHAP| 0.895 vs 0.381 for `assignee_prior_delay_rate`),
yet adding them moves AUC only +0.010. Text carries substantial
information that priority, type, project, and assignee history already
capture through other channels.

**6. The method generalizes across projects — pooling helps.**
Cross-project training (train on 38, test on the 39th) beats
within-project training on 30 of 38 projects: mean AUC 0.710 vs 0.657.
The usual "collapses on unseen projects" failure mode does not occur.

**7. A finding that did not survive scrutiny — report this deliberately.**
An intermediate result showed per-checkpoint models beating a pooled
model by up to +0.147 AUC. Validation-based tuning revealed this was an
artifact: the pooled model had inherited hyperparameters tuned for a
different problem. With each properly tuned, the two are **equivalent**
(max gap 0.0098 across M1/M5/M10). Pooled is adopted as the headline
model — same performance, one classifier instead of ten.

---

## Phase 1-5 — EDA

- 458,232 issues, 39 projects → **203,290 genuinely resolved** after a
  filter requiring a real completion `Resolution` AND excluding non-work
  `Status` values (some projects, e.g. 28, reuse `Resolution='Done'` for
  Won't-Fix/Invalid tickets).
- Cross-validated three ways (SQL counts, Python `repr()` checks, direct
  pandas checks) — all reconcile.

## Phase 6 — Feature engineering

- 203,290 rows; 12 base columns + 30 text PCA components.
- Priority normalized across two Jira vocabularies onto a 1-5 ordinal
  scale plus a separate `Unknown` category (its delay rate exceeds every
  numbered tier — folding it into the middle would have hidden signal).
- Type: rare categories (<100 issues) bucketed to `Other`.
- Workload features computed chronologically per assignee (expanding
  mean over that assignee's prior issues only).
- SBERT `all-MiniLM-L6-v2` over title+description, 9.3 min on GPU for
  203,290 issues; PCA to 30 components (42.5% cumulative variance),
  **fitted on training split only**.
- **Leakage tests pass:** max |correlation| with label 0.328
  (`assignee_prior_delay_rate`); no post-resolution columns; individual
  text components correlate weakly (max 0.129); manual expanding-mean
  recompute matched exactly on spot-checked assignees.

## Phase 7 — Static baseline (tuned)

| Metric | Value |
|---|---|
| Test AUC | **0.743** |
| F1 @ 0.5 | 0.650 |
| F1 @ train-tuned threshold (0.405) | 0.684 |
| Majority-class accuracy | 53.0% |

Report majority-class **accuracy**, not F1 — with the majority class now
"not delayed," an all-zeros prediction yields F1 = 0.000, which is
correct but meaningless as a baseline.

## Phase 8 — Dynamic model

**Setup:** 167,174 issues had ≥1 observation checkpoint (1,342,536
checkpoint rows, ~8.03 per issue). ~36k resolved issues never reached a
first checkpoint — they resolved faster than 10% of their group's
historical median. The dynamic model's evaluation population is therefore
narrower than the static model's, and becomes survivorship-biased at
later checkpoints by construction.

**Data-quality fixes found during the build:**
- `Change_Log` rows labelled `Change_Type='STATUS'` mix genuine workflow
  transitions with Resolution-field changes (one side null) — ~412k
  dropped, ~871k genuine transitions kept.
- ~9.6% of naive dwell segments ran negative because status logging
  continues past the recorded `Resolution_Date` — fixed by clipping.

**Pattern discovery:** DTW + hierarchical clustering (complete linkage;
average linkage produced a degenerate chaining artifact) on a stratified
4,000-issue training subsample, over `log1p(stall_ratio)` — the raw
signal is heavy-tailed (median 1.3, p99 ~37,000, max ~1.4M). K=4 from a
clean WSS elbow.

| Cluster | Size | Delay rate | Shape |
|---|---|---|---|
| 1 | 3,727 (93%) | 0.606 | Steady low stall, gentle drift — the unremarkable majority |
| 2 | 227 (5.7%) | **0.692** | Escalates continuously, never levels off |
| 3 | 8 (0.2%) | ~0.50 (n too small) | Sustained high plateau, sharp crash before resolution |
| 4 | 38 (0.95%) | 0.605 | Early-onset jump to a high plateau, stays elevated |

Cluster membership is stable: 88.9% of issues never change pattern label
once assigned; median stabilization at M3.

**Final tuned comparison** (identical per-checkpoint test populations):

| Checkpoint | N | %delayed | Static | Dynamic (pooled) | Dynamic (per-checkpoint) |
|---|---|---|---|---|---|
| M1 | 34,398 | 57.7% | **0.727** | 0.713 | 0.704 |
| M3 | 29,838 | 66.2% | 0.722 | **0.738** | 0.728 |
| M5 | 26,279 | 74.0% | 0.725 | **0.787** | 0.778 |
| M8 | 22,243 | 83.1% | 0.731 | **0.835** | 0.828 |
| M10 | 20,236 | 87.8% | 0.734 | 0.859 | **0.861** |

Crossover: **M3.**

## Phase 9 — Cross-project generalization (all 39 projects)

- **Cross-project:** mean AUC 0.710, median 0.718, IQR [0.668, 0.745]
- **Within-project:** mean 0.657, median 0.660, IQR [0.597, 0.699]
- Cross-project beats within-project on **30 of 38** projects (project 25
  skipped — its chronological test split landed on a single class).

Training-set size dominates: project 10's within-project split yields
~199 training rows versus ~203,000 available cross-project.

## Phase 10 — Explainability (SHAP)

- **Static:** text group #1 (0.895 summed across 30 components) >
  `assignee_prior_delay_rate` (0.381) > `Type_Normalized` > `n_links` >
  `Story_Point`.
- **Dynamic (M8):** `elapsed_minutes` #1 > `Project_ID` #2 > text group
  #3 (0.752) > … > `pattern_label` #12 of 14 (0.028).
- **Early vs. late importance shift:** `elapsed_minutes` is the only
  feature that gains importance late (+0.68); everything else, including
  `pattern_label` (−0.051) and `log1p_stall` (−0.014), loses importance.
  The stall and pattern signals matter most early-to-mid, not late.

---

## ⚠️ Interpretation cautions (state before someone asks)

**1. Use AUC, not F1 or accuracy, as the primary metric.** The surviving
test population grows steadily more delay-skewed (57.7% at M1 → 87.8% at
M10). A trivial "always predict delayed" baseline scores F1 ≈ 0.935 at
M10; the model scores ~0.937. AUC is not distorted by shifting base
rates; F1 and accuracy are.

**2. `elapsed_minutes` is not an independent progress measurement.**
Checkpoints are defined as `M_i = (i/10) × expected_duration_proxy`, so
within any checkpoint `elapsed_minutes` is *perfectly* proportional to
`expected_duration_proxy` (verified: r = 1.000000 at M5 and M8) — a
static group-level property, not individual progress. Its pooled
correlation with checkpoint index is only 0.288, so it is *not* simply
"checkpoint index in disguise" either. Neither is leakage; both are
knowable at prediction time. But "elapsed time dominates" ≠ "progress
dynamics dominate." An ablation confirms the split: removing
`elapsed_minutes` costs ~0.02-0.04 AUC, meaningful but not the mechanism.

**3. The test period is systematically more separable than the
validation period.** Test AUC exceeds validation AUC for every model
type (static 0.705→0.743; pooled untuned 0.699→0.741; pooled tuned
0.704→0.796). This is a property of the chronological split. Two
consequences: absolute AUC figures may be optimistic relative to another
time period, and hyperparameters selected on validation are not
guaranteed optimal for test.

**4. Phase 9's two setups differ on time.** Within-project uses a
chronological split (true forecasting). Cross-project trains on all other
projects across all time periods, including issues created after the test
project's test issues. So cross-project has both more data and an easier
temporal task. LOPO is routinely done this way, but the asymmetry should
be stated.

**5. Text SHAP importance is a group sum.** The 0.895 figure sums
mean|SHAP| across 30 PCA components; per component the average (~0.030)
is well below `assignee_prior_delay_rate` (0.381). The defensible claim
is "largest feature *group*," not "most important feature."

---

## Comparison to Kula et al. (ESEC/FSE '23)

**They never reported classification accuracy.** Their task was
regression over a continuous label (BRE). Their metrics were MAE and
**Standardized Accuracy**, where `SA = (1 − MAE/MAE_random) × 100` — the
percentage by which the model beats random guessing, *not* the share of
correct predictions. Do not cite their SA as accuracy, and do not let it
be compared against this project's accuracy.

| | Kula et al. | This project |
|---|---|---|
| Unit of analysis | 4,040 epics, 270 teams, one company (ING) | 203,290 issues, 39 open-source projects |
| Label | Continuous BRE | Binary (> project median) |
| Metrics | SA 66% → 92%; MAE 0.19 → 0.04 | AUC 0.713 → 0.859 across checkpoints |
| Dynamic vs. static | +12–57% SA | Crossover at M3; +0.125 AUC by M10 |
| Pattern benefit | +9–20% SA from milestone 3 on | +0.005–0.01 early, negligible late |
| Cluster sizes | 36% / 44% / 14% / 6% | 93% / 5.7% / 0.95% / 0.2% |
| K chosen | 4 (elbow) | 4 (elbow, independently) |

**Why patterns underperform here.** Kula's four patterns split the
population into four meaningfully-sized groups. Ours collapse into one
dominant cluster holding 93% of the subsample. A feature taking the same
value for the overwhelming majority of rows cannot discriminate,
regardless of model quality.

**They explicitly called for this replication.** From their Threats to
Validity: *"our data may not be representative of software projects in
other organizations and open source settings... Replication of our work
is needed to validate the findings in other settings and reach more
general conclusions."* And: *"The patterns in other organizations might
differ from the four patterns identified at the case company."*

---

## On accuracy (for the viva/defense)

Static-model accuracy is ~66-67% against a ~53% majority-class baseline
on a near-balanced label. **Accuracy above 90% would be a red flag on
this task, not a goal** — it would indicate either leakage (using
post-resolution information) or a lopsided label where predicting one
class always scores high while learning nothing. The ~88% accuracy
available at late checkpoints is a base-rate artifact (87.8% of surviving
issues are delayed there). AUC is the honest measure of skill and is
reported throughout.

---

## Methodological rigor (worth presenting explicitly)

- Chronological splits throughout — never random.
- All historical statistics (`expected_duration_proxy`, dwell baselines,
  assignee delay rates) computed on expanding windows from prior issues
  only.
- PCA fitted on training data only; DTW medoids derived from training
  data only; pattern assignment uses only each issue's own M1..Mt
  partial trajectory.
- Explicit leakage tests, reported.
- **A label-definition leak found and closed mid-project:** project
  medians were originally computed over the full resolved population
  including test issues. Quantified at 3.36% of test labels affected,
  then fixed by recomputing medians from train+validation only, and every
  affected phase re-run. Conclusions were compared across both labelings.
- Hyperparameters selected on a held-out validation split, per model,
  never on test.
- One intermediate conclusion was retracted after tuning revealed it as
  an artifact (see headline finding 7).

## Limitations

- The label is a relative proxy (slower than the project's own median),
  not an absolute SLA breach.
- Priority/Type normalization approximates across projects that don't
  share one Jira vocabulary; rare-category thresholds are
  population-dependent (Milestone has 29 issues in the resolved
  population and folds into `Other`).
- Dynamic evaluation covers 167,174 issues with ≥1 checkpoint, not all
  203,290, and is survivorship-biased at later checkpoints.
- Clustering ran on a 4,000-issue stratified subsample — a full pairwise
  DTW matrix over 142,303 training trajectories is computationally
  infeasible. Medoids from that subsample classify everything else.
- Per-checkpoint models were individually tuned for only 3 of 10
  checkpoints (M1, M5, M10) before the equivalence verdict.
- The tuned-vs-untuned comparison confounds hyperparameters with
  training-set size: the tuned model was refit on train+validation while
  the untuned one used train only (it needed validation for early
  stopping). The +0.13 AUC difference cannot be cleanly attributed to
  hyperparameters alone.
- Test-period separability exceeds validation-period separability (see
  caution 3).
- Phase 9 carries a temporal asymmetry (see caution 4).
- Small projects yield noisy per-project results; within-project test
  sets for the five smallest hold only 50-130 issues.
- The Bayesian logistic layer (Phase 8 stretch) was not implemented.

## Future work

- Bayesian logistic layer for calibrated uncertainty (Kula's approach).
- Time-respecting cross-project splits to remove Phase 9's asymmetry.
- Full DTW clustering at scale via approximate methods, to test whether
  the 93%-dominant cluster is an artifact of subsampling.
- More PCA components for text (42.5% variance at 30 was still climbing).
- Per-checkpoint tuning across all ten checkpoints.

## Framing for the paper

"We adapt Kula et al.'s dynamic delay-prediction framework to issue-level
TAWOS data, retaining the core mechanism — discovering recurrent temporal
delay patterns via trajectory clustering and feeding the evolving pattern
into dynamic prediction — while adapting the intermediate delay
representation (stall ratio in place of delayed story points), target
(binary Relative Resolution Delay in place of continuous BRE), milestone
definition (historically-estimated expected duration in place of planned
iterations), and predictors to the structure of TAWOS. We find that
dynamic prediction overtakes a static creation-time baseline at 30% of an
issue's expected duration and improves steadily thereafter (+0.125 AUC by
full expected duration); that this crossover is robust across label
definitions, model architectures, and hyperparameter regimes; that the
discovered delay patterns — unlike at epic level in Kula et al. —
contribute only marginally at issue level, with 93% of issues falling
into a single undifferentiated cluster; and that the approach generalizes
across projects, with cross-project training outperforming within-project
training on 30 of 38 projects." **Do not claim to have reproduced Kula
et al.**
