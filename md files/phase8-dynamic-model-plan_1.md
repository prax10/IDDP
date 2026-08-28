# Phase 8 — Dynamic Model Plan (LOCKED — final version)

Base paper: Kula, Greuter, van Deursen, Gousios, "Dynamic Prediction of Delays in Software
Projects using Delay Patterns and Bayesian Modeling," ESEC/FSE '23 (TU Delft + ING). Code/data
under NDA — not reproducible directly. This adapts their core mechanism (trajectory clustering
feeding a dynamic model) to issue-level TAWOS data. Not a literal replication — state this
explicitly in the thesis (see closing framing at the bottom).

## Core design principle (state this explicitly in the methodology writeup)

**The milestone determines WHEN we observe the issue. The stall ratio provides WHAT we observe
at that point. The two must never be the same quantity** — an earlier draft of this plan defined
both identically, which would have forced every issue's trajectory into an uninformative straight
line `[0.1, 0.2, ..., 1.0]`, with nothing for DTW clustering to find. Caught and fixed before
implementation.

## Global temporal leakage rule

No feature, statistic, trajectory, pattern label, normalization value, or model input may use
information occurring after the current prediction milestone. Applies to: expected-duration
calculations, dwell-time baselines, workload features, text/comment features, dependency/link
features, pace/stall trajectories, cluster/pattern assignment, preprocessing statistics, model
training, validation and test construction.

## Comparison with Kula et al. (state explicitly, not a replication)

| | Kula et al. | This project |
|---|---|---|
| Unit of analysis | Epics (4,040, 270 teams) | Individual issues |
| Label | Continuous BRE, 42% exactly zero | Binary "Relative Resolution Delay" (> per-project median) — explicitly NOT equivalent to BRE |
| Milestones | 10, by completion rate = iterations done / iterations **planned** (real-time-computable) | 10, by elapsed time / historically-estimated `expected_duration_proxy` (real-time-computable) |
| Intermediate signal | DSP (delayed story points), non-cumulative | Stall ratio (time since last status change / historical dwell baseline) |
| Predictors | 13 team/org-level metrics (many have no TAWOS equivalent, e.g. dev-age-ing, changed-leads) | Issue/assignee-level (priority, type, workload, links, SBERT text) |
| Model | Zero-Inflated Beta regression (3 linked GLMs), Stan/HMC | Bayesian logistic regression (Bernoulli/logit), PyMC |
| Eval | MAE, Standardized Accuracy, 90% credible interval width | AUC, F1, Brier score, log loss, credible interval width |

## 8.1 Data preparation
- Valid completion population: use the EDA-validated definition (COMPLETION_RESOLUTIONS +
  NON_WORK_STATUS filter), not raw "Resolution is not null."
- Target: `Delayed_i = 1 if Resolution_Time_i > median(Resolution_Time for Project_i) else 0`,
  computed without the issue itself or future/test information. Refer to this as "Relative
  Resolution Delay" in the thesis, not as Kula's BRE.
- Temporal train / validation / test split (chronological, not random). Test set completely
  unseen during feature engineering, expected-duration estimation, trajectory construction,
  clustering, K selection, and medoid construction.
- Reconstruct per-issue timelines from `Change_Log` (order events, resolve duplicates, compute
  elapsed time and state at any point).

## 8.2 Real-time lifecycle representation
- `expected_duration_proxy`: hierarchical historical median (project+type+priority → project+type
  → project+priority → project → global), computed only from issues resolved before the issue/time
  in question.
- Observation checkpoints (real-time-computable, NOT leaky): `M_i = i/10 × expected_duration_proxy`.
  Can exceed 1.0 — intentional, signals the issue has run past its historically expected duration.
  Note the terminology: M1–M10 are checkpoints at which we *observe* the issue, not guaranteed
  lifecycle milestones the issue passes through — an issue can resolve before M5, for example.
  Casual references to "M1–M10" throughout the rest of this doc and in implementation mean
  "checkpoint 1 through 10," not "milestone reached."
- Snapshot gating: at each checkpoint M_i, check whether the issue was still active (not yet
  resolved) at that elapsed time. If already resolved, do not generate that snapshot — trajectories
  are naturally variable-length; DTW handles this. Cap tracked checkpoints at M10; for issues that
  run past expected duration, add a binary `overran_expected_duration` flag rather than open-ended
  M11+ checkpoints (keeps trajectory length bounded and comparable; revisit only if a large share
  of issues badly overrun).
- Stall ratio (the trajectory signal, deliberately independent of the checkpoint formula):
  `stall_ratio_{i,t} = (t - last_status_change_i) / median_historical_dwell(current_status)`,
  where the dwell baseline uses the same hierarchical fallback (status+project+type → status+project
  → status+type → status), computed only from dwell periods completed before the prediction time.
  `<1` = less time in current state than typical; `≈1` = typical; `>1` = stalled longer than typical.
  **Implementation warning:** the dwell-baseline denominator is the easiest place to accidentally
  break the leakage-free design. `df.groupby('Status')['dwell_time'].median()` computed over the
  whole dataset looks harmless and will pass any ordinary unit test while silently leaking future
  information into every checkpoint before the split point. When implementing 8E, explicitly test
  that the dwell baseline used at prediction time for issue *i* was computed only from dwell periods
  that had already completed as of that issue's current checkpoint — never from the full TAWOS
  table, and never from the issue's own eventual dwell period.
- Story_Point/Sprint_ID/Estimation_Date audit: time-boxed investigatory pass — check if Story_Point
  (14.3% populated) is reliable enough in some subset (by project/type/period) to support a more
  Kula-like workload signal there. If not sufficiently reliable, proceed with stall ratio as the
  primary signal (already decided) — don't let this become open-ended.

## 8.3 Delay-pattern discovery
- Build one stall-ratio trajectory per training issue (variable length, full history, train-only).
- DTW distance + hierarchical clustering. Elbow/WSS method to pick K empirically — do not assume
  Kula's K=4 applies to TAWOS.
- Represent each cluster by its DTW medoid (an actual training trajectory with smallest total DTW
  distance to other cluster members) — NOT a Euclidean centroid, which isn't well-defined for
  DTW-based clusters.
- Characterize clusters qualitatively from what TAWOS actually shows (e.g., steady progress, early
  stall → recovery, persistent stall, late stall) — descriptions must come from the real TAWOS
  clusters, not be copied from Kula's four patterns.

## 8.4 Dynamic pattern assignment
- A pattern may only be assigned using information available at the current milestone: M1–M2 =
  `insufficient_history`; from M3 on, classify the partial trajectory (M1..Mt only) by DTW distance
  to the TRAIN-derived medoids.
- Validation and test trajectories are classified against TRAIN medoids only — never used to
  redefine clusters. Validation may be used to choose K or tune hyperparameters; test is touched
  once, for final evaluation. If K is finalized using validation, optionally retrain clustering on
  TRAIN+VALIDATION before final test evaluation — only if explicitly documented as part of the
  protocol; the simpler strict version (train-only medoids throughout) is preferred for a capstone.

## 8.5 Predictive models
- Static XGBoost (existing Phase 7 baseline).
- Dynamic XGBoost + pattern_label (primary ML model).
- Bayesian logistic regression (Bernoulli likelihood, logit link — the correct structural analogue
  of Kula's Zero-Inflated Beta given a binary label, not continuous BRE), with and without
  pattern_label as an ablation.
- Avoid full model refit at every milestone unless necessary — the requirement is a valid posterior
  predictive result conditioned on information available at that point, not literally 10 separate
  full refits if that's computationally prohibitive.

## 8.6 Dynamic evaluation
Evaluate across M1–M10: ROC-AUC, F1, precision, recall, Brier score, log loss, and 90% credible
interval width for P(delayed). Do not assume the credible interval narrows monotonically — report
what TAWOS actually shows.

## 8.7 Generalization and stretch experiments
- Cross-project generalization (leave-one-project-out): report per-project AND macro-mean AND
  median/IQR — don't let large projects dominate the interpretation (ties back to the small-project
  noise issue already flagged in EDA: projects 10, 6, 9, 2, 35 are all under ~700 resolved issues).
- Global Iterative baseline (repeatedly reapply the static model with updated features, no
  retraining, no pattern) — cleaner isolation of "does more information over time help at all."
- Pattern stability experiment (optional): track how often an issue's assigned pattern changes
  across milestones; report first milestone at which it stabilizes.

## 8.8 Interpretation
- SHAP for XGBoost models (does pattern_label contribute meaningfully; does feature importance
  shift across milestones).
- Posterior coefficient distributions + credible intervals for the Bayesian model.
- Key comparisons, correctly isolated:
  - Global Iterative vs. Dynamic → does dynamic temporal/pattern-aware modeling improve over
    repeatedly reapplying the global model?
  - Dynamic WITHOUT pattern vs. WITH pattern → does the discovered delay-pattern representation
    itself improve prediction? (mirrors Kula's RQ2)
  - Dynamic XGBoost vs. Dynamic Bayesian → does Bayesian modeling improve probability estimation
    and uncertainty quantification?

## Scope: locked core vs. stretch vs. cut-first

🔴 **CORE (load-bearing chain, must implement):** 8.1–8.4 in full; real-time milestones +
expected-duration proxy + stall ratio; trajectory construction, DTW, hierarchical clustering,
empirical K, medoids, leakage-free dynamic assignment; static XGBoost; dynamic XGBoost + pattern;
Bayesian logistic ± pattern; milestone-wise (M1–M10) evaluation including basic calibration
(Brier score, log loss).

🟡 **STRETCH (implement if time permits):** Global Iterative baseline; full cross-project
leave-one-project-out analysis; pattern stability experiment; extensive statistical significance
testing.

🟢 **Cut in this order if time runs short:** (1) pattern stability, (2) full Global-Iterative
experiment, (3) full six-way model hierarchy comparison, (4) extensive statistical testing,
(5) full leave-one-project-out analysis. Never cut: DTW pattern discovery, dynamic pattern
assignment, dynamic XGBoost, Bayesian ± pattern ablation, milestone-wise evaluation — these are
what make the work genuinely Kula-inspired rather than "another issue-delay classifier."

## Implementation sequence

8A Validate data + target → 8B Temporal split → 8C Reconstruct timelines → 8D Build real-time
milestones → 8E Build stall-ratio trajectories → 8F DTW + clustering → 8G Train-derived medoids →
8H Dynamic pattern assignment → 8I Static XGBoost → 8J Dynamic XGBoost + pattern → 8K Bayesian
logistic − pattern → 8L Bayesian logistic + pattern → 8M Milestone-wise evaluation → 8N Calibration
+ uncertainty → 8O Cross-project / stretch experiments → 8P Interpretation + final comparison.

Do not start PyMC (8K/8L) before 8A–8J are working and validated.

## Final framing for the thesis

"We adapt Kula et al.'s dynamic delay-prediction framework to issue-level TAWOS data. We retain
the core mechanism of discovering recurrent temporal delay patterns via trajectory clustering and
incorporating the evolving pattern into dynamic prediction, while adapting the intermediate delay
representation (stall ratio in place of DSP), target variable (binary Relative Resolution Delay in
place of continuous BRE), milestone definition (historically-estimated expected duration in place
of planned iterations), predictors, and Bayesian likelihood (Bernoulli/logit in place of
Zero-Inflated Beta) to the structure and limitations of TAWOS." Do not claim to have reproduced
Kula et al.

## Status

🟢 Design locked → start 8A.
