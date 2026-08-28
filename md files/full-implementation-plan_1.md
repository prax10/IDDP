# Intelligent Dynamic Delay Prediction — Full Implementation Plan (Phases 1–12)

Consolidated plan reflecting the original development guide plus everything since
locked in EDA and Phase 8. This supersedes the original guide's Phase 2 and
Phase 8 sections wherever they conflict — the decisions below are what was
actually validated on the data. Timeline assumes a 4–6 week, 4-person team;
week ranges are rough scheduling anchors, not hard deadlines.

Full detail for Phases 1–5 lives in `eda-conclusions.md`; full detail for
Phase 8 lives in `phase8-dynamic-model-plan.md`. This document is the
end-to-end map — condensed for Phases 1–5 and 8, full for everything else.

---

## Scope discipline (applies to the whole project, not just Phase 8)

🔴 **Core chain — must ship, in this order:**
- Phases 1–5: EDA (complete).
- Phase 6: leakage-safe features, **including explicit leakage tests**
  (e.g. assert no feature correlates ~1.0 with the label by construction,
  and audit every hierarchical fallback statistic for chronological
  safety) — not just leakage-safe design, the test itself.
- Phase 7: static XGBoost baseline, **including comparison against a
  majority-class baseline** — this is what answers "is the model learning
  anything beyond the dominant class," and it's cheap enough that skipping
  it has no excuse.
- Phase 8 core: expected-duration proxy, M1–M10 observation checkpoints,
  stall ratio, **DTW distance + hierarchical clustering**, empirical K
  selection, medoids, dynamic pattern assignment, dynamic XGBoost,
  checkpoint-wise (M1–M10) evaluation.
- Phase 9: meaningful cross-project generalization on a representative
  subset of projects.
- Phase 10: SHAP explainability.

🟡 **Stretch — do only if the core chain is solid with time left:** Bayesian
logistic layer (Ph.8), exhaustive leave-one-project-out sweep across all 39
projects (Ph.9), Global Iterative baseline, pattern stability experiment,
Streamlit prototype (Ph.12).

🟢 **First things to cut under time pressure:** reassignment suggester
(Ph.11), prototype polish beyond a bare demo, elaborate Bayesian analysis,
exhaustive statistical testing, full 39-project LOPO if it becomes
computationally expensive.

**DTW pattern discovery is non-negotiable.** If implementation time becomes
constrained, reduce secondary experiments rather than replacing DTW
clustering with raw stall-ratio features fed directly into XGBoost.
Removing DTW would change the central research contribution from
Kula-inspired delay-pattern discovery to a conventional temporal
issue-delay classifier — a materially weaker and different claim.

A defensible paper exists even if everything 🟡 and 🟢 above is dropped —
but not if DTW pattern discovery is cut from the 🔴 list.

---

## Phase 1–5 — EDA (COMPLETE, locked)

**Status:** done and cross-validated (SQL counts, Python `repr()` checks,
direct pandas checks all reconcile). Full writeup: `eda-conclusions.md`.

Locked decisions carried into every later phase:
- **Resolved population:** 203,290 of 458,232 issues. Definition requires
  `Resolution_Date` not null OR (`Resolution` not null AND
  `Resolution_Time_Minutes > 0`), AND `Resolution` in a genuine-completion set,
  AND `Status` not in a non-work set (Won't Fix/Invalid/Duplicate/etc.) — this
  last condition matters because some projects (e.g. project 28) recycle
  `Resolution='Done'` as a generic closure flag.
- **Label:** `delayed = Resolution_Time_Minutes > per-project median`. 50/50
  balance by construction.
- **Sanity checks passed:** delay rate varies sensibly by Priority and Type;
  most issues have 2+ status changes (dynamic prediction is feasible); links
  and text are populated enough to be usable features.
- **Priority:** two Jira vocabularies merged onto a 1–5 ordinal scale, with
  blank/unclear kept as a separate `Unknown` category (higher delay rate —
  don't fold it into the middle of the scale).
- **Type:** rare categories (<100 issues) bucketed into `Other`.
- **Known limitations to state in the paper:** label is a relative proxy, not
  an absolute SLA breach; smallest projects (10, 6, 9, 2, 35 — all under ~700
  resolved issues) will be noisy in Phase 9; unassigned issues (42.9%) are a
  structural feature of the data, not a data-quality problem.

**Paper section fed:** Method (data + label definition), Threats to Validity
(label proxy, cross-project vocabulary differences).

---

## Phase 6 — Feature engineering

**Goal:** one row per resolved issue (for Phase 7) and one row per snapshot
(for Phase 8), using only information available at prediction time.

**Leakage rule (absolute), with one necessary distinction:** resolution
information (`Resolution_Date`, `Resolution_Time_Minutes`, final `Status`)
may be used to construct the **target** and **historical reference
statistics** (per-project medians, expected-duration proxies, dwell-time
baselines) — provided only information chronologically available before the
prediction point is used. It must **never** enter the feature vector for the
issue currently being predicted. This is the single most common way a
capstone gets invalidated — test for it explicitly (e.g. assert no feature
column correlates ~1.0 with the label by construction, and audit every
hierarchical fallback statistic for chronological safety, not just the raw
columns).

- **6.1 Base features:** `Project_ID`, normalized `Priority`, normalized
  `Type`, `Story_Point` (14.3% populated — include but expect XGBoost to lean
  on it lightly; do not impute aggressively). One-hot / native categorical
  encoding depending on whether you use XGBoost's categorical support.
- **6.2 Workload features (per assignee, computed chronologically):**
  prior resolved-issue count, prior delay rate (`shift().expanding().mean()`
  over that assignee's own history only — never include the current issue),
  concurrent open issues at creation time. Unassigned issues: leave
  assignee-history features as `NaN` (XGBoost handles this natively) plus an
  explicit `is_unassigned` flag; concurrent workload = 0 for these.
- **6.3 Dependency features:** link count from `Issue_Link`, split into
  blocking vs. non-blocking where the `Description` field allows; optionally,
  whether any blocking issue was itself delayed as of the current issue's
  creation (chronologically safe check).
- **6.4 Text features:** SBERT (`all-MiniLM-L6-v2`) over title + description,
  reduced to ~30 dims via PCA. Encode only the resolved subset, batch it,
  cache to disk (`data/*.parquet`) — 203k texts is heavy on a laptop.
- **Caching:** every expensive step (SQL pulls, SBERT encoding, workload
  aggregation) should be cached to parquet so re-running downstream phases
  doesn't re-hit MySQL or re-encode text.

**Paper section fed:** Method (feature engineering), and this is where the
leakage-safety argument for the whole paper gets made explicit.

---

## Phase 7 — Static model (baseline)

**Goal:** a one-shot predictor — features known at creation → `delayed`.
Must work end-to-end before Phase 8 starts; this is also your paper's
baseline for every later comparison.

- **7.1 Chronological split** (not random) — sort by `Creation_Date`, train
  on the older ~80%, test on the newer ~20%. State explicitly in the paper
  why: predicting delay is predicting the future, so a random split leaks
  future issues into training and inflates every score.
- **7.2 Model:** `XGBClassifier` (n_estimators=400, max_depth=6,
  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8). Justify gradient
  boosting over deep learning for tabular data in the paper (cite
  Grinsztajn et al. 2022 or equivalent from your lit review).
- **7.3 Evaluate:** AUC, precision/recall/F1 (binary), confusion matrix, and
  a majority-class baseline for comparison — with a 50/50 label, majority
  baseline is close to chance, so this mainly matters if you report on an
  imbalanced subset (e.g. per-project splits in Phase 9).

**Paper section fed:** Results (baseline numbers every later model is
compared against), Method (why gradient boosting, why chronological split).

---

## Phase 8 — Dynamic model (novelty; LOCKED design)

Full detail in `phase8-dynamic-model-plan.md` — condensed here.

**Core mechanism:** adapts Kula et al. (ESEC/FSE '23) — trajectory
clustering feeding a dynamic model — to issue-level TAWOS data. Not a
literal replication (different unit of analysis, label, milestones,
predictors, and Bayesian likelihood — full comparison table in the locked
doc). State this explicitly in the paper.

- **8.1 Data prep:** reuse Phase 1–6 resolved population and label;
  chronological train/validation/test split; reconstruct per-issue timelines
  from `Change_Log`.
- **8.2 Real-time lifecycle representation:**
  - `expected_duration_proxy` — hierarchical historical median (by
    project+type+priority, falling back to project+type → project+priority →
    project → global), computed only from issues resolved before the
    current one.
  - 10 **observation checkpoints** `M_i = i/10 × expected_duration_proxy`,
    i = 1..10, corresponding to 10%–100% of the historical expected
    duration. These are checkpoints at which the issue is *observed*, not
    guaranteed milestones it passes through — an issue can resolve before
    M5. **M10 is the final fixed checkpoint.** Issues still active after M10
    are flagged `overran_expected_duration` rather than given additional
    M11+ trajectory points — checkpoint count stays bounded regardless of
    how far an issue overruns.
  - Snapshot gating: skip a checkpoint if the issue already resolved before
    it; variable-length trajectories are expected and DTW handles this.
  - **Stall ratio** (the actual trajectory signal, deliberately independent
    of the checkpoint formula): time since last status change ÷ historical
    dwell-time baseline for the current status (same hierarchical fallback
    pattern). `<1` = faster than typical, `>1` = stalled.
  - ⚠️ Leakage trap to test for explicitly: the dwell-time baseline must be
    computed only from dwell periods completed before the prediction time —
    never from the full dataset.
- **8E.0 Trajectory sanity check (run before scaling up — do not skip):**
  before running DTW/clustering at scale: (1) select 5–10 manually
  inspectable issues representing different histories; (2) also select a
  small random sample from the training set — hand-picked cases show
  expected behavior, random ones show whether the signal actually varies in
  the data rather than only in cases chosen to look right; (3) plot/inspect
  their stall-ratio trajectories; (4) verify trajectories actually vary
  across issues and checkpoints; (5) check for impossible values, missing
  denominators, extreme outliers, and suspiciously identical trajectories;
  (6) confirm each trajectory point only uses information available at that
  checkpoint. **Do not proceed to large-scale DTW clustering until this
  check passes.**
- **8.3 Delay-pattern discovery:** DTW distance + hierarchical clustering
  over training-set stall-ratio trajectories only. Pick K empirically
  (elbow/WSS) — don't assume Kula's K=4. Represent clusters by DTW medoid
  (not centroid).
- **8.4 Dynamic pattern assignment:** M1–M2 = `insufficient_history`; from M3
  on, classify the partial trajectory against train-derived medoids only.
  Validation/test never redefine clusters.
- **8.5 Models:** dynamic XGBoost + pattern label (primary); Bayesian
  logistic regression (Bernoulli/logit — 🟡 stretch, needs PyMC).
- **8.6 Evaluation:** AUC/F1/precision/recall/Brier/log-loss at each
  checkpoint M1–M10 — the accuracy-improves-over-time plot is a headline
  result.

**Paper section fed:** Method (core novelty section), Results (headline
dynamic-accuracy-over-time plot), Related Work (direct comparison to Kula
et al.).

---

## Phase 9 — Generalization experiment (core contribution)

**Research question:** does the model transfer across projects, or is delay
prediction inherently project-specific?

🔴 Core: demonstrate cross-project generalization on a representative
subset of projects. 🟡 Stretch: exhaustive leave-one-project-out evaluation
across all 39 projects.

- **9.1 Leave-one-project-out:** for the chosen subset (mix of large/small
  projects; extend to all 39 as the stretch goal), train on the rest, test
  on the held-out one. Record AUC/F1 per project.
- **9.2 Report:** per-project table AND macro-mean AND median/IQR — do not
  let large projects dominate the average. Flag the small projects (10, 6,
  9, 2, 35 — under ~700 resolved issues) explicitly as noisy rather than
  omitting them.
- **9.3 Compare:** within-project performance (Phase 7 style, per project)
  vs. cross-project (this phase) — the drop is the "generalization gap," and
  is publishable either way (transfer works, or transfer fails and that's a
  finding about delay prediction being project-specific).
- Optional stretch (if time remains): repeat leave-one-project-out with the
  *dynamic* model (Phase 8) instead of static, to see if temporal
  information also generalizes.

**Paper section fed:** Results (the headline generalization table/plot),
Discussion (what the gap implies).

---

## Phase 10 — Explainability (SHAP)

- **Global:** `shap.summary_plot` on the static and dynamic models — which
  features drive delay predictions overall (paper: "workload and
  dependencies are the top delay drivers," or whatever TAWOS actually
  shows).
- **Local:** `shap.plots.waterfall` for individual issues — why *this* task
  is flagged. Feeds Phase 11's honesty check if you get there.
- Also worth reporting: does feature importance shift across Phase 8
  checkpoints (e.g. workload matters early, stall ratio matters late)?

**Paper section fed:** Results/Discussion (interpretability), and it's your
strongest material for the viva.

---

## Phase 11 — Reassignment suggester (🟢 stretch, cut first if short on time)

On-demand only, not automatic. For a flagged issue: swap in each candidate
developer's workload/history features, re-run the model, rank by predicted
risk, suggest the lowest-risk assignee. **Honesty check via SHAP:** if the
delay driver is a blocking dependency rather than the assignee, say
reassignment won't help instead of suggesting a swap. Build last — depends
on Phases 6, 7/8, and 10 all working.

**Paper section fed:** optional "applications" subsection; not required for
the core research contribution.

---

## Phase 12 — Streamlit prototype (🟡 stretch)

Thin wrapper around the trained model — no new modeling work. Views: project
risk dashboard, filterable task list, click-to-expand SHAP, developer
workload, reassignment suggester (if Phase 11 is built). Scope discipline:
no live Jira sync, no task editing, no notifications. A bare version (one or
two working tabs) is enough to demo; polish is the first thing to cut.

**Paper section fed:** not typically a paper section itself, but useful as a
demo for the viva/defense.

---

## Rough week-by-week anchor (4–6 weeks, tight)

- **Week 1:** Phase 6 (feature table + snapshot table) built and leakage-
  tested; Phase 7 static baseline running end-to-end.
- **Week 2:** Phase 8 core (8A–8J from the locked plan: split, timelines,
  checkpoints, stall ratio, DTW clustering, dynamic XGBoost).
- **Week 3:** Phase 9 generalization experiment; start Phase 10 SHAP in
  parallel once Phase 7/8 models are stable.
- **Week 4:** Finish Phase 10; begin paper writeup (Method + Results
  sections can be drafted as each phase closes, not all at the end).
- **Weeks 5–6 (if available):** Bayesian layer (8K/8L) and/or prototype
  (Phase 12) as stretch; otherwise, this time goes entirely to paper
  writing, polishing figures, and the generalization write-up (usually the
  most reviewer-scrutinized part).

If the timeline compresses further, drop straight to the 🟢 cut list above
— the core chain through Phase 10 is what makes this a complete, defensible
paper on its own.

---

## Paper structure ↔ phase mapping

- **Introduction / Motivation:** the gap this fills (dynamic, explainable,
  cross-project delay prediction) — no phase-specific work, drafted from
  your lit review and Kula comparison.
- **Related Work:** Kula et al. comparison table (from Phase 8 doc) is the
  centerpiece; cite tabular-ML-over-deep-learning literature for the Phase 7
  model choice.
- **Method:** Phases 1–2 (data + label), Phase 6 (features + leakage
  argument), Phase 7 (static model + chronological split rationale), Phase 8
  (dynamic model — this is the novelty section and should be the longest).
- **Results:** Phase 7 baseline numbers, Phase 8 checkpoint-accuracy curve,
  Phase 9 generalization table, Phase 10 SHAP plots.
- **Discussion:** what the generalization gap means; what SHAP reveals about
  delay drivers; honest comparison to Kula et al.'s reported numbers (not a
  head-to-head, since the tasks differ structurally).
- **Threats to Validity / Limitations:** label is a relative proxy (not
  absolute SLA breach), Priority/Type normalization approximates across
  inconsistent per-project vocabularies, small-project noise in Phase 9,
  Bayesian layer is an adaptation not a replication of Kula et al.
- **Conclusion + Future Work:** Phase 11 (reassignment) and full Phase 12
  prototype belong here if not fully built — "the model supports these
  applications; a full implementation is future work" is a legitimate and
  common framing.

---

## Status

🟢 Design locked → start 8A: validate the resolved population, target
construction, chronological split, and leakage assumptions against the
actual TAWOS data.
