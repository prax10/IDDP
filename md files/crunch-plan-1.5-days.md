# 1.5-Day Crunch Plan — Target: Genuine 60-70% Progress with Proof

Budget: ~20 hours, 2 people coding in parallel (~35-40 productive person-hours
after setup/debugging). Not attempting Phases 9-12 or the Bayesian layer —
that's honest and correct, not a shortfall. Full detail on every phase
referenced here is in `full-implementation-plan.md` and
`phase8-dynamic-model-plan.md`.

## Why not build everything
- Time doesn't fit Phases 9-12 + Bayesian at any reasonable quality.
- DTW/hierarchical clustering over the full ~160k training issues needs an
  infeasible pairwise distance matrix regardless of deadline — clustering
  MUST run on a representative subsample (a few thousand issues), with
  everything else classified against the resulting medoids. Build it this
  way because it's correct, not just because time is short.

## Setup (30 min, both)
- Create the repo. `.gitignore` everything under `data/` (raw CSVs and any
  embeddings — regenerable, don't commit them).
- Work in separate notebooks/scripts per person to avoid merge conflicts.
  Push small commits directly to `main` — skip branching/PR overhead.

## Window 1 (5h) — parallel, no dependency between tracks
- **Person A — Phase 6:** base features (Priority/Type normalization,
  Story_Point), chronological workload features (assignee prior delay rate,
  concurrent load), link counts. Skip SBERT text embeddings for now.
- **Person B — Phase 8 foundation:** reconstruct per-issue timelines from
  `change_log_status.csv`; build `expected_duration_proxy` (hierarchical
  historical median fallback).

## Window 2 (5h)
- **Person A:** finish + validate Phase 6 feature table — run the leakage
  assertion (no feature correlates ~1.0 with the label). Then Phase 7:
  chronological split, XGBoost, AUC/F1 vs. majority-class baseline.
- **Person B:** finish checkpoints (M1-M10) + `stall_ratio`. Run the 8E.0
  trajectory sanity check: 5-10 hand-picked issues + a random sample,
  plotted. Verify trajectories actually vary.

## Window 3 (5h)
- **Person B:** pick a representative subsample (stratified across a few
  projects/types, a few thousand issues). Run DTW distance + hierarchical
  clustering, pick K via elbow, extract medoids, write a short qualitative
  description of each discovered cluster.
- **Person A:** merge Phase 6 features onto the same subsample (for an
  apples-to-apples comparison). Start drafting slides with EDA + Phase 6 +
  Phase 7 results while waiting.

## Window 4 (5h) — buffer-heavy, expect debugging
- **Both:** dynamic pattern assignment on a held-out slice of the
  subsample; train dynamic XGBoost + pattern_label vs. without, on that
  subsample. Get one real comparison number.
- Assemble the presentation.

## What to actually show
1. EDA: resolved population, label definition, the three-way cross-check
   story (SQL / Python repr / direct pandas) — proof of rigor.
2. Phase 6: feature table + passing leakage test.
3. Phase 7: AUC/F1 vs. majority baseline — first hard number.
4. Phase 8 (reduced scale): trajectory plots showing real variation,
   discovered clusters with descriptions, dynamic-vs-static comparison on
   the subsample.
5. The locked full plan (Phase 8 doc + full implementation plan) as
   evidence of what's next and why it's scoped the way it is — design
   maturity counts as progress too.

## Framing for the "60-70%" claim
Argue it by substance, not phase count: the hardest and most novel piece
(discovered delay patterns via DTW feeding a dynamic model) is proven
working, even if not yet run at full scale or evaluated across all 10
checkpoints. That's a stronger claim than "N of 12 phases done."
