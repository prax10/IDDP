# Task: Phase 8.3 — DTW distance + hierarchical clustering (subsample)

**Stop after Step 5 (cluster characterization) and report back. Do not
proceed to dynamic pattern assignment or dynamic XGBoost training — that's
the next task, after these clusters are reviewed.**

Input: `data/features/phase8_checkpoints.csv` (1,342,536 rows), filtered to
**training-set issues only** (use `data/features/phase8_split.csv` to
identify the 142,303 train issues — validation/test issues must never
influence clustering).

## Step 0 — Transform stall_ratio before anything else (required)

`stall_ratio` is heavy-tailed (median 1.3, p99 ~37,000, max ~1.4M, per the
Phase 8 foundation report). Raw values would let a handful of extreme
trajectories dominate every DTW distance calculation regardless of K.
Apply `log1p(stall_ratio)` (i.e. `log(1 + stall_ratio)`) to every value
before building trajectories or computing any distance. Do this
transform-only — do not clip/winsorize on top of it unless log1p still
leaves extreme outliers dominating (check the distribution after
transforming; report if a cap is still needed).

## Step 1 — Select a representative subsample

Full-scale DTW + hierarchical clustering over 142,303 training trajectories
is not computationally feasible (pairwise distance matrix would need
~10^10 comparisons). Select a stratified subsample of **3,000-5,000**
training issues, stratified across `Project_ID` and normalized `Type` so
small projects/types aren't invisible in the sample. Use a fixed random
seed and report it.

## Step 2 — Build trajectories

One `log1p(stall_ratio)` sequence per sampled issue, in checkpoint order
(M1, M2, ... up to wherever that issue's trajectory ends — variable
length is expected, this is why DTW is used instead of a fixed-length
distance metric). Checkpoints where the issue hadn't started accumulating
history yet (`insufficient_history`, per the locked design — effectively
M1-M2 for issues without enough prior status-change data) should be
excluded from the trajectory rather than treated as a real observation.

## Step 3 — DTW distance + hierarchical clustering

- Compute pairwise DTW distance across the subsample (e.g. `dtaidistance`
  or `tslearn`'s DTW implementation — either is fine, report which was
  used and its version).
- Hierarchical clustering (average or complete linkage) on the resulting
  distance matrix.
- Pick K via the elbow/WSS method — plot WSS vs. K for K=2 through K=8,
  save to `reports/phase8_elbow.png`, and pick the elbow point. Do not
  assume K=4 (Kula et al.'s value) applies here — TAWOS may need a
  different K, or may not show a clean elbow at all (report honestly if
  the elbow is ambiguous, and pick a reasonable K with justification if so).

## Step 4 — Medoids (not centroids)

For the chosen K, extract the DTW medoid of each cluster — the actual
training trajectory with the smallest total DTW distance to every other
member of its cluster (not a synthetic average, which isn't well-defined
under DTW). Save the medoid trajectories (as arrays/CSV) to
`data/features/phase8_medoids.csv` or similar — these are what validation
and test issues get classified against in the next task, so they need to
persist.

## Step 5 — Characterize clusters and report back

For each cluster: size (how many of the subsample fell into it), the
delay rate of its members (using the `delayed` label from the Phase 6
feature table, joined on issue ID) — does the cluster's delay rate differ
meaningfully from the overall 50%?, and a qualitative one-sentence
description of the trajectory shape (e.g. "steady low stall throughout,"
"early stall then recovery," "persistent late-stage stall," "stall spikes
near the end") based on actually looking at the medoid and a few member
trajectories — descriptions must come from what TAWOS actually shows, not
be copied from Kula et al.'s four named patterns.

Report back: chosen K and why, per-cluster size/delay-rate/description
table, the elbow plot, and whether the log1p transform alone was
sufficient or an additional cap was needed.
