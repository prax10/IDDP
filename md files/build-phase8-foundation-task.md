# Task: Phase 8 foundation — timelines, checkpoints, stall ratio, sanity check

**Stop after Step 6 (the sanity check) and report back. Do not proceed to
DTW/clustering — that's a separate task, reviewed after these plots are
checked by a human.**

Inputs: `data/raw/change_log_status.csv` (timelines), and
`data/features/phase6_feature_table.csv` (the 203,290-row resolved
population + `delayed` label + `Project_ID`/normalized `Type`/normalized
`Priority`). Merge in `Creation_Date`/`Resolution_Date` from
`data/raw/issue.csv` if not already present in the feature table.

## Step 1 — Chronological three-way split (must align with Phase 7)

Sort the 203,290 resolved issues by `Creation_Date`. Split: **train** =
first 70%, **validation** = next 10% (70th-80th percentile), **test** =
last 20% (80th-100th percentile). The test set here must be the identical
set of issues as Phase 7's test set (Phase 7's train = this train +
validation combined; Phase 7's test = this test).

## Step 2 — Reconstruct per-issue timelines

From `change_log_status.csv`: for each `Issue_ID`, sort its rows by
`Creation_Date` to get an ordered sequence of `(From_String, To_String,
Creation_Date)` — this is the status history for that issue.

## Step 3 — `expected_duration_proxy` (leakage-safe, chronological)

For each resolved issue *i*, compute the hierarchical historical median of
`Resolution_Time_Minutes` among **other resolved issues whose
`Resolution_Date` is strictly before issue i's `Creation_Date`**, trying
these groupings in order and falling back when a group has fewer than 5
qualifying prior issues:
1. same `Project_ID` + `Type_norm` + `Priority_norm`
2. same `Project_ID` + `Type_norm`
3. same `Project_ID` + `Priority_norm`
4. same `Project_ID`
5. global median

**This must be an expanding/chronological computation, not a single
static median over the whole train set** — use the same pattern already
used for `assignee_prior_delay_rate` in Phase 6 (sort by time, `shift()` +
`expanding()`, grouped). A static train-wide median would leak later
training issues' outcomes into earlier issues' checkpoints — exactly the
kind of leakage this project has been careful to avoid everywhere else.

## Step 4 — Observation checkpoints

For each resolved issue, checkpoints `M_1..M_10 = (i/10) × expected_duration_proxy`
(elapsed minutes since `Creation_Date`). At each checkpoint, check whether
the issue was still unresolved at that elapsed time
(`Resolution_Date > Creation_Date + M_i`); if it had already resolved,
skip that checkpoint entirely (no row generated — variable-length
trajectories are expected). Issues still active past `M10` get
`overran_expected_duration = 1`.

## Step 5 — Dwell-time baseline and `stall_ratio`

For the status an issue is in at a given checkpoint, compute the
historical median dwell time in that status via the same hierarchical
fallback pattern (`status+project+type` → `status+project` →
`status+type` → `status`), again using only dwell periods that had fully
completed (issue moved out of that status) strictly before the current
issue's checkpoint time.

```
stall_ratio(i, t) = (t - last_status_change_time) / dwell_baseline(current_status)
```

`<1` = faster than typical for that status, `>1` = stalled.

## Step 6 — 8E.0 trajectory sanity check (stop here)

- Pick 5-10 hand-picked issues spanning different projects/types/outcomes,
  plus a random sample of ~30-50 training issues.
- Plot their `stall_ratio` trajectories (x-axis = checkpoint index,
  y-axis = stall_ratio). Save to `reports/phase8_trajectory_sample.png`.
- Check for: trajectories that visibly vary across issues and checkpoints
  (not all flat/identical); no impossible values (negative, infinite, NaN
  from a zero denominator); confirm for 2-3 spot-checked issues that every
  value used at checkpoint *t* only reflects information available as of
  that checkpoint's timestamp.

## Report back

Number of resolved issues with at least one checkpoint, average
checkpoints per issue, the sanity-check plot, and explicit confirmation
that `expected_duration_proxy` was computed via the expanding/chronological
method (not a static train-wide median). Flag anything that looks like
flat or suspiciously identical trajectories — that would mean the signal
isn't varying and needs fixing before clustering.
