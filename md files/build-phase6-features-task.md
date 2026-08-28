# Task: Build the Phase 6 feature table

Give this to Claude Code in the project folder. It has local access to the
raw cache CSVs built earlier (`data/raw/issue.csv`,
`data/raw/change_log_status.csv`, `data/raw/issue_link.csv`,
`data/raw/comment.csv`) and can verify its own output.

**Crunch-timeline note: skip section 4 (SBERT text features) entirely for
now.** It's the most expensive part for the least presentation payoff. Only
build it if everything else here is done with time to spare.

## Step 1 — Apply the EDA-validated resolved-issue filter

Load `data/raw/issue.csv`. Apply this exact filter (already validated and
locked — do not modify the logic):

```python
COMPLETION_RESOLUTIONS = ['Fixed','Done','Complete','Completed','Resolved',
                          'Resolved Locally','Implemented','Deployed','Handled by Support']
NON_WORK_STATUS = ["Won't Fix", "Invalid", "Duplicate", "Cannot Reproduce", "Won't Do",
                   "Timed out", "Obsolete", "Inactive", "Not A Bug", "Rejected"]

base_resolved = df['Resolution_Date'].notna() | (df['Resolution'].notna() & (df['Resolution_Time_Minutes'] > 0))
genuine_completion = df['Resolution'].isin(COMPLETION_RESOLUTIONS) & ~df['Status'].isin(NON_WORK_STATUS)
res = df[base_resolved & genuine_completion].copy()
```

**Verify this produces exactly 203,290 rows.** If it doesn't, stop and
report the discrepancy rather than continuing — do not adjust the filter
to force a match.

## Step 2 — Build the label

```python
res['proj_median'] = res.groupby('Project_ID')['Resolution_Time_Minutes'].transform('median')
res['delayed'] = (res['Resolution_Time_Minutes'] > res['proj_median']).astype(int)
```

Verify the label is ~50/50 balanced overall (expected from a median split).

## Step 3 — Feature engineering (sections 6.1-6.3 of the implementation plan; skip 6.4)

**6.1 Base features:**
- `Project_ID` (keep as-is, used for grouping, not as a raw model feature
  unless one-hot encoded per project — use judgement, categorical is fine).
- Normalize `Priority`: merge the two Jira vocabularies (Trivial/Minor/
  Major/Critical/Blocker and Lowest/Low/Medium/High/Highest, including
  "- P#" suffixes) onto a single 1-5 ordinal scale. Keep blank/None/
  "To be reviewed" as a separate `Unknown` category rather than folding it
  into the middle of the scale — its delay rate is higher than any numbered
  tier (documented in EDA).
- Normalize `Type`: bucket categories with under 100 total issues into
  `Other`; keep everything else as its own category.
- `Story_Point`: keep as a feature as-is (14.3% populated — don't impute
  aggressively, let missing stay missing).

**6.2 Workload features (computed chronologically, using the FULL raw
`issue.csv` population — not just the resolved subset — as the basis for
computing history, since unresolved issues still count toward workload):**
- Assignee's number of previously-resolved issues at the time of the
  current issue's creation.
- Assignee's historical delay rate: `shift().expanding().mean()` over that
  assignee's own prior resolved issues only, sorted by `Creation_Date` —
  never include the current issue in its own history.
- Assignee's concurrent open issues at creation time (issues where
  `Creation_Date < T < Resolution_Date` or still unresolved, for that
  assignee, at the current issue's creation time).
- For unassigned issues (`Assignee_ID` null): leave the above three as NaN
  and add an explicit `is_unassigned` flag (1/0); set concurrent workload
  to 0 for these.

**6.3 Dependency features:**
- From `data/raw/issue_link.csv`: count of links per issue (`n_links`),
  filled to 0 where absent. If the link-type/description column allows
  distinguishing blocking vs. non-blocking links, also add a
  `n_blocking_links` count.

## Step 4 — Leakage test (do not skip)

Before saving anything, explicitly assert:
1. No feature column has correlation > 0.95 with `delayed` (a near-1.0
   correlation would indicate something derived from post-resolution
   information leaked in).
2. None of `Resolution_Date`, `Resolution_Time_Minutes`, or final `Status`
   appear anywhere in the final feature matrix (only in the separate label
   column, which is expected).
3. Spot-check the workload features: pick 2-3 issues, manually verify their
   `assignee_prior_delay_rate` was computed only from issues with an
   earlier `Creation_Date`.

Report pass/fail on each. If any fails, stop and report rather than
proceeding to save the table.

## Step 5 — Save and report

Save the final table to `data/features/phase6_feature_table.csv` — one row
per resolved issue (203,290 rows), including the `delayed` label column.

Report back: final row/column count, the leakage test results, delay rate
by normalized Priority and Type (compare against EDA's documented pattern —
urgent tickets should resolve faster, Epic/Milestone types should have
higher delay rates), and how many issues ended up with `is_unassigned = 1`
(expect roughly 42.9%, per EDA).
