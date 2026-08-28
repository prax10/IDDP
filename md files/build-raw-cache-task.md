# Task: Build the TAWOS raw-data cache

Give this file to Claude Code in the project folder you just created. It has
full local access to MySQL, so it can run the extraction, verify the output
against the live database, and report back — something the cloud session
that wrote this brief cannot do itself.

## Context

This is a capstone project predicting issue-resolution delay on the TAWOS
Jira dataset (MySQL `TAWOS_DB`, connection string
`mysql+pymysql://root:root%40123@localhost/TAWOS_DB`). Known reference
numbers from completed EDA, to sanity-check against — **do not use these to
filter anything in this task**, they're for verification only:
- Total issues in `Issue`: 458,232
- Distinct projects: 39
- Resolved-issue population (a later, separate filtering step — not part of
  this task): 203,290

## Goal

Build a narrow, disk-efficient local cache of exactly the fields later
phases need, for the **full row population** (not just resolved issues —
downstream workload/concurrency features need issues that are still open or
unresolved too, not just finished ones). This replaces repeated MySQL
queries during feature engineering and modeling.

## Step 1 — Schema check (do this first, before writing the extraction queries)

Run `DESCRIBE Issue;`, `DESCRIBE Change_Log;`, `DESCRIBE Issue_Link;`, and
`DESCRIBE Comment;` against `TAWOS_DB`. Print the columns for each. The
column lists below for `Issue` and `Change_Log` are confirmed from prior
work on this project; the ones for `Issue_Link` and `Comment` are **not**
confirmed beyond the columns named — check the real schema and adjust the
SELECTs in Step 2 accordingly (e.g. there is likely a linked-issue-ID column
on `Issue_Link` beyond `Description`, and `Comment` may have more than
`Issue_ID`/`Creation_Date` worth keeping — use judgment, but don't add
columns nothing downstream will use).

## Step 2 — Extract and cache

Write a Python script (`build_raw_cache.py` in the project root) that pulls,
for the **full row population** of each table:

- **Issue**: `ID, Project_ID, Priority, Type, Status, Resolution, Resolution_Time_Minutes, In_Progress_Minutes, Story_Point, Total_Effort_Minutes, Assignee_ID, Creator_ID, Reporter_ID, Sprint_ID, Creation_Date, Resolution_Date, Estimation_Date, Title, Description_Text`
- **Change_Log**: `Issue_ID, Change_Type, From_String, To_String, Creation_Date`, filtered to `WHERE Change_Type = 'STATUS'` (this is what drives Phase 8's timeline/checkpoint reconstruction — don't pull other change types unless a later task explicitly asks for them)
- **Issue_Link**: `Issue_ID`, `Description`, plus whatever the real linked-issue-ID column is called (confirm in Step 1)
- **Comment**: `Issue_ID`, `Creation_Date` (plus anything else confirmed useful in Step 1)

Tighten dtypes before saving: cast `Priority`, `Type`, `Status`, `Resolution`
(on Issue) and `Change_Type`, `From_String`, `To_String` (on Change_Log) to
pandas `category`.

Write each table to its own CSV under `data/raw/`:
`data/raw/issue.csv`, `data/raw/change_log_status.csv`,
`data/raw/issue_link.csv`, `data/raw/comment.csv`.

## Step 3 — Verify (do not skip; this is the point of running it locally)

After writing the CSVs, run these checks and report the results:

1. **Row counts match the source, exactly.** Run `SELECT COUNT(*) FROM Issue`
   directly against MySQL and confirm it equals `len(issue_df)` — should be
   458,232. Same for `Change_Log WHERE Change_Type='STATUS'`,
   `Issue_Link`, and `Comment` (compare against their own unfiltered
   `COUNT(*)`, since those aren't filtered).
2. **No accidental row filtering.** Confirm `issue_df['Project_ID'].nunique()`
   is 39 and that the cache was not filtered to "resolved" issues (it should
   contain issues with `Resolution_Date` null too).
3. **Column presence and dtypes.** Print `df.dtypes` for each cached table
   and confirm every requested column is present and non-corrupted (e.g.
   date columns actually parse as datetimes, not strings).
4. **Null-rate spot check.** Print `df.isna().mean()` for each table's key
   columns — this won't have a "correct" answer to compare against, but
   flag anything unexpected (e.g. `Issue_ID` null in `Change_Log` would be
   a real problem; `Story_Point` mostly null is expected and fine).
5. **File sizes.** Report the size of each CSV in `data/raw/` so we know
   whether parquet is worth switching to later.

Report back a short summary: row counts (cached vs. live-queried), any
schema surprises found in Step 1, and pass/fail on each verification check
above. If any check fails, stop and report it rather than continuing to
build on top of a bad cache.

## Explicitly out of scope for this task

- Do not filter to resolved issues only — that's a separate, later step.
- Do not build the Phase 6 feature table or Phase 8 snapshot table here —
  this task is only the raw cache those get built from.
- Do not modify anything in `TAWOS_DB` itself — read-only queries only.
