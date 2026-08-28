# EDA Conclusions — Delay Prediction Capstone (TAWOS)

Final state after Phases 1–5, closing out EDA before Phase 6 (feature engineering).

## Resolved-issue definition (final)

An issue is "resolved" if:
1. `Resolution_Date` is not null, OR `Resolution` is not null AND `Resolution_Time_Minutes > 0`
   (catches genuine completions where `Resolution_Date` wasn't populated — a real gap affecting ~28,102 issues dataset-wide, concentrated partly in project 28)
2. AND it represents genuine completed work:
   - `Resolution` in `['Fixed','Done','Complete','Completed','Resolved','Resolved Locally','Implemented','Deployed','Handled by Support']`
   - AND `Status` NOT in `["Won't Fix","Invalid","Duplicate","Cannot Reproduce","Won't Do","Timed out","Obsolete","Inactive","Not A Bug","Rejected"]`
   (the Status check matters because some projects, e.g. project 28, use `Resolution='Done'` as a generic flag even when `Status` says Won't Fix/Invalid)

```python
COMPLETION_RESOLUTIONS = ['Fixed','Done','Complete','Completed','Resolved',
                          'Resolved Locally','Implemented','Deployed','Handled by Support']
NON_WORK_STATUS = ["Won't Fix", "Invalid", "Duplicate", "Cannot Reproduce", "Won't Do",
                   "Timed out", "Obsolete", "Inactive", "Not A Bug", "Rejected"]

base_resolved = df['Resolution_Date'].notna() | (df['Resolution'].notna() & (df['Resolution_Time_Minutes'] > 0))
genuine_completion = df['Resolution'].isin(COMPLETION_RESOLUTIONS) & ~df['Status'].isin(NON_WORK_STATUS)
res = df[base_resolved & genuine_completion].copy()
```

**Result: 203,290 resolved issues** (of 458,232 total), across all 39 projects.

## Label

`delayed = Resolution_Time_Minutes > per-project median(Resolution_Time_Minutes)`. Balance: 50.0%/50.0% (expected from a median split).

## Sanity checks (all passed)

- Delay rate by Priority runs from `To be reviewed` (0.67, worst) down to `Blocker-P1` (0.26, best) — urgent tickets resolve faster.
- Delay rate by Type: `Milestone`/`Epic` near 1.0 (long-running), `Support Request`/`Build Failure`/`Incident` at 0.09–0.18 (fast triage).
- Dynamic feasibility confirmed: most issues have 2+ status changes in `Change_Log` (only 43,501 of ~458k have exactly 1).
- Links: 246,587 total, with real blocks/depends-on categories (9,159 / 5,449).
- Text: 93.6% of issues have description text — SBERT features viable.

## Priority normalization

Two Jira vocabularies merged (Trivial/Minor/Major/Critical/Blocker and Lowest/Low/Medium/High/Highest, plus "- P#" suffixes) onto one 1–5 ordinal scale. Blank/"None"/"To be reviewed" kept as a separate `Unknown` category (37,404 rows) rather than folded into a mid-scale number, since its delay rate is notably higher than any numbered tier — folding it in would have hidden real signal. This is a documented approximation, not an exact unification (some projects mix vocabularies internally, e.g. project 7 uses "Critical" alongside "Low/Medium/High").

## Type normalization

Rare categories (<100 issues each, e.g. Investigation, Release, Wish, Public Security Vulnerability) bucketed into `Other`. Categories with a few hundred+ issues (Documentation, Test Task, Problem Ticket) kept separate.

## Known limitations to state explicitly in the paper

- The label is a proxy (relative to project median), not an absolute SLA breach.
- Priority/Type normalization is an approximation across projects that don't share one consistent vocabulary.
- Smallest projects (10: 313 total, 6, 9, 2, 35 — all under ~700 resolved issues) will produce noisier per-project AUC/F1 in Phase 9's generalization test. Report N alongside the score, don't bury it in an average.
- No-assignee issues (42.9% of all issues) are handled at Phase 6 by leaving assignee-history features as NaN (XGBoost handles this natively) plus an explicit `is_unassigned` flag; concurrent-workload is set to a true 0 for these.

## Kula's Bayesian layer (Phase 8, deferred)

Real paper identified: Kula, Greuter, van Deursen, Gousios, "Dynamic Prediction of Delays in Software Projects using Delay Patterns and Bayesian Modeling," ESEC/FSE '23 (TU Delft + ING). Their method (zero-inflated Beta regression over a continuous BRE label, with DTW-based delay-pattern clustering, fit in Stan) is NOT reproducible as-is: data/code are under NDA, and their label/unit-of-analysis (continuous BRE over epics at 10 milestones) differs structurally from this project's binary `delayed` label over issues via Change_Log snapshots. Planned approach: a Bayesian logistic model over the binary label, updated at each Phase 8 snapshot, in PyMC — described in the paper as "in the spirit of Kula et al.," not a full replication. Decision deferred until Phase 8 is reached.

## Status

Phases 1–5 (EDA) complete and cross-validated (SQL-level counts, Python-level repr checks, and direct pandas source checks all reconcile). Ready to proceed to Phase 6 (feature engineering).
