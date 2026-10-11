# Scrum mode — a separate prediction mode for sprint-run projects

*Paper section draft, also shown as the Scrum model card in `app/app_v5.py`.*

## Why a separate mode

In flow (Kanban-style) work, "late" is relative: the issue takes longer than this project normally takes. In Scrum the deadline is explicit and much stricter. Each issue is committed to a sprint, and it is late if it is not finished when that sprint closes.

The project manager knows whether a project runs sprints, so nothing has to be detected. Switching the project into Scrum mode changes three things:
1. the label, to sprint spillover;
2. the clock, to sprint progress instead of the issue's expected duration;
3. what is shown, to the sprint board, burndown, spill risk and a sprint-level forecast.

**RQ5.** Can the dynamic framework predict whether a committed issue will be finished within its sprint, and how many issues a sprint will spill, early enough to act?

## Data and label

- **Sprints.** The TAWOS `Sprint` table: 4,594 sprints with start, planned end and actual completion. It is joined to `Change_Log` rows with `Field = 'Sprint'` (175k sprint-membership changes).
- **Committed sprint.** The first sprint the issue was actually *in while that sprint was running*. Sprint fields give a membership timeline per issue, and planning moves made before a sprint started do not count as a commitment to it.
- **Sprint close.** The *earlier* of the planned end and the actual completion date. A third of sprints closed before their planned end, and Jira logs the carry-over of unfinished issues at the close.
- **Label.** `spilled = 1` if the issue is not resolved by the close + 1 day. Issues that are still unresolved are included (they spilled), so unlike the flow label this target is not affected by the resolved-only censoring found in J1.
- **Population.** 33,291 committed issues in 3,732 closed sprints; overall spill rate 49.9%. Rows by split:

| Split | Checkpoint rows |
|---|---|
| Train | 77,247 |
| Validation | 10,206 |
| Test | 16,744 (747 sprints) |

- **Split.** Chronological by sprint start, 70/10/20 of sprints.

## Checkpoints and features

**Checkpoints:** 0%, 25%, 50% and 75% of the *planned* sprint window (Kula-style milestones on the sprint clock). A row exists only while the issue is still open, still in the sprint, and strictly before the real close.

**Features, all computed from events before the checkpoint moment:**
- **Sprint state:**
  - progress through the sprint and days left;
  - sprint length;
  - whether the issue joined late or was carried in from an earlier sprint;
  - sprint load at that moment;
  - **burndown share** (fraction of the sprint's committed issues already resolved);
  - **team spill rate known at that moment** (outcomes of the project's earlier closed sprints).
- **Issue activity** (same engine as flow mode): workflow stage, time since the last status change / comment / any activity, status moves, review reached, reassignments, edits in the last week, story points at that moment, and assignee and reporter track records.
- **Context:** project, priority and type.

**Model:** XGBoost, depth 6, lr 0.05, 134 trees, selected on validation; 5-seed ensemble (`models/xgb_scrum_spillover_*`).

## Results (test sprints)

**Issue level** (`j5_scrum_metrics.csv`):

| Sprint progress | AUC | Accuracy | Majority accuracy | PR-AUC | Brier |
|---|---|---|---|---|---|
| Day 1 (0%) | 0.721 | 0.677 | 0.602 | 0.796 | 0.206 |
| 25% | 0.727 | 0.697 | 0.645 | 0.828 | 0.197 |
| 50% | 0.736 | 0.728 | 0.693 | 0.856 | 0.182 |
| 75% | **0.764** | **0.789** | 0.775 | **0.912** | 0.147 |
| All | **0.746** | 0.718 | 0.671 | 0.855 | 0.186 |

- **Baseline:** a team-history baseline (logistic regression on the team's past spill rate plus sprint progress) scores AUC 0.590. The difference is **+0.156** [+0.148, +0.165].

**Triage** (`k1_precision_at_top.csv`):

| Point in the sprint | Top-10% flags actually spilled | Base rate |
|---|---|---|
| All checkpoints | **95%** | 67% |
| Day 1 | **92%** | 60% |
| 75% through | **97%** | 78% |

**Sprint level** (`k1_scrum_sprint_level.csv`): the predicted spill count (sum of issue risks) against the actual count.

| Point in the sprint | Correlation | Mean absolute error | Sprints |
|---|---|---|---|
| Day 1 | 0.87 | 1.6 issues | 475 |
| 50% | 0.89 | 1.5 issues | 420 |
| 75% | **0.92** | 1.2 issues | 401 |

The average sprint spills about 6 issues. AUC for "half or more of the open commitments will spill" is 0.71 on day 1 and 0.76 at 75%.

**What drives spill risk:** time since the status last moved, time since any activity, the project, the assignee's experience, workflow stage, and edits in the last week. Sprint-specific signals come next: days left, progress, burndown share and the team's spill history (`j5_scrum_gain_importance.csv`).

## Validity notes specific to Scrum mode

- **Two leaks were found and fixed before reporting.** Logs: `j5_run_log_v1_with_carryover_leak.txt` and `j5_run_log_v2_planning_moves.txt`.
  1. Carry-over events logged at an early sprint completion.
  2. Pre-sprint planning moves counted as commitments.

  The honest model scores 0.746; the first version scored 0.752.
- **Coverage.** Only sprints that can be matched to the Sprint table are covered (44% of sprint-tracked issues), and only projects that use Jira sprints.
- **Base-rate drift.** The spill rate rises from 0.55 in train to 0.67 in test. AUC is unaffected; accuracy should be read against the majority baseline.
- **Sprint goals and scope changes are not modelled.** Adding issues mid-sprint is captured by `joined_late`; story-point re-estimation by `sp_at_T`.

## In the app (Scrum mode)

- **Sprint board:**
  - KPIs: committed, done, still open, predicted to spill, predicted completion, days left;
  - burndown with the model's projected end-of-sprint count;
  - every open committed issue with its spill risk and top reasons.
- **Issue detail:** spill-risk trajectory since the issue joined the sprint, plus SHAP factors.
- **Sprint history:** past sprints, comparing the predicted spill count at sprint start with the actual count, and the team's completion rate.
