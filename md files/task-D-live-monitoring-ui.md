# Task D — Turn the prototype into something genuinely useful

The app currently reads as a historical lookup tool. Three changes make
it feel like a live monitoring system a team would actually use. Build in
priority order — Priority 1 alone transforms the demo.

**Honesty constraint (non-negotiable):** everything here replays *real
historical data with real leakage-safe predictions*. Do not fabricate
data or present a simulation as a live production feed. Label the view
clearly as a replay of historical project data. It is more impressive
because it is real — say so in the UI.

---

## Priority 1 — "Mission Control": time-travel replay (the centrepiece)

New first tab, becoming the app's landing view.

**Core mechanic:** a date cursor (slider + date input, plus a ▶ Play
button that auto-advances in steps of 1 day / 1 week, configurable).
Given a selected project and a cursor date **T**:

- **Open issues at T** = created before T, and either unresolved at T or
  resolved after T. This is genuinely what was on the team's plate that
  day.
- **Risk at T for each open issue** = the dynamic model's prediction at
  that issue's most recent checkpoint whose timestamp ≤ T. If the issue
  hasn't reached checkpoint M3 yet, show the static prediction and label
  it *"creation-time estimate — below the M3 reliability threshold"*.
  This makes the project's headline finding a visible operating rule.
- Everything must use only information available at T. The existing
  checkpoint data is already leakage-safe; do not recompute features
  using later data.

**What the view shows:**
- Metric cards: open issues, at-risk count (risk > 0.5), average risk,
  and **deltas versus the previous cursor step** (e.g. "+3 newly at
  risk"). The deltas are what make it feel alive.
- A live-updating table of open issues sorted by risk, colour-coded with
  the existing green/amber/red bands.
- An **event feed** panel — a reverse-chronological log of what changed
  between the previous step and now: *"⚠️ Issue 12345 crossed into high
  risk (0.48 → 0.71)"*, *"✅ Issue 6789 resolved — on time, predicted
  0.22"*, *"❌ Issue 4471 resolved — delayed, predicted 0.79 (correct
  call)"*, *"🆕 Issue 9910 created — initial risk 0.35"*. This panel is
  what sells it; make it prominent.
- A small running scoreboard: of issues resolved so far in the replay,
  how many the model called correctly. Real, honest, and it accumulates
  as the replay runs — a compelling live accuracy demonstration.
- A project-level timeline chart: x = date, y = count of open issues,
  stacked or shaded by risk band, with a vertical marker at T.

**Pick a project and date window with good activity** for the demo
(reasonable issue volume, visible resolutions and risk transitions) and
report the recommended project ID and start/end dates.

---

## Priority 2 — Explainability that speaks human

The current SHAP output shows 42 features, 30 of which are anonymous PCA
components. Fix that.

**A. Group features before display.** Collapse the 30 `text_pc_*`
components into a single "Issue description content" contribution. Group
the assignee features into "Assignee history & workload". Show at most
6-8 named groups, never raw PCA components.

**B. Generate real natural-language explanations** from SHAP values
*combined with raw feature values* — templated, not generic. Examples of
the target register:

> **Why this issue is flagged (78% risk):**
> - It has been in *In Progress* for **12 days** — about **3.2× longer**
>   than the typical 3.7 days for this status in this project.
> - Its assignee currently holds **14 open issues** and has finished
>   late on **61%** of their past work (project average: 44%).
> - It has **3 blocking dependencies**, more than 89% of issues in this
>   project.
> - Issues of this type and priority in this project typically resolve in
>   **4 days**; this one is at **day 11**.

You already computed the dwell-time baselines and hierarchical expected
durations in Phase 8 — reuse them rather than recomputing. Each bullet
should be generated only when the underlying SHAP contribution is
material, so the explanation reflects the model rather than being a fixed
script.

**C. Peer comparison.** For the selected issue, show where it sits
against similar issues (same project + type + priority): a small
distribution plot of their resolution times with this issue's elapsed
time marked. Instantly answers "is this actually unusual?"

**D. What-if controls.** Sliders for assignee concurrent workload,
assignee historical delay rate, and priority — re-score live and show the
risk change. This demonstrates the model responding rather than just
reporting, and it feeds naturally into the reassignment tab.

---

## Priority 3 — Better visuals

- **Cohort band on the trajectory chart:** overlay the median and
  interquartile range of trajectories for similar issues (same project +
  type), so a single issue's line is read against its peers rather than
  in isolation. This is the single biggest upgrade to the existing chart.
- **Project timeline:** issues as horizontal bars (creation → resolution)
  coloured by final risk, for the selected project and window.
- **Team heatmap:** developers × time, cell colour = average risk of
  their open issues. Makes overload visible at a glance.
- Retire or fold in any tab that becomes redundant — four tabs of tables
  is worse than three well-designed views.

---

## Priority 4 — Demo safety

- Re-run the headless `AppTest` after each priority block (the replay
  logic and what-if controls are both new scoring paths that can throw
  the same dtype/API errors caught twice already).
- Verify every view renders for at least 3 projects and 3 issues.
- Confirm the replay runs end-to-end over the recommended window without
  error, including the Play auto-advance.

---

## Report back

Confirmation it runs cleanly; the recommended demo project and date
window for the replay; a description of the event feed and what a typical
few steps look like; a sample generated natural-language explanation for
one issue; and any view you'd cut as redundant.
