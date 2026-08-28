# Task C2 — Add the risk-trajectory view + visual polish to the Streamlit app

The app currently shows static and dynamic risk as two numbers. It never
shows the prediction *updating over time*, which is the core contribution
of the project. Fix that first; polish second.

## Priority 1 — Risk trajectory chart (the missing centrepiece)

In **Tab 2 (Issue Risk & Explanation)**, for the selected issue, add a
chart showing how predicted delay risk evolves across its checkpoints.

Build it by scoring the dynamic model on that issue's rows from
`data/features/phase8_checkpoints.csv` — one prediction per checkpoint
the issue actually reached (variable length is expected; issues that
resolved early simply have fewer points).

The chart must show:
- **Line + markers:** dynamic predicted risk (y, 0-1) vs. checkpoint index
  (x, M1-M10).
- **Horizontal dashed line:** the static model's single creation-time
  prediction for the same issue — visually, the "one guess, never
  updated" baseline.
- **Vertical line at M3**, annotated something like *"dynamic becomes
  more reliable than static from here"*. This is the project's headline
  finding rendered as a UI element.
- **Shaded region M1-M2** (light grey) labelled *"insufficient history —
  static prediction preferred"*.
- **A horizontal line at 0.5** (the decision threshold), so it's obvious
  when the issue crosses from predicted-on-track to predicted-at-risk.
- **The actual outcome** if the issue is resolved: annotate whether it
  ended up delayed or not, so the viewer can see whether the trajectory
  converged on the truth.

Use matplotlib or Streamlit's native charting — whichever renders more
reliably. Make the axes and annotations readable at projector size
(fontsize ≥ 11, clear labels).

Below the chart, add one plain-language line generated from the data,
e.g. *"At creation this issue was predicted 34% likely to be delayed. By
checkpoint M7 that had risen to 78%. It was ultimately delayed."*

**Pick 2-3 issues with visually interesting trajectories** (one that
rises sharply, one that stays flat and low, ideally one that falls) and
list their IDs in the report so they can be used in the live demo instead
of hunting for a good example on stage.

## Priority 2 — Visual polish

- `st.set_page_config(layout="wide", page_title="...", page_icon=...)` —
  the default narrow column wastes projector space.
- Replace bare numbers with `st.metric` cards (risk %, at-risk count,
  total issues, delta vs. project average where sensible).
- Colour-code risk consistently everywhere: green < 0.4, amber 0.4-0.7,
  red > 0.7. Use the same thresholds in the dashboard table, the metric
  cards, and the trajectory chart.
- Tab 1: add a small distribution chart of predicted risk across the
  selected project, not just counts.
- Tab 4: show the candidate ranking as a horizontal bar chart of
  predicted risk (current assignee highlighted in a different colour), so
  the comparison is visual rather than a table of decimals.
- Make the honesty verdict in Tab 4 visually prominent — `st.warning` or
  `st.info` with a clear heading, not body text.
- Add a compact header line under the title stating the model in one
  sentence, e.g. *"XGBoost · trained on 203,290 resolved issues across 39
  projects · AUC 0.743 static / 0.859 dynamic at M10"*.

## Priority 3 — Demo safety

- Re-run the headless `AppTest` check after these changes; the trajectory
  chart is new code that touches model scoring, so it can throw the same
  class of dtype error caught last time.
- Ensure every tab renders without error for at least 3 different
  projects and 3 different issues.

## Report back

Confirmation the app still runs cleanly, a description of the trajectory
chart, and the 2-3 recommended demo issue IDs with a one-line description
of what each trajectory looks like.
