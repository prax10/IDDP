# Task E — Make the app genuinely usable for a project manager

Substantial UX redesign. Work in the order below — Step 0 gates one of
the features, and Steps 1-3 are the ones that change how the app feels.

---

## Step 0 — Verify link direction (gates the blocker feature)

Check `data/raw/issue_link.csv`: does it record **direction** — i.e., can
you tell that issue A *blocks* issue B, as opposed to only that A and B
are linked? Look at the `Description` values and whether there is a
linked-issue-ID column identifying the other side.

Report what you find:
- **If direction is available:** build the full blocker cascade view
  (Step 3).
- **If only undirected links exist:** say so plainly and build the weaker
  version — "this overrunning issue has N links to other issues" — and do
  not describe it as a cascade or as blocking. Do not infer direction
  from keyword guesses.

---

## Step 1 — Split overrunning issues out of the risk pool

**The problem:** 119 of 121 open issues show as at-risk, average 86%.
Useless to a manager — everything is red.

**The fix, and the reasoning:** an issue that has already exceeded its
`expected_duration_proxy` (past checkpoint M10) is not a prediction — it
is already running long, and the model saying "91% likely to be slow"
about it is stating a fact. Separate the two populations:

- **Overrunning** — elapsed time already exceeds `expected_duration_proxy`
  (i.e. past M10, or `overran_expected_duration = 1`). These are facts,
  not forecasts. Give them their own count, their own list, and their own
  metric (e.g. "how far past expected duration", as a multiple: 1.4×,
  2.7×).
- **At risk** — still within expected duration, but predicted to exceed
  the project median. **This is the genuine prediction queue** and the
  list a manager acts on.
- **On track** — within expected duration, predicted fine.

**Naming:** use "Overrunning" or "Past expected duration" — **not "past
deadline."** TAWOS has no deadline field; nothing in the data represents
a commitment, so "deadline" would misrepresent it.

**Average risk must be computed over the at-risk + on-track population
only**, excluding overrunning issues. State in the UI that overrunning
issues are excluded and why ("already past expected duration — tracked
separately").

Metric cards become: Open · Overrunning · At risk · On track · Average
risk (of non-overrunning).

---

## Step 2 — Sticky context header

The project selector and date cursor currently scroll off the top, so
once you scroll to the table you cannot tell which project or date you
are looking at.

Put a persistent header at the top of Mission Control — visible at all
scroll positions (`st.container` pinned at top, or repeat the context
line above each major section) — showing: **project name/ID · current
replay date · step size**. The project selector and date cursor stay
adjacent to it.

---

## Step 3 — Blocker / cascade view (conditional on Step 0)

Among **overrunning** issues, surface those holding up other work:

- If direction is available: rank overrunning issues by **how many other
  open issues they block**, and show which ones. A single overdue issue
  blocking six others is the highest-leverage thing a manager can act on.
  Show it prominently — this is arguably the app's most valuable output.
- If direction is unavailable: rank by number of links, labelled honestly
  as "connected issues", with no blocking claim.

Add a compact visual — a small dependency graph or an indented list
showing the overrunning issue and the issues attached to it.

---

## Step 4 — Fix the running scoreboard

Currently shows "3514 / 4480 resolved issues so far" on the very first
step, before any replay has occurred. It appears to be counting the
project's all-time resolved issues rather than accumulating during the
replay.

It must count **only issues resolved between the replay start date and
the current cursor date**, incrementing as the cursor advances. Also show
the majority-class baseline alongside it, so the accuracy figure is
honest ("78% correct · predicting 'delayed' for everything would score
X%").

---

## Step 5 — Open mid-window so the app doesn't look dead on arrival

On first load the app shows "+0 vs. last step", "no change", and "No
events in this step" — every dynamic element reads as broken.

Default the cursor to roughly **one-third into the replay window**, with
the previous steps already simulated, so metrics deltas, the event feed,
and the scoreboard are all populated on arrival. Keep a "Reset to start"
control for anyone who wants to run from the beginning.

Also: filter the event feed to **significant** changes — risk-band
crossings, resolutions, new high-risk issues — rather than every small
movement, so the feed reads as signal rather than noise.

---

## Step 6 — Fold Reassignment into the issue view

Remove the standalone Reassignment Suggester tab. Instead, on any issue
in the issue detail view, add a **"Check reassignment"** button that runs
the suggester inline and shows the ranked candidates or the honesty
verdict in place.

For issues below M3 (`source = static`), run the suggestion using the
**static** model rather than the dynamic one, consistent with the
existing source logic, and label it as such ("creation-time estimate —
this issue has not yet reached the M3 reliability threshold").

---

## Step 7 — Fix the what-if controls

The what-if sliders reportedly do not work. Diagnose and fix. Likely
causes: the modified feature vector is not being passed to the model, the
sliders are not triggering a re-score, or a dtype error is being silently
swallowed. Report what was actually wrong.

Verify each slider (assignee workload, assignee delay rate, priority)
produces a visible, sensible change in predicted risk.

---

## Step 8 — Formatting and naming

- **Rename the app** to **"Intelligent Dynamic Delay Prediction"**
  everywhere: page title, browser tab, headers.
- **Numbers:** whole numbers display with no decimals (`148358`, not
  `148358.000000`); non-integers display to at most 2 decimal places.
  Apply across every table and metric — `Assignee_ID` is the worst
  current offender.

---

## Step 9 — Verify

Re-run the headless `AppTest` after each of Steps 1, 3, 6 and 7 (each
touches model scoring or data flow). Confirm every view renders for at
least 3 projects and 3 issues, and that the replay runs end to end.

---

## Report back

Step 0's finding on link direction; the new metric-card numbers for the
demo project (how many overrunning vs. at risk vs. on track, and what the
average risk becomes once overrunning issues are excluded); what was
actually wrong with the what-if controls; and confirmation the scoreboard
now accumulates correctly.
