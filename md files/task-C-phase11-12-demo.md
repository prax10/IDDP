# Task C — Phase 11 (reassignment suggester) + Phase 12 (Streamlit prototype)

**Run AFTER Tasks A and B**, so the demo loads the final model generation
rather than one that gets replaced.

Scope discipline: this is a demo wrapper around existing models. No new
modeling, no live Jira sync, no task editing, no notifications.

## Phase 11 — Reassignment suggester (on-demand, not automatic)

Build as a function/module (`reassignment.py`), importable by the
Streamlit app:

```
suggest_reassignment(issue_id, top_n=5) -> ranked candidates + honesty verdict
```

Logic:
1. Take the issue's current feature vector.
2. Build a candidate pool: developers with at least ~20 prior resolved
   issues **in that issue's project** (report the threshold used; adjust
   if it yields too few candidates on small projects).
3. For each candidate, swap in *their* assignee features
   (`assignee_prior_delay_rate`, prior resolved count, current concurrent
   open load) — leave every non-assignee feature unchanged.
4. Re-run the model on each variant → predicted delay risk per candidate.
5. Rank ascending by risk; return the top N with their predicted risk and
   the delta vs. the current assignee.

**Honesty check (required — this is the interesting part):** compute SHAP
for the issue under its current assignee. If the top contributors to its
risk are *non-assignee* features (e.g. `n_links`, `Type_Normalized`,
`elapsed_minutes`, text components), return a verdict saying reassignment
is unlikely to help and naming what's actually driving the risk — instead
of suggesting a swap that won't change the outcome. Only recommend
reassignment when assignee-related features are materially responsible
**and** the best candidate's predicted risk is meaningfully lower (use a
threshold, e.g. ≥0.05 absolute risk reduction; report what you chose).

## Phase 12 — Streamlit prototype

`app/app.py`, run with `streamlit run app/app.py`. Load the final model
via joblib and the feature table from disk — no retraining at runtime,
and cache loads with `@st.cache_data` / `@st.cache_resource` so the app
is responsive.

Four tabs, in priority order (build 1 and 2 first — if time runs short,
those two alone are a legitimate demo):

**1. Project Risk Dashboard.** Pick a project from a dropdown. Show
count of at-risk vs. on-track issues, and a table of the top-N highest
risk issues (ID, title, type, priority, assignee, predicted risk).

**2. Issue Risk & Explanation.** Pick/search an issue. Show its predicted
risk, and its SHAP waterfall explaining the top contributing factors in
plain language. This is the "why is this at risk" view and the most
compelling thing to demo.

**3. Team Workload.** For a project, show each developer's current
concurrent open issue count and their historical delay rate — the
workload picture the model uses.

**4. Reassignment Suggester.** For a selected at-risk issue, call
`suggest_reassignment()` and show the ranked candidates with predicted
risk reduction — *or* the honesty verdict when reassignment won't help.
Make the honesty verdict visually prominent; it's the most defensible
design decision in the whole app and worth showing off deliberately.

Add a short "How it works" expander somewhere: label definition
(slower than that project's median), the leakage-safe design, and that
predictions are checkpoint-based for in-progress issues.

## Report back

Confirmation the app runs (`streamlit run app/app.py`), a screenshot or
description of each tab, the candidate-pool threshold and
risk-reduction threshold chosen for Phase 11, and one worked example of
each reassignment outcome — one issue where a swap is recommended, and
one where the honesty check correctly says it won't help.
