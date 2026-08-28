# Task C — Phase 11 (reassignment suggester) + Phase 12 (Streamlit prototype)

Scope discipline: this is a demo wrapper around **already-final** models.
No retraining, no new modeling, no live Jira sync, no task editing, no
notifications.

## Which artifacts to use (all settled — do not retrain)

- **Feature table:** `data/features/phase6_feature_table_v2.csv` (with the
  30 SBERT text PCA components).
- **Label:** the v3 train+validation-median label (`delayed_v2` or
  whatever it was named in Task B3 — check the file).
- **Static model:** the tuned one (lr=0.2, max_depth=4, n_estimators=42).
- **Dynamic model:** the tuned pooled model with pattern label (lr=0.1,
  max_depth=8, n_estimators=11).
- **Thresholds:** 0.5 for accuracy-style decisions, 0.405 if optimizing
  for F1. Pick one for the demo, state which in the UI.

If any of these aren't saved to disk as joblib files, save them first
from the existing scripts — do not refit with different settings.

## Phase 11 — Reassignment suggester

Build as an importable module (`reassignment.py`):

```
suggest_reassignment(issue_id, top_n=5) -> ranked candidates + honesty verdict
```

Logic:
1. Take the issue's current feature vector.
2. Candidate pool: developers with ≥20 prior resolved issues **in that
   issue's project**. Report the threshold used; loosen it for small
   projects if it yields too few candidates.
3. For each candidate, swap in *their* assignee features
   (`assignee_prior_delay_rate`, prior resolved count, current concurrent
   open load). Leave every non-assignee feature unchanged.
4. Re-run the model per variant → predicted delay risk per candidate.
5. Rank ascending by risk; return top N with predicted risk and the delta
   vs. the current assignee.

**Honesty check (required — this is the most defensible part of the
app):** compute SHAP for the issue under its current assignee. If the top
contributors to its risk are *non-assignee* features (`n_links`,
`Type_Normalized`, `elapsed_minutes`, text components), return a verdict
saying reassignment is unlikely to help, and name what is actually
driving the risk — rather than suggesting a swap that won't change the
outcome. Only recommend reassignment when assignee-related features are
materially responsible **and** the best candidate's predicted risk is
lower by a meaningful margin (e.g. ≥0.05 absolute; report your choice).

## Phase 12 — Streamlit prototype

`app/app.py`, run with `streamlit run app/app.py`. Load models via joblib
and the feature table from disk. Cache with `@st.cache_resource` (models)
and `@st.cache_data` (data) so the app stays responsive.

Four tabs, in build priority order — **tabs 1 and 2 alone are a
legitimate demo** if time runs short:

**1. Project Risk Dashboard.** Project dropdown → count of at-risk vs.
on-track issues, and a table of the top-N highest-risk issues (ID, title,
type, priority, assignee, predicted risk).

**2. Issue Risk & Explanation.** Search/select an issue → predicted risk
plus a SHAP waterfall explaining the top contributing factors in plain
language. This is the most compelling thing to demo live.

**3. Team Workload.** Per project: each developer's concurrent open issue
count and historical delay rate — the workload picture the model uses.

**4. Reassignment Suggester.** For a selected at-risk issue, call
`suggest_reassignment()` and show ranked candidates with predicted risk
reduction — *or* the honesty verdict when reassignment won't help. Make
that verdict visually prominent; it is the strongest design decision in
the app.

**Add a "How it works" expander** stating: the label definition (slower
than that project's own median, median computed from training data only),
the leakage-safe design, that predictions for in-progress issues are
checkpoint-based, and — importantly — that **dynamic predictions only
become more reliable than creation-time predictions after ~30% of an
issue's expected duration (checkpoint M3)**. That last line turns your
headline research finding into a visible product decision, which is
exactly what a demo should do.

## Report back

Confirmation the app runs; a short description or screenshot of each tab;
the candidate-pool and risk-reduction thresholds chosen; and one worked
example of **each** reassignment outcome — one issue where a swap is
recommended, and one where the honesty check correctly says it won't
help.
