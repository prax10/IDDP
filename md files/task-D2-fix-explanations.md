# Task D2 — Fix the natural-language explanations (they currently contradict the prediction)

Targeted fix, not a rebuild. The generated explanations are misleading in
their current form and would undermine confidence in correct predictions.

## The problem, concretely

For issue 258176 the app renders **"Why this issue is flagged (81%
risk)"** followed by bullets stating that the issue is *under* its typical
resolution time, that its assignee is *better* than the project average,
and that having no linked issues *lowers* its risk. Three of four bullets
do not support the verdict; one explicitly contradicts it. Meanwhile
whatever actually drives the 81% is not shown.

## Fix 1 — Separate increasing from decreasing factors

Never present risk-lowering factors under a "why this is flagged"
heading. Restructure into two clearly labelled sections, ordered by
absolute SHAP contribution:

```
⬆️ Raising the risk estimate
  • …
  • …

⬇️ Lowering the risk estimate
  • …
```

If a risk-lowering factor is among the largest contributors, showing it
is good — it demonstrates the model weighing evidence both ways. It just
must not sit under a heading claiming it explains the flag.

## Fix 2 — Always surface the dominant driver

The top 2-3 contributors by absolute grouped-SHAP value must **always**
appear, regardless of whether a nice sentence template exists for them.
If `elapsed_minutes` or the text-content group is the top driver, say so
in plain language rather than skipping to a feature with a prettier
template:

- elapsed time: *"It has been open for 8.8 days, which is 97% of the
  9.1 days issues like this typically take — the single largest factor
  in this estimate."*
- text group: *"The wording of the title and description resembles
  issues that have historically run late — the model's second-largest
  factor here. (This signal is learned from text patterns and is not
  directly interpretable.)"* — being upfront that an embedding-derived
  signal is not human-readable is more credible than hiding it.

The explanation must account for the prediction, not merely list true
facts about the issue.

## Fix 3 — Repair the percentile phrasing

*"fewer than 100% of issues in this project"* is not meaningful. Use a
correct construction and verify the direction:
- *"It has no linked issues; 64% of issues in this project have at least
  one."*
- *"It has 3 blocking dependencies — more than 89% of issues in this
  project."*

Check every generated percentile string for this class of error.

## Fix 4 — Coherence check (add this as a guard)

After assembling the explanation, verify the shown factors are directionally
consistent with the prediction: if predicted risk is above 0.5, the summed
SHAP contribution of the displayed risk-raising factors should exceed that
of the displayed risk-lowering ones. If it does not, widen the set of
displayed factors until it does, rather than emitting an explanation that
argues against its own verdict.

Add a one-line summary above the bullets that states the verdict and its
main cause together, e.g.:

> **81% likely to be delayed** — driven mainly by how long it has already
> been open and by description content resembling historically slow
> issues, partly offset by an experienced assignee and no blocking
> dependencies.

## Fix 5 — Re-verify on varied issues

Generate explanations for at least 6 issues spanning high risk, low risk,
and mid-range, and check each for: no contradictory bullets under the
wrong heading, the dominant driver present, correct percentile phrasing,
and directional coherence. Report all six.

## Report back

The revised explanation for issue 258176, plus five more spanning the
risk range, and confirmation the coherence guard is in place and the app
still passes the headless test.
