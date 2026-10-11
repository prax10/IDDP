# Viva / presentation prep, v6 (after Tasks H–L)

> **Task M update (deployed model v6):** overall AUC is now **0.872** (5-seed ensemble 0.873), up from
> 0.866, through rolling-origin re-tuning. M10 0.915, M8–M10 0.908, confident-late precision 97.5%.
> Comment text and stacking were tried and did not help (`task-M-push-auc.md`). Where this doc says
> 0.866 / 96.4%, quote the v6 numbers; the logic of every answer is unchanged.

Replaces `viva-prep.md` (Aug), which predates the v5 model and still says "dynamic wins from
M3", "93% cluster" and "cross beats within 30/38". **Don't use those lines any more.** Numbers here
come from `paper-updates-v5.md`, `paper-updates-L.md`, `task-K-headline-metrics.md` and
`task-L-results.md`.

---

# PART 1: THE FIVE HEADLINE ANSWERS

## "Explain your main finding"

> "We predict, while an issue is still open, whether it will take longer than its project
> normally takes, and we update that prediction as things happen to it: status moves, review,
> comments, reassignments, edits.
>
> Re-forecasting beats a creation-time prediction from the very first checkpoint. At 10% of an
> issue's expected duration we're at AUC 0.80 against 0.73, and by the end 0.91 against 0.74.
> Overall AUC is 0.866 on a held-out later period, and 0.857 if we retrain monthly the way a real
> deployment would.
>
> And it travels. Trained only on TAWOS, it ranks issues at 5 companies it has never seen at AUC
> 0.76 to 0.85, about as well as a model trained on each company's own history."

**If asked how sure:** 5 training seeds, paired bootstrap CIs, and every comparison re-checked on
censoring-safe rows and on rows whose outcome isn't already decided. The hardest subset
(censoring-safe *and* outcome still open) is 0.823.

## "What happened to Kula's delay patterns?" (RQ2: the answer changed, know this one cold)

> "We rebuilt their DTW clustering. Our first version, on a 4,000-issue sample with complete
> linkage, put 93% of issues in one cluster. We originally blamed issue-level data for that.
>
> Then we ran it at full scale, over all 115,000 training trajectories with k-medoids, and the
> clusters came out balanced, roughly 65/27/7/1, with very different late rates: 6% for one
> cluster against 71% for another. So the 93% was an artifact of complete linkage, and we
> corrected our own explanation.
>
> The better clusters *do* help the older feature set, about +0.012 AUC at M3, which is Kula's
> effect. But once we add workflow-activity features (time since the status last moved, whether
> it reached review, reassignments) the patterns add exactly nothing. They're redundant, not
> useless: the activity features measure the same behaviour directly."

## "Does it generalise?"

> "Two levels. Inside TAWOS, on a common test set, training on other projects matches training
> on a project's own history, and pooling everything is significantly better: 29 of 38 projects,
> p ≈ 10⁻⁶.
>
> Outside TAWOS, we took five Jira organisations from Montgomery et al.'s Public Jira Dataset
> that TAWOS doesn't include: MariaDB, Qt, Sakai, JFrog and Intel DAOS. With no retraining, AUC
> is 0.76 to 0.85, mean 0.80. Training on each company's own history gives a mean of 0.79. TAWOS
> alone ties or wins on 3 of 5, and pooling both is best on 4 of 5."

**Volunteer the caveats:**
- That dataset has no comments, so the portable model drops them (it gets 0.858 on TAWOS, against 0.866).
- Two repos are heavily censored.
- The ranking transfers, but the threshold needs re-setting per company.

## "What numbers are in the 90s, and are they real?"

> "AUC overall is 0.87. Pushing a ranking of *every* open issue into the 90s would need
> information from the future. The 90s numbers answer narrower, operational questions:
> - Late-stage AUC is 0.89 to 0.91.
> - Of the top-10% issues we flag, 92 to 100% really end up late, against a 56 to 78% base rate.
> - When the model says it's *confident* an issue will be late, and that issue isn't already
>   overdue, it's right 96% of the time. That holds at every checkpoint.
> - In Scrum mode, the predicted spill count per sprint correlates 0.87 to 0.92 with the actual count."

## "What does the tool actually give a project manager?"

> "**Flow mode:** a ranked list of open issues with the reasons for each risk. Each issue
> gets a confidence label ('confident late', 'confident on time', 'uncertain') with 92% coverage,
> plus a range for how many days are left. Then 'What would help?': re-scoring the issue if you
> started it, got it into review, discussed it, or handed it to a specific colleague.
>
> **Scrum mode:** the deadline is the sprint end. It shows a burndown with a forecast, spill risk
> per committed issue, and how many issues will spill as a range, which held the true count in 96%
> of test sprints."

---

# PART 2: HARD QUESTIONS

## "Why isn't overall AUC above 0.9?"

> "Because 'overall' averages in the very early checkpoints. At 10% of an issue's expected life
> almost nothing has happened yet; we're at 0.81 there, and 0.90 to 0.915 from 80% onward. We
> tried hard to push it: re-tuning across three time windows gave +0.006, comment text +0.0004,
> and ensembles +0.0006. Every route we know to 0.9 overall uses information from after the
> prediction moment, which would be leakage."

## "Is XGBoost even necessary?"

> "Barely. With the same features, logistic regression gets 0.856, an MLP 0.864, a GRU reading
> the whole checkpoint sequence 0.863, and XGBoost 0.866. The jump from our Phase-8 model, 0.80
> to 0.87, comes from the features. A classic delay-risk feature set from the literature gets
> 0.78 on the same data. Averaging XGBoost, MLP and GRU adds another +0.005, but we kept single
> XGBoost because its explanations are exact."

## "Why doesn't the sequence model win, if this is about trajectories?"

> "Because the snapshot already summarises the trajectory: counts, time since the last event,
> whether it ever reached review. Same lesson as the patterns: once you measure the behaviour
> directly, the shape of the history adds nothing."

## "Can you predict *how* late, not just whether?"

> "Yes. We regress log2 of the duration relative to the project median. It beats random
> guessing by 59% (standardised accuracy), rising from 54% at the first checkpoint to 63% at the
> last, and it beats creation-time and project-median baselines. But the typical error is about
> 20 days against a median of 27 remaining days. So the app shows a range, not a date.
> Kula's 66–92% is for whole epics at one bank, so it isn't comparable."

## "How do you know the confidence labels mean anything?"

> "Conformal prediction. We calibrate on the validation period and measure on the later test
> period, so drift is included. The 90% sets contain the true outcome 92% of the time. We also
> re-checked with the exact models the app deploys, not just the clean experimental copies.
> One honest weak spot: at the very first checkpoint, coverage for late issues is 79%."

## "Are the 'what would help' suggestions causal?"

> "No, and the app says so. They're the model's estimate of how risk would move if one thing
> changed. We checked them against history: issues that really did get started, reviewed,
> discussed or reassigned in the next week, compared with similar-risk issues that didn't, and
> only counting issues still open a week later. All four point the right way. Reassignment is
> about the right size, status moves are overstated about 2×, and commenting is understated. A
> causal claim would need a randomised trial; that's future work."

## "Did you have leakage?"

> "We found and fixed several, all before reporting:
> - A label median computed over test issues (3.4% of test labels).
> - A status field that fell back to the *final* status.
> - Two Scrum label leaks: carry-overs after early sprint completion, and planning moves
>   counted as commitments.
>
> Everything uses chronological splits and features computed strictly before the prediction
> moment. We also show results on censoring-safe rows. Only resolved issues are modelled, so
> recent long-running ones are under-represented near the snapshot end, and every headline
> holds on the safe subset."

## "Did any of your conclusions change?"

> "Three did, and we report all three:
> 1. 'Dynamic only wins from M3' became 'from M1', once activity features were added.
> 2. 'Cross-project beats within on 30/38' came from scoring the two on different test sets.
>    On a common test set they're equal, and pooling is best.
> 3. The 93% cluster turned out to be a clustering artifact."

## "Would this work at my company?"

> "Now we can say more than 'retrain it'. A model trained on TAWOS ranked five other
> organisations' issues at AUC 0.76–0.85 straight away. You'd want a few months of your own
> history to set the alert threshold, because late rates differ between companies. One structural
> caveat: lateness is relative to your own median, so the tool triages within your pace and
> can't tell you your pace is slow."

## "Isn't 'late = slower than the project median' too easy? Half of everything is late."

> "It's balanced by design, so AUC is the honest metric. We also built a severe-delay label,
> more than 2× the median. It's harder to rank (AUC 0.83) but easier to triage: the top-10%
> flags are 97% right against a 50% base rate."

---

# PART 3: CONCEPTS IN PLAIN LANGUAGE

**Checkpoints.** M1–M10 are moments we *look* at an issue: k/10 of how long similar earlier
issues took. It's computed only from issues finished before this one was created, so it's
knowable in real time. They aren't stages the issue passes through.

**Why accuracy rises late by itself.** Issues still open late in their life are mostly late. That's
the inspection paradox, the doctor's waiting room. So quote late-stage accuracy next to its
baseline (0.848 vs 0.780), and lead with AUC.

**DTW.** It compares two sequences that are stretched or shifted in time, like matching two people
saying the same word at different speeds. **k-medoids** picks real example issues as cluster
centres, which works under DTW. **Complete linkage** merges clusters by their *farthest* members.
That isolates a few extreme trajectories as their own clusters and leaves everything else lumped
together, which is the cause of our old 93%.

**Conformal prediction.** On data the model didn't train on, look at how wrong it is, take the
90th-percentile error, and use that margin from then on. If the future looks like the past, 90%
of answers fall inside. A "set" can be {late}, {on time} or {both}, and "both" means
"I don't know".

**Standardised accuracy (SA).** How much better than random guessing the error is. 59% means the
average error is 41% of a random guess's error. It is not the percentage of correct answers.

**Zero-shot.** Using the model on a new company without training on any of its data.

**Counterfactual re-scoring.** Change one input ("status = in review") and ask the model again.
It tells you what the *model* thinks would change, not what reality would do.

---

# PART 4: DEMO SCRIPT (v5 app, about 5 minutes)

Run `streamlit run app/app_v5.py`. Pick the issues live, or rehearse with your own picks
beforehand.

1. **Flow → Portfolio.** Pick a large project and a date in the test period.
   - Point at the "Confident late calls" KPI and the confidence column.
   - *Say:* "Every number is computed from events before this date. Overdue issues are
     separated out, because they're late by definition."
2. **Issue detail** on a high-risk issue.
   - The risk trajectory over time.
   - The top reasons, as exact contributions to the score.
   - "Likely done in X–Y days".
3. **What would help?** on the same issue.
   - Read the table, then *say the caveat out loud*: model estimates, checked against history,
     direction right on all four actions.
   - Then pick an issue where nothing helps (an overdue one) and show that it says so.
4. **Toggle "Show actual outcomes".** Show the snapshot AUC and how many top flags really were late.
5. **Scrum mode → Sprint board.** Show the burndown with the red dashed forecast, and
   "Predicted to spill 5.9 (90% range 2–8)".
   - Move the day slider and watch the forecast update.
   - Then open **Sprint history**: predicted vs actual spills, correlation about 0.87–0.92.
6. **Close on generalisation:** "and trained only on this data, it ranks issues at five other
   companies at AUC 0.80."

---

# PART 5: NUMBERS TO HAVE MEMORISED

| Quantity | Value |
|---|---|
| Dataset | TAWOS: 458,232 issues, 39 projects, 12 repos; 203,290 resolved |
| Test boundary | 2018-05-10 (most recent 20%) |
| Overall AUC | **0.872** (v6; was 0.866 for v5b); v5b with monthly retraining 0.857 |
| Static vs dynamic, M1 | 0.728 vs **0.803** |
| Static vs dynamic, M10 | 0.739 vs **0.911** |
| Late-stage AUC (M5–M10) | 0.891 (accuracy 0.848 vs majority 0.780) |
| Hardest subset (censoring-safe, outcome open) | 0.823 |
| Top-10% flag precision | 0.92–1.00 (base 0.56–0.78) |
| Confident-late precision (not yet overdue) | **97.5%** (v6), catching 23% of late issues |
| Conformal coverage (90% target) | 92% flow · 84% / 93% how-late (80 / 90% targets) · 96% Scrum (deployed) |
| Regression SA | 59% (54% at M1 → 63% at M10) |
| Baselines | LR 0.856 · RF 0.850 · MLP 0.864 · GRU 0.863 · literature 0.782 |
| Scrum spillover AUC | 0.746 (team-history baseline 0.590) |
| Sprint spill-count correlation | 0.87 (start) → 0.92 (75%) |
| Severe-delay AUC / top-10% precision | 0.829 / 0.97 |
| LOPO pooled vs within (dynamic) | 0.800 vs 0.756; 29/38 projects |
| External, zero-shot mean AUC | **0.801** (range 0.755–0.852), 5 organisations |
| External, own-history mean | 0.792 |
| Full-scale clusters | 65/27/7/1% (level) · 76/12/12% (shape); late rates 6–72% |
| Patterns on top of v5b | Δ AUC 0.000 |

---

# PART 6: SELF-TEST (answer out loud, then check above)

1. Why is "93% in one cluster" no longer your explanation, and what replaced it?
2. A panel member says "0.87 isn't 90s." Give three honest 90s numbers and say what each one measures.
3. Why does accuracy rise towards M10 even for a mediocre model? Which metric do you lead with, and why?
4. Zero-shot IntelDAOS has AUC 0.852 but accuracy 0.61. How can both be true?
5. What exactly does "confident: late" guarantee, and what doesn't it guarantee?
6. Why can't the "What would help?" numbers be called causal? What would you need?
7. Logistic regression is only 0.011 behind XGBoost. Is that a weakness of your work or a finding?

*(Answers: 1 → Part 1, RQ2. 2 → Part 1. 3 → Part 3. 4 → ranking transfers, threshold doesn't;
late rate 36%. 5 → coverage at the 90% level and 96% precision on not-yet-overdue issues; it does
not catch most late issues (28%). 6 → Part 2. 7 → Part 2: the features are the contribution.)*
