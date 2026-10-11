# Task K — headline metrics: by stage, top flags, severe delays

Every number is from the test set (the most recent 20% of issues, never used for training or tuning). Sources: `reports/j4_metrics_panel.csv`, `k1_precision_at_top.csv`, `k1_scrum_sprint_level.csv`, `k2_severe_metrics.csv`, `k2_severe_precision_at_top.csv`.

## 1. Results by stage (flow mode, model v5b)

A single blended number hides two different jobs. Report them separately.

| Stage | Checkpoints | AUC | Accuracy | Majority-class accuracy | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| **Early warning** | M1–M3 (10–30% of expected duration) | **0.818** | 0.745 | 0.598 | 0.804 | 0.867 |
| **Late stage** | M5–M10 (50–100%) | **0.891** | **0.848** | 0.780 | **0.907** | **0.965** |
| **Final stretch** | M8–M10 (80–100%) | **0.904** | **0.871** | 0.817 | **0.925** | **0.977** |

Per checkpoint AUC: M1 0.803 · M2 0.818 · M3 0.833 · M4 0.848 · M5 0.862 · M6 0.881 · M7 0.890 · **M8 0.897 · M9 0.904 · M10 0.911**.

**Viva phrasing:** "From the halfway point the model separates late from on-time issues with AUC 0.89–0.91. Even at 10% of an issue's expected life it already reaches 0.80."

Always quote late-stage accuracy next to its baseline (0.848 vs 0.780). Later on, more of the issues still open really are late, so accuracy rises partly by itself. AUC is not affected by that, which is why it stays the main metric.

## 2. Precision of top flags: "of the issues we flag, how many really end up late?"

This is what a PM acts on, since they can only chase a handful of issues.

### Flow mode (late = takes longer than the project's typical duration)

| Who is flagged | Precision | Base rate (flagging at random) |
|---|---|---|
| Top 10% of all open issues | **1.00** | 0.70 |
| Top 10% at M1 (10% into expected duration) | **0.96** | 0.56 |
| Top 10% at M3 | **0.99** | 0.64 |
| Top 10%, **outcome still undecided** (not yet past the typical duration) | **0.98** | 0.66 |
| Top 10% at M1, censoring-safe and outcome still undecided (the hardest subset) | **0.96** | 0.59 |
| Top 10% within each project and checkpoint (what one PM sees), averaged | **0.92** | 0.73 |
| Same, outcome still undecided | **0.90** | 0.70 |

At M1, flagging the top 20% instead of 10% gives precision 0.90 (base rate 0.56).

### Severe delays (more than 2× the project's typical duration)

| Who is flagged | Precision | Base rate |
|---|---|---|
| Top 5% of all open issues | **0.99** | 0.50 |
| Top 10% | **0.97** | 0.50 |
| Top 10%, outcome still undecided | **0.94** | 0.49 |
| Top 10% at M5 | **0.96** | 0.50 |
| Top 10% at M1 | 0.85 | 0.39 |

### Scrum mode (late = not finished by the end of its sprint)

| Who is flagged | Precision | Base rate |
|---|---|---|
| Top 10% of open committed issues | **0.95** | 0.67 |
| Top 10% on the first day of the sprint | **0.92** | 0.60 |
| Top 10% at 50% of the sprint | **0.95** | 0.69 |
| Top 10% at 75% of the sprint | **0.97** | 0.78 |

**Sprint-level forecast:** the predicted number of spills per sprint (sum of issue risks) against the actual number.

| Point in the sprint | Correlation | Mean error | Sprints |
|---|---|---|---|
| Sprint start | 0.87 | ±1.6 issues | 475 |
| 50% | 0.89 | ±1.5 issues | 420 |
| 75% | **0.92** | ±1.2 issues | 401 |

The average sprint has about 6 spills.

**Viva phrasing:** "When the model raises its top-10% flags, 92–100% of those issues really do end up late, against a 56–78% base rate. On day 1 of a sprint, it predicts how many committed issues will spill with a correlation of 0.87 to the actual count."

Caveat to state: precision@top-k ignores the issues *not* flagged. Recall of the top 10% is 12–17% by construction. Report it as a triage metric, alongside AUC.

## 3. Severe-delay model (new second target)

- **Label:** `severe = Resolution_Time > 2 × project median`, with the median from train+val as before.
- **Base rate:** 33% of test issues, 50% of checkpoint rows.
- **Setup:** same v5b features plus `past_2x_median`; same protocol (validation-selected config, lr 0.05 / depth 8 / 139 trees, 5 seeds).
- **Models:** `models/xgb_severe_v5b_*`.

| | AUC | Balanced accuracy | Accuracy (majority) | PR-AUC |
|---|---|---|---|---|
| All test rows | **0.829** | **0.743** | 0.743 (0.500) | 0.834 |
| Outcome still undecided | 0.819 | 0.735 | 0.736 (0.514) | 0.809 |
| M1 | 0.794 | 0.688 | 0.722 (0.610) | 0.708 |
| M10 | 0.852 | 0.755 | 0.773 (0.625) | 0.908 |

**Honest finding:** severe delays are **not** easier to rank overall. AUC is 0.829, against 0.866 for the main label, so my expectation was wrong. They are easier to *triage*: the top 10% flags are 97% right against a 50% base rate, a lift of 1.9×, compared with 1.4× for the main label. Use severe mode in the app as the "what will blow up the plan" list. Don't present it as a higher-AUC model.

## 4. What does *not* reach the 90s, and why that is correct

- **Overall AUC: 0.866** (0.857 with strictly causal monthly retraining).
- **Early-warning AUC: about 0.82.**
- **Scrum AUC: 0.746.**

These are rankings across every open issue. Pushing them into the 90s would need information from after the prediction moment, which is leakage. The 90s numbers above are real because they answer narrower, operational questions: late-stage ranking, the precision of the issues you actually act on, and sprint-level counts.
