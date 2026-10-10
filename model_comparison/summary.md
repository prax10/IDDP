# Model comparison — static (Phase 7) problem

Six model families, identical features (`phase6_feature_table_v3.csv`,
30 text PCA components included), identical label (`delayed_v2`), identical
chronological split (`phase8_split.csv`: 142,303 train / 20,329 validation /
40,658 test — never re-split, never touched for tuning). Logistic
regression, random forest, and LightGBM were each given a small
validation-only grid (never test); XGBoost's hyperparameters were not
re-selected here — they were already chosen the same way in Task B4/B5 —
but it was retrained once on that locked config so its training time is
measured under the same conditions as everything else in this run.

| Model | Test AUC | Accuracy @ 0.5 | F1 @ 0.5 | Train time | Imputation needed? |
|---|---|---|---|---|---|
| Majority-class baseline | 0.500 | 0.530 | 0.000 | — | — |
| Logistic regression | 0.729 | 0.666 | 0.636 | 0.5s | Yes |
| Random forest | 0.730 | 0.670 | 0.632 | 32.7s | Yes |
| **XGBoost (tuned)** | **0.744** | 0.681 | 0.638 | 0.4s | No |
| **LightGBM (tuned)** | **0.749** | 0.683 | 0.645 | 0.9s | No |
| LightGBM (XGBoost's exact hyperparams) | 0.740 | 0.676 | 0.635 | 0.4s | No |
| MLP (64, 32) | 0.730 | 0.672 | 0.637 | 8.2s | Yes |

## LightGBM's own tuned configuration edges out XGBoost — reported plainly

**XGBoost does not win this benchmark.** LightGBM, given its own
validation-selected hyperparameters (`learning_rate=0.1, max_depth=6,
n_estimators=100`), reaches **0.749 AUC** against XGBoost's **0.744** — a
real, if modest, +0.005 gap in LightGBM's favour. The two are close
enough to call **near-identical, with a slight edge to LightGBM** — not a
tie, but not a case for switching either, given how small the gap is
relative to the other findings below.

**Why the gap disappears — and slightly reverses — under matched
hyperparameters.** Handing LightGBM XGBoost's exact configuration
(`lr=0.2, depth=4, 42 rounds`) drops it to 0.740 — *below* XGBoost at
those same settings. XGBoost's hyperparameters were validation-selected
specifically for XGBoost's own depth-wise tree growth; LightGBM's
default leaf-wise growth does not respond identically to the same
`max_depth`/`learning_rate` values, so lending it XGBoost's numbers
undersells it. This isolates the two libraries fairly: at matched
settings XGBoost has a small edge; each given its own fair tuning,
LightGBM has a small edge. Neither result should be over-read.

## Linear vs. non-linear: a smaller gap than expected

Logistic regression reaches 0.729 AUC — only **0.015 below tuned
XGBoost**. A substantial share of the signal in this feature set is
close to linearly separable; the tree ensembles' non-linear
capacity adds a real but modest amount on top of it.

## Boosting vs. bagging

Random forest (0.730) sits almost exactly where logistic regression and
the MLP do, and **0.014–0.019 AUC below** the two boosted models.
Boosting specifically — not just "any tree ensemble" — is doing the
work here.

## The missing-value story is real, but confounded with boosting

XGBoost and LightGBM, which pass `NaN` through natively, both
outperform every model that required median imputation (logistic
regression, random forest, MLP: 0.729–0.730). 33,366 of 203,290 rows
carry at least one missing assignee-history feature — a large enough
share that fabricating a value for them is not a rounding error. That
said, this comparison **cannot cleanly separate "native missing-value
handling" from "boosting" as the cause**, since every native-NaN model
tested is also a boosted model, and no imputed *boosted* model was
run as a control. The honest claim is: native-NaN boosted models beat
every imputation-requiring model tested, not that missingness handling
alone explains the full gap.

## Training time

XGBoost (0.4s) and LightGBM (0.9s) are both faster than random forest
(32.7s) and the MLP (8.2s), as well as more accurate — a straightforward
practical argument for either, over both alternatives, for this dataset
size.

## Recommendation for the paper

XGBoost is a **defensible choice among near-equivalent top performers**,
not the outright best model in this benchmark. The paper's sentence
should read something like:

> "We selected XGBoost from among several near-equivalent top
> performers (LightGBM scored marginally higher, +0.005 AUC, under its
> own independently-tuned configuration) for its native handling of the
> 16% of rows missing assignee-history features, and its training speed
> relative to random forest and a small MLP."
