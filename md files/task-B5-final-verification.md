# Task B5 — Final verification, then stop

Two checks. This is the last iteration on the model — after this we
settle the numbers and move to the demo and write-up regardless of
outcome. Do not open new questions beyond what is asked here.

## Check 1 — Sanity-check the tuned configs (the 11-estimator anomaly)

`n_estimators=11` for the dynamic model and `42` for static are unusually
small, and the 11-round pooled model scoring 0.859 at M10 while the
400-round version scored 0.727 is counterintuitive (less capacity,
substantially better). Verify the mechanism rather than assuming it.

Report, for the pooled dynamic model in both configurations
(lr=0.05/depth6/400 rounds, and lr=0.1/depth8/11 rounds):

1. **Train AUC, validation AUC, and test AUC** for each. If the
   400-round model shows train AUC far above test (e.g. >0.90 train vs
   ~0.73 test) while the 11-round model shows them close together, the
   overfitting explanation is confirmed. Report the actual numbers.
2. **Confirm both were evaluated on identical test rows** — same
   per-checkpoint populations, same feature columns, same label column
   (v3 train+val-median labels). Rule out an evaluation mismatch as the
   source of the +0.13 jump.
3. Plot the validation AUC learning curve (AUC vs. boosting round, 1-100
   rounds) for the pooled model at lr=0.1/depth=8, so the early-stopping
   point is visible rather than asserted. Save to
   `reports/pooled_early_stopping_curve.png`.

## Check 2 — Tune per-checkpoint models individually (fixes the asymmetry)

Currently pooled was tuned on its own validation data while
per-checkpoint inherited pooled's config — the mirror image of the error
corrected in B4. Give per-checkpoint a fair shot.

For **three checkpoints only** — M1, M5, M10 (enough to detect whether it
matters; do not do all ten) — run the same small grid on validation data
drawn from *that checkpoint's* rows, select by validation AUC, retrain on
that checkpoint's train+validation rows, and evaluate on that
checkpoint's test rows.

Report per checkpoint: selected config, validation AUC, test AUC, and the
comparison against (a) the pooled tuned model and (b) the per-checkpoint
model using pooled's inherited config.

## Step 3 — Final verdict

Based on both checks, state one of:

- **"Pooled and per-checkpoint are equivalent when each is properly
  tuned"** — if individually-tuned per-checkpoint models land within
  ~0.01 AUC of pooled. Then recommend pooled for simplicity (one model,
  not ten) and report the equivalence as the finding.
- **"Per-checkpoint is genuinely better when properly tuned"** — if
  individually-tuned per-checkpoint models clearly beat pooled.
- **"Pooled is genuinely better"** — if per-checkpoint remains behind
  even with its own tuned configs.

Also confirm the M3 crossover holds in the final configuration.

## Report back

Train/val/test AUCs for both pooled configs, confirmation there was no
evaluation mismatch, the early-stopping curve, the three
individually-tuned per-checkpoint results, and the final verdict with the
M3 crossover confirmed.

**After this, the model work is done.** No further tuning rounds.
