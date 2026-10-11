"""
Task L4, step 3: external validation -- does the TAWOS model work on a Jira repository it
has never seen?

  python3 l4_transfer_test.py <ext_root> [<ext_root2> ...]

Portable feature set = the v5b live features minus anything tied to TAWOS itself:
  dropped  Project_ID (unseen categories), text_pc_* (TAWOS-fitted SBERT->PCA), and the
           creation-time static columns built by the TAWOS Phase-6 notebook
           (assignee_prior_*, assignee_concurrent_open, is_unassigned, n_links,
           n_blocking_links, Story_Point) -- their dynamic, known-at-T versions stay in
  kept     every FeatureEngine feature (activity, workflow, people, sprint, story points,
           field churn, known-at-T project / assignee / reporter track records),
           elapsed_minutes, Priority_Normalized, Type_Normalized

Experiments (all AUC on the external TEST period, chronological, untouched):
  A  zero-shot      portable model trained on TAWOS train+val only
  B  within-repo    same config trained on the external repo's own train+val
  C  combined       TAWOS train+val + external train+val
  ref               portable model on TAWOS test, and full v5b on TAWOS test (cost of portability)
Also per checkpoint, censoring-safe subset, and a paired bootstrap of A vs B.
"""

import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded, paired_bootstrap_seeds, verdict
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

SEEDS = [0, 1, 2]
DROP = {"Project_ID", "assignee_prior_resolved_count", "assignee_prior_delay_rate", "assignee_concurrent_open",
        "is_unassigned", "n_links", "n_blocking_links", "Story_Point"}
CATS = ["Priority_Normalized", "Type_Normalized", "status_bucket"]


def portable_cols(meta):
    return [c for c in meta["features"] if c not in DROP and not c.startswith("text_pc_")]


def prep(df, cols, cats):
    X = df[cols].copy()
    for c in CATS:
        if c in X:
            X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    for c in X.columns:
        if c not in CATS:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def fit(X, y, w, cfg, seeds=SEEDS):
    return [xgb.XGBClassifier(**cfg, **seeded(s)).fit(X, y, sample_weight=w) for s in seeds]


def pred(models, X):
    return [m.predict_proba(X)[:, 1] for m in models]


def main():
    roots = sys.argv[1:]
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    cols = portable_cols(meta)
    print(f"portable features: {len(cols)} of {len(meta['features'])}", flush=True)

    ck = load_all()
    tw_tv = ck[ck.split.isin(["train", "val"])]
    tw_tr, tw_va = ck[ck.split == "train"], ck[ck.split == "val"]
    tw_te = ck[ck.split == "test"].reset_index(drop=True)
    exts = {r.rstrip("/").split("/")[-1]: pd.read_parquet(f"{r}/features/checkpoint_features.parquet") for r in roots}
    cats = {c: sorted(set(ck[c].astype(str).unique()) | set().union(*[set(e[c].astype(str).unique()) for e in exts.values()]))
            for c in CATS}

    # choose n_estimators for the portable model on TAWOS val (v5b lr/depth)
    base = {k: meta["config"][k] for k in ("learning_rate", "max_depth")}
    w_tr = weights(tw_tr, 4.0, tw_tr["age_years"].min())
    m = xgb.XGBClassifier(**base, n_estimators=2000, early_stopping_rounds=40, **seeded(0))
    m.fit(prep(tw_tr, cols, cats), tw_tr[LABEL], sample_weight=w_tr,
          eval_set=[(prep(tw_va, cols, cats), tw_va[LABEL])], verbose=False)
    cfg = {**base, "n_estimators": m.best_iteration + 1}
    print(f"portable config {cfg}", flush=True)
    del m

    w_tv = weights(tw_tv, 4.0, tw_tv["age_years"].min())
    A = fit(prep(tw_tv, cols, cats), tw_tv[LABEL], w_tv, cfg)
    rows = []
    yt = tw_te[LABEL].to_numpy()
    pa_tw = pred(A, prep(tw_te, cols, cats))
    z = np.load(f"{REPORTS_DIR}/j4_test_predictions.npz")
    rows.append({"dataset": "TAWOS test", "model": "full v5b (reference)", "auc": roc_auc_score(z["y"], z["v5b"])})
    rows.append({"dataset": "TAWOS test", "model": "portable, trained on TAWOS", "auc": np.mean([roc_auc_score(yt, p) for p in pa_tw])})

    for name, e in exts.items():
        e = e.copy()
        tv, te = e[e.split.isin(["train", "val"])], e[e.split == "test"].reset_index(drop=True)
        y = te[LABEL].to_numpy()
        Xte = prep(te, cols, cats)
        pa = pred(A, Xte)
        # B and C are unweighted (external repos span fewer years; recency weighting is a TAWOS-tuned choice)
        B = fit(prep(tv, cols, cats), tv[LABEL], None, cfg)
        pb = pred(B, Xte)
        comb = pd.concat([tw_tv[cols + [LABEL]], tv[cols + [LABEL]]], ignore_index=True)
        C = fit(prep(comb, cols, cats), comb[LABEL], None, cfg)
        pc = pred(C, Xte)
        bs = paired_bootstrap_seeds(y, pa, pb, n_boot=500)
        safe = te["safe"].to_numpy() == 1
        c = te["checkpoint_index"].to_numpy()
        for label, P in [("A zero-shot (TAWOS only)", pa), ("B within-repo (own history only)", pb),
                         ("C combined (TAWOS + own history)", pc)]:
            r = {"dataset": name, "model": label, "n_test_rows": len(y), "n_test_issues": te.Issue_ID.nunique(),
                 "late_rate": y.mean(), "auc": np.mean([roc_auc_score(y, p) for p in P]),
                 "auc_safe_only": np.mean([roc_auc_score(y[safe], p[safe]) for p in P]) if safe.sum() > 100 else np.nan,
                 "accuracy@0.5": np.mean([accuracy_score(y, p >= 0.5) for p in P])}
            for k in (1, 3, 5, 8, 10):
                mk = c == k
                r[f"auc_M{k}"] = np.mean([roc_auc_score(y[mk], p[mk]) for p in P]) if len(np.unique(y[mk])) == 2 else np.nan
            rows.append(r)
        rows[-3].update({"A_minus_B": bs["d_auc"], "lo": bs["d_auc_lo"], "hi": bs["d_auc_hi"],
                         "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
        print(pd.DataFrame(rows[-3:]).round(3).to_string(index=False), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l4_external_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
