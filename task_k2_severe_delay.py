"""
Task K2: a second, stricter target -- SEVERE delay.

    severe = Resolution_Time > 2 x project median   (median from train+val, as for delayed_v2)

"Will this take more than twice what this project normally takes?" is the
question that matters most to a PM (the issues that blow up a plan), and big
overruns leave clearer traces in the activity data.

Same v5b feature set and protocol: recency weights (4y half-life), config
re-selected on validation (3 configs x 2 seeds, early stopping), refit on
train+val with 5 seeds, scored on test. One extra knowable-at-T feature,
past_2x_median, mirrors past_median for the new threshold. Reported on all
rows and on rows where the severe label is still undetermined at T.
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import REPORTS_DIR, MODELS_DIR, SEEDS, seeded, seed_metric_panel, paired_bootstrap_seeds
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

LABEL = "severe"
HL = 4.0
GRID = [{"learning_rate": 0.1, "max_depth": 6}, {"learning_rate": 0.1, "max_depth": 8},
        {"learning_rate": 0.05, "max_depth": 8}]


def main():
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_pooled_v5b_meta.json"))
    cols = meta["features"] + ["past_2x_median"]
    ck = load_all()
    feat = pd.read_csv("data/features/phase6_feature_table_v3.csv", usecols=["ID", "Project_ID", "Resolution_Time_Minutes"])
    sp = pd.read_csv("data/features/phase8_split.csv", usecols=["ID", "split"])
    f = feat.merge(sp, on="ID")
    med = f[f.split.isin(["train", "val"])].groupby("Project_ID")["Resolution_Time_Minutes"].median()
    f["severe"] = (f["Resolution_Time_Minutes"] > 2 * f["Project_ID"].map(med)).astype(int)
    print("severe rate by split:", f.groupby("split")["severe"].mean().round(3).to_dict(), flush=True)
    ck = ck.merge(f[["ID", "severe"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    pm = ck["Project_ID"].astype(int).map(med).to_numpy()
    ck["past_2x_median"] = (ck["elapsed_minutes"].astype(float).to_numpy() > 2 * pm).astype(int)

    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    w_tr, w_tv = weights(tr, HL, tr["age_years"].min()), weights(tv, HL, tv["age_years"].min())
    best = None
    for cfg in GRID:
        aucs, it = [], []
        for s in [0, 1]:
            m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(tr[cols], tr[LABEL], sample_weight=w_tr, eval_set=[(va[cols], va[LABEL])], verbose=False)
            aucs.append(roc_auc_score(va[LABEL], m.predict_proba(va[cols])[:, 1]))
            it.append(m.best_iteration + 1)
        print(f"  {cfg}: val AUC {np.mean(aucs):.4f}, trees {int(np.median(it))}", flush=True)
        if best is None or np.mean(aucs) > best[0]:
            best = (np.mean(aucs), {**cfg, "n_estimators": max(int(np.median(it)), 10)})
    cfg = best[1]
    P, models = [], []
    for s in SEEDS:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(tv[cols], tv[LABEL], sample_weight=w_tv)
        P.append(m.predict_proba(te[cols])[:, 1])
        joblib.dump(m, f"{MODELS_DIR}/xgb_severe_v5b_seed{s}.joblib")
    json.dump({"features": cols, "config": cfg, "label": "Resolution_Time > 2 x train+val project median",
               "recency_half_life_years": HL, "seeds": SEEDS},
              open(f"{MODELS_DIR}/xgb_severe_v5b_meta.json", "w"), indent=2)

    y, c = te[LABEL].to_numpy(), te["checkpoint_index"].to_numpy()
    und = te["past_2x_median"].to_numpy() == 0
    safe = te["safe"].to_numpy(bool)
    rows = []
    for sub, m0 in [("all", np.ones(len(y), bool)), ("undetermined", und), ("safe & undetermined", safe & und)]:
        for sc in ["all"] + [f"M{t}" for t in range(1, 11)]:
            m = m0 if sc == "all" else m0 & (c == int(sc[1:]))
            rows.append(seed_metric_panel(y[m], [p[m] for p in P], subset=sub, scope=sc))
    panel = pd.DataFrame(rows)
    panel.to_csv(f"{REPORTS_DIR}/k2_severe_metrics.csv", index=False)
    pe = np.mean(P, axis=0)
    prec = []
    for sub, m0 in [("all", np.ones(len(y), bool)), ("undetermined", und)]:
        for sc in ["all", "M1", "M5", "M10"]:
            m = m0 if sc == "all" else m0 & (c == int(sc[1:]))
            for frac in (0.05, 0.10, 0.20):
                k = max(int(len(y[m]) * frac), 1)
                idx = np.argsort(-pe[m])[:k]
                prec.append({"subset": sub, "scope": sc, "top": f"{int(frac*100)}%",
                             "precision": y[m][idx].mean(), "base_rate": y[m].mean()})
    prec = pd.DataFrame(prec)
    prec.to_csv(f"{REPORTS_DIR}/k2_severe_precision_at_top.csv", index=False)
    print(f"\nselected {cfg}")
    print(panel[panel.scope.isin(["all", "M1", "M3", "M5", "M8", "M10"])][
        ["subset", "scope", "N", "pos_rate", "auc", "accuracy", "majority_accuracy", "balanced_accuracy", "f1",
         "pr_auc", "brier"]].round(3).to_string(index=False))
    print("\nPrecision of top flags (severe):\n" + prec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
