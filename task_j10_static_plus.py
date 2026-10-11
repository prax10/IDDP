"""
Task J10: fairness check for "dynamic beats static from M1".

The v5b dynamic model has extra NON-temporal information that a creation-time
model could also have had (track records known at creation, workload at
creation, text length, story points at creation, ...). So M1's win could be
partly "better features" rather than "observing progress". static+ gives the
static model every feature that is computable AT CREATION (FeatureEngine at
T = creation + 1 minute), re-selected on validation, 5 seeds, and compares it
with v5b on the identical checkpoint test rows.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from feature_engine import FeatureEngine
from h_common import LABEL, REPORTS_DIR, SEEDS, seeded, static_cols, load_static, paired_bootstrap_seeds, verdict

AT_CREATION = ["proj_rate_known", "proj_rate_known_90d", "proj_n_resolved_90d", "assignee_rate_known",
               "assignee_n_known", "reporter_rate_known", "project_open_at_T", "project_resolved_30d",
               "assignee_open_at_T", "log_title_len", "log_desc_len", "desc_has_code", "sp_at_T",
               "in_sprint_at_T", "assigned_at_T", "n_link_chg"]
GRID = [{"learning_rate": lr, "max_depth": d} for lr in [0.05, 0.1, 0.2] for d in [4, 6, 8]]


def main():
    fe = FeatureEngine()
    st = load_static()
    c_s = fe.iss["c_s"].reindex(st["ID"]).to_numpy()
    X = fe.at(st["ID"].to_numpy(), c_s + 60)
    for c in AT_CREATION:
        st[c] = X[c].to_numpy()
    cols = static_cols(30) + AT_CREATION
    tr, va = st[st.split == "train"], st[st.split == "val"]
    tv = st[st.split.isin(["train", "val"])]
    rows = []
    for cfg in GRID:
        aucs, it = [], []
        for s in [0, 1]:
            m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(tr[cols], tr[LABEL], eval_set=[(va[cols], va[LABEL])], verbose=False)
            aucs.append(roc_auc_score(va[LABEL], m.predict_proba(va[cols])[:, 1]))
            it.append(m.best_iteration + 1)
        rows.append({**cfg, "val_auc": np.mean(aucs), "trees": int(np.median(it))})
    g = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b.learning_rate), "max_depth": int(b.max_depth), "n_estimators": max(int(b.trees), 10)}
    print("static+ selected", cfg, "val", round(b.val_auc, 4), flush=True)

    # score on the dynamic test checkpoint rows (static features do not change across checkpoints)
    from task_j4_v5_model import load_all
    ck = load_all()
    te = ck[ck.split == "test"].reset_index(drop=True)
    te = te.merge(st[["ID"] + AT_CREATION].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left",
                  suffixes=("_dyn", ""))
    y, c = te[LABEL].to_numpy(), te["checkpoint_index"].to_numpy()
    P_plus, P_st = [], []
    from h_common import TUNED_STATIC
    for s in SEEDS:
        m = xgb.XGBClassifier(**cfg, **seeded(s)); m.fit(tv[cols], tv[LABEL]); P_plus.append(m.predict_proba(te[cols])[:, 1])
        m = xgb.XGBClassifier(**TUNED_STATIC, **seeded(s)); m.fit(tv[static_cols(30)], tv[LABEL])
        P_st.append(m.predict_proba(te[static_cols(30)])[:, 1])
    z = np.load(f"{REPORTS_DIR}/j4_test_predictions.npz")
    assert (z["y"] == y).all()
    p_v5b = z["v5b"]   # seed-averaged
    out = []
    for t in range(1, 11):
        msk = c == t
        bs = paired_bootstrap_seeds(y[msk], [p_v5b[msk]], [np.mean(P_plus, 0)[msk]], n_boot=1000, seed=t)
        out.append({"checkpoint": f"M{t}", "static_auc": np.mean([roc_auc_score(y[msk], p[msk]) for p in P_st]),
                    "static_plus_auc": np.mean([roc_auc_score(y[msk], p[msk]) for p in P_plus]),
                    "v5b_auc_ensemble": roc_auc_score(y[msk], p_v5b[msk]),
                    "v5b_minus_static_plus": bs["d_auc"], "lo": bs["d_auc_lo"], "hi": bs["d_auc_hi"],
                    "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
    res = pd.DataFrame(out)
    res.to_csv(f"{REPORTS_DIR}/j10_static_plus_vs_v5b.csv", index=False)
    st_te = st[st.split == "test"]
    pp = [xgb.XGBClassifier(**cfg, **seeded(s)).fit(tv[cols], tv[LABEL]).predict_proba(st_te[cols])[:, 1] for s in [0, 1]]
    print(f"static+ issue-level test AUC: {np.mean([roc_auc_score(st_te[LABEL], p) for p in pp]):.4f}")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
