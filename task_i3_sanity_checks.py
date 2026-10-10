"""
Task I3: sanity checks on the +activity gain before trusting it.
 1. Gain importance of the B model (which new features carry it?)
 2. Group ablation: add each activity group alone to the baseline
 3. Recency-robustness: does the gain survive when trained on TRAIN only
    (no validation data), i.e. is it independent of the H3 recency effect?
 4. C vs B (recency weighting) paired bootstrap on seed-mean predictions
All fits: B's selected config (lr=0.05, depth=8, 32 trees), 2 seeds.
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from h_common import LABEL, REPORTS_DIR, seeded, paired_bootstrap_seeds
from task_i2_model_upgrades import load, BASE, ACT

CFG = dict(learning_rate=0.05, max_depth=8, n_estimators=32)
GROUPS = {
    "comments": ["n_comments", "n_comment_authors", "comments_last_7d", "log_min_since_last_comment"],
    "status_history": ["n_transitions", "n_reopens", "n_back_moves", "ever_in_progress", "ever_in_review",
                       "status_bucket", "log_min_since_last_change"],
    "last_activity": ["log_min_since_last_activity"],
    "workload_at_T": ["project_open_at_T", "project_resolved_30d", "assignee_open_at_T"],
    "text_stats": ["log_title_len", "log_desc_len", "desc_has_code"],
}

def run(fit, test, cols, seeds=(0, 1)):
    P = []
    for s in seeds:
        m = xgb.XGBClassifier(**CFG, **seeded(s)); m.fit(fit[cols], fit[LABEL])
        P.append(m.predict_proba(test[cols])[:, 1])
    return P, m

def per_ck(test, P):
    p = np.mean(P, axis=0); y = test[LABEL].to_numpy(); c = test["checkpoint_index"].to_numpy()
    return {"all": roc_auc_score(y, p), **{f"M{t}": roc_auc_score(y[c == t], p[c == t]) for t in (1, 5, 10)}}

ck = load()
train = ck[ck.split == "train"]; tv = ck[ck.split.isin(["train", "val"])]
test = ck[ck.split == "test"].reset_index(drop=True)

rows = []
PB, mB = run(tv, test, BASE + ACT)
imp = pd.Series(mB.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
imp = (imp / imp.sum()).rename("gain_share")
imp.to_csv(f"{REPORTS_DIR}/i3_gain_importance.csv")
print("Top 20 features by gain share (B model):"); print(imp.head(20).round(4).to_string())

PA, _ = run(tv, test, BASE)
rows.append({"setting": "baseline (train+val)", **per_ck(test, PA)})
for g, cols in GROUPS.items():
    P, _ = run(tv, test, BASE + cols)
    rows.append({"setting": f"baseline + {g}", **per_ck(test, P)})
rows.append({"setting": "baseline + ALL activity (train+val)", **per_ck(test, PB)})
PA_tr, _ = run(train, test, BASE); PB_tr, _ = run(train, test, BASE + ACT)
rows.append({"setting": "baseline (TRAIN only)", **per_ck(test, PA_tr)})
rows.append({"setting": "baseline + ALL activity (TRAIN only)", **per_ck(test, PB_tr)})
abl = pd.DataFrame(rows); abl.to_csv(f"{REPORTS_DIR}/i3_ablation.csv", index=False)
print("\nAblation (test AUC):"); print(abl.round(4).to_string(index=False))

z = np.load(f"{REPORTS_DIR}/i2_test_predictions.npz")
y, c = z["y"], z["checkpoint"]
print("\nC (recency) vs B, seed-mean predictions, 95% CI:")
for sc in ["all", 1, 5, 10]:
    m = np.ones(len(y), bool) if sc == "all" else c == sc
    bs = paired_bootstrap_seeds(y[m], [z["C_activity_recency"][m]], [z["B_activity"][m]], n_boot=1000)
    print(f"  {sc}: {bs['d_auc']:+.4f} [{bs['d_auc_lo']:+.4f}, {bs['d_auc_hi']:+.4f}]")
