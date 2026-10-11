"""
Task L6b: stricter observational check of the recommender's directions.
Same 20,000 test rows as L6, but only issues STILL OPEN 7 days later, so "it reached review
because it was finishing anyway" cannot drive the comparison. Risk-decile matched as in L6.
"""
import numpy as np, pandas as pd
from app_v5_backend import Backend
from h_common import REPORTS_DIR
WEEK = 7 * 86400

b = Backend()
ck = pd.read_csv("data/features/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes", "split"])
te = ck[(ck.split == "test") & ck.checkpoint_index.between(2, 6)].sample(20000, random_state=0).reset_index(drop=True)
c_s = b.fe.iss["c_s"].reindex(te["Issue_ID"]).to_numpy()
T = c_s + np.floor(te["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
X = b._design(te["Issue_ID"].to_numpy(), T)
y = b.static.reindex(te["Issue_ID"])["delayed_v2"].to_numpy()
base = b._predict(b.live, X, b.cols)
X7 = b.fe.at(te["Issue_ID"].to_numpy(), T + WEEK)
open7 = b.fe.iss["r_s"].reindex(te["Issue_ID"]).to_numpy() > T + WEEK
stage = X["status_bucket"].astype(str).to_numpy()
acts = {"start_work": (np.isin(stage, ["todo", "no_history", "reopened", "waiting", "other"]),
                       X7["ever_in_progress"].to_numpy() > X["ever_in_progress"].to_numpy()),
        "move_to_review": (~np.isin(stage, ["review", "testing", "done"]),
                           X7["ever_in_review"].to_numpy() > X["ever_in_review"].to_numpy()),
        "discuss": (np.ones(len(X), bool), X7["n_comments"].to_numpy() > X["n_comments"].to_numpy()),
        "reassign": (np.ones(len(X), bool), X7["n_assignee_changes"].to_numpy() > X["n_assignee_changes"].to_numpy())}
dec = pd.qcut(base, 10, labels=False, duplicates="drop")
rows = []
for a, (app, did) in acts.items():
    for scope, extra in [("all (as L6)", np.ones(len(X), bool)), ("still open after 7 days", open7)]:
        m0 = app & extra
        diffs, wts = [], []
        for k in np.unique(dec):
            m = m0 & (dec == k)
            t1, t0 = m & did, m & ~did
            if t1.sum() >= 20 and t0.sum() >= 20:
                diffs.append(y[t1].mean() - y[t0].mean()); wts.append(m.sum())
        rows.append({"action": a, "scope": scope, "rows": int(m0.sum()), "took_share": did[m0].mean(),
                     "late_rate_diff_did_minus_didnt": np.average(diffs, weights=wts) if diffs else np.nan,
                     "deciles_used": len(diffs)})
r = pd.DataFrame(rows); r.to_csv(f"{REPORTS_DIR}/l6b_strict_check.csv", index=False)
print(f"still open after 7 days: {open7.mean():.1%}"); print(r.round(3).to_string(index=False))
