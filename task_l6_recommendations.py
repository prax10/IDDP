"""
Task L6: "what would help?" -- evaluate the recommender before it goes in the app.

On a sample of 20,000 open test issues (checkpoints M2-M6, where a PM can still act):
  1. model-estimated effect of each action (risk change from recommender.py)
  2. an OBSERVATIONAL check against history: did issues that actually took the action in
     the next 7 days end up late less often than similar issues that did not?
     "similar" = same decile of predicted risk at the checkpoint (stratified difference).
     Actions observed in the 7-day window: reached review, started work, got a comment,
     got reassigned. This is not a causal estimate (confounding, reverse causality: work
     that is nearly done moves to review), but it shows whether the model's direction and
     rough size agree with what really happened.
"""

import json
import numpy as np
import pandas as pd

from app_v5_backend import Backend
from recommender import candidate_actions, active_colleagues, reassign_variants
from h_common import REPORTS_DIR

N = 20000
WEEK = 7 * 86400


def main():
    b = Backend()
    ck = pd.read_csv("data/features/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes", "split"])
    te = ck[(ck.split == "test") & ck.checkpoint_index.between(2, 6)].sample(N, random_state=0).reset_index(drop=True)
    c_s = b.fe.iss["c_s"].reindex(te["Issue_ID"]).to_numpy()
    T = c_s + np.floor(te["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    X = b._design(te["Issue_ID"].to_numpy(), T)
    y = b.static.reindex(te["Issue_ID"])["delayed_v2"].to_numpy()
    base = b._predict(b.live, X, b.cols)
    cats = b.live_meta["categories"]["status_bucket"]

    # 1. model-estimated effects (vectorised over all rows where the action applies)
    est = {}
    stage = X["status_bucket"].astype(str).to_numpy()
    for action, applies in [("start_work", np.isin(stage, ["todo", "no_history", "reopened", "waiting", "other"])),
                            ("move_to_review", ~np.isin(stage, ["review", "testing", "done"])),
                            ("discuss", np.ones(len(X), bool))]:
        Xa = candidate_actions(X[applies].reset_index(drop=True), cats)[action]
        d = b._predict(b.live, Xa, b.cols) - base[applies]
        est[action] = (applies, d)
    # reassignment on a 2,000-row subsample (needs per-issue colleague search)
    sub = np.random.default_rng(0).choice(len(X), 2000, replace=False)
    dre = np.full(len(X), np.nan)
    for i in sub:
        info = b.fe.iss.loc[te.Issue_ID[i]]
        col = active_colleagues(b.fe, info["Project_ID"], int(T[i]), current=info["Assignee_ID"])
        if len(col):
            ps = [b._predict(b.live, v, b.cols)[0] for v in reassign_variants(X.iloc[[i]], col).values()]
            dre[i] = min(ps) - base[i]
    est["reassign (best colleague)"] = (~np.isnan(dre), dre[~np.isnan(dre)])

    # 2. what actually happened in the next 7 days
    X7 = b.fe.at(te["Issue_ID"].to_numpy(), T + WEEK)
    resolved_7 = b.fe.iss["r_s"].reindex(te["Issue_ID"]).to_numpy() <= T + WEEK
    did = {"move_to_review": (X7["ever_in_review"].to_numpy() > X["ever_in_review"].to_numpy()),
           "start_work": (X7["ever_in_progress"].to_numpy() > X["ever_in_progress"].to_numpy()),
           "discuss": (X7["n_comments"].to_numpy() > X["n_comments"].to_numpy()),
           "reassign (best colleague)": (X7["n_assignee_changes"].to_numpy() > X["n_assignee_changes"].to_numpy())}
    dec = pd.qcut(base, 10, labels=False, duplicates="drop")
    rows = []
    for action, (applies, d) in est.items():
        a = applies if action != "reassign (best colleague)" else np.ones(len(X), bool)
        took = did[action] & a
        diffs, wts = [], []
        for k in np.unique(dec):
            m = a & (dec == k)
            t1, t0 = m & took, m & ~took
            if t1.sum() >= 20 and t0.sum() >= 20:
                diffs.append(y[t1].mean() - y[t0].mean())
                wts.append(m.sum())
        obs = np.average(diffs, weights=wts) if diffs else np.nan
        rows.append({"action": action, "rows_where_applicable": int(applies.sum()),
                     "model_mean_risk_change": float(np.mean(d)), "model_share_lowering_risk": float((d < 0).mean()),
                     "took_action_within_7d_share": float(took[a].mean()),
                     "observed_late_rate_difference_(did - didnt, risk-matched)": obs,
                     "same_direction": bool(np.sign(obs) == np.sign(np.mean(d))) if diffs else None})
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l6_recommendation_check.csv", index=False)
    print(f"resolved within the 7-day window: {resolved_7.mean():.1%}")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
