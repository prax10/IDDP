"""
recommender.py -- "what would help?" for an open issue (flow mode).

For one issue at moment T, each candidate action is applied to the issue's
feature row and re-scored with the live model (v6, or v5b if v6 is absent):

  start_work      status -> in progress, work started, status just moved
  move_to_review  status -> review, review reached, status just moved
  discuss         a comment now (comments +1, last comment / activity = now)
  reassign        hand the issue to each active colleague in the project (resolved >= 5
                  issues before T, active in the last 180 days); their known-at-T track
                  record, experience and open workload replace the current assignee's

IMPORTANT: these are MODEL-ESTIMATED effects of changing the evidence the model
sees, not causal effects. task_l6_recommendations.py checks them against what
actually happened historically (observational, confounded) before they are shown.
"""

import numpy as np
import pandas as pd

from feature_engine import SCALE

ACTION_LABELS = {"start_work": "Start work on it (move to In Progress)",
                 "move_to_review": "Get it into review",
                 "discuss": "Unblock it: comment / discuss today",
                 "reassign": "Reassign to"}


def _set_status(X, bucket, cats):
    X = X.copy()
    X["status_bucket"] = pd.Categorical([bucket] * len(X), categories=cats)
    X["log_min_since_last_change"] = 0.0
    X["log_min_since_last_activity"] = 0.0
    X["n_transitions"] = X["n_transitions"] + 1
    return X


def candidate_actions(X, cats):
    """X: one-row design frame (Backend._design). Returns {action: modified frame}."""
    out = {}
    stage = str(X["status_bucket"].iloc[0])
    if stage in ("todo", "no_history", "reopened", "waiting", "other"):
        s = _set_status(X, "in_progress", cats)
        s["ever_in_progress"] = 1
        out["start_work"] = s
    if stage not in ("review", "testing", "done"):
        r = _set_status(X, "review", cats)
        r["ever_in_review"] = 1
        r["ever_in_progress"] = 1
        out["move_to_review"] = r
    d = X.copy()
    d["n_comments"] = d["n_comments"] + 1
    d["comments_last_7d"] = d["comments_last_7d"] + 1
    d["log_min_since_last_comment"] = 0.0
    d["log_min_since_last_activity"] = 0.0
    out["discuss"] = d
    return out


def active_colleagues(fe, project, T, current=None, min_resolved=5, active_days=180, top=8):
    """Assignees with a known track record in the project before T, still active recently."""
    r = fe.res[(fe.res["Project_ID"] == project) & (fe.res["r_s"] < T) & fe.res["Assignee_ID"].notna()]
    if r.empty:
        return pd.DataFrame()
    g = r.groupby("Assignee_ID").agg(n=("ID", "size"), delayed=("delayed_v2", "sum"), last=("r_s", "max"))
    g = g[(g.n >= min_resolved) & (g["last"] >= T - active_days * 86400)]
    if current is not None and not pd.isna(current):
        g = g.drop(index=current, errors="ignore")
    if g.empty:
        return g
    g["rate"] = (g["delayed"] + 10 * 0.5) / (g["n"] + 10)
    ids = g.index.astype(np.int64).to_numpy()
    c_keys, r_keys = fe.open_idx["Assignee_ID"]
    q, b = ids * SCALE + T, ids * SCALE
    g["open_now"] = (np.searchsorted(c_keys, q) - np.searchsorted(c_keys, b)) - \
                    (np.searchsorted(r_keys, q) - np.searchsorted(r_keys, b))
    return g.sort_values(["rate", "open_now"]).head(top)


def reassign_variants(X, colleagues):
    out = {}
    for aid, row in colleagues.iterrows():
        v = X.copy()
        v["assignee_rate_known"] = row["rate"]
        v["assignee_prior_delay_rate"] = row["delayed"] / row["n"]
        v["assignee_n_known"] = np.log1p(row["n"])
        v["assignee_prior_resolved_count"] = row["n"]
        v["assignee_open_at_T"] = np.log1p(max(row["open_now"], 0))
        v["assignee_concurrent_open"] = max(row["open_now"], 0)
        v["is_unassigned"] = 0
        v["assigned_at_T"] = 1
        v["n_assignee_changes"] = v["n_assignee_changes"] + 1
        v["ever_reassigned"] = 1
        v["log_min_since_assign_change"] = 0.0
        out[int(aid)] = v
    return out


def recommend(backend, issue_id, date, top_colleagues=8):
    T = int(pd.Timestamp(date).timestamp())
    X = backend._design([issue_id], [T])
    cols = backend.cols
    cats = backend.live_meta["categories"]["status_bucket"]
    base = float(backend._predict(backend.live, X, cols)[0])
    rows = []
    for name, Xa in candidate_actions(X, cats).items():
        p = float(backend._predict(backend.live, Xa, cols)[0])
        rows.append({"action": ACTION_LABELS[name], "new_risk": p, "change": p - base})
    info = backend.fe.iss.loc[issue_id]
    col = active_colleagues(backend.fe, info["Project_ID"], T, current=info["Assignee_ID"], top=top_colleagues)
    if len(col):
        best = None
        for aid, Xa in reassign_variants(X, col).items():
            p = float(backend._predict(backend.live, Xa, cols)[0])
            if best is None or p < best[1]:
                best = (aid, p)
        r = col.loc[best[0]]
        rows.append({"action": f"{ACTION_LABELS['reassign']} assignee {best[0]} "
                               f"({int(r['n'])} resolved here, {r['delayed']/r['n']:.0%} late, {int(r['open_now'])} open)",
                     "new_risk": best[1], "change": best[1] - base})
    return base, pd.DataFrame(rows).sort_values("change").reset_index(drop=True)
