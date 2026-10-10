"""
Task I1: checkpoint-time ACTIVITY features (new signal for the dynamic model).

For every checkpoint row (Issue_ID, checkpoint_index) the checkpoint moment is
    T = Creation_Date + elapsed_minutes
and every feature below uses ONLY events with timestamp strictly < T. Rows
only exist while the issue is unresolved at T (build_phase8_foundation.py),
so nothing here can see the resolution.

Comments (comment.csv)
  n_comments, n_comment_authors, comments_last_7d,
  log_min_since_last_comment   (since creation if no comment yet)
Status history (change_log_status.csv, genuine transitions only = both
From and To non-null, same filter as Phase 8)
  n_transitions, n_reopens, n_back_moves (moved to an earlier workflow stage),
  ever_in_progress, ever_in_review, status_bucket (current status at T,
  bucketed into 8 workflow stages), log_min_since_last_change
  NOTE: Phase 8's current_status falls back to the issue's FINAL status when it
  has no transitions at all. That would leak, so here such issues get
  status_bucket = "no_history".
Workload at T (issue_slim: all 458,232 issues, incl. unresolved)
  project_open_at_T     issues in the project created < T and not resolved by T
  project_resolved_30d  issues in the project resolved in [T-30d, T)
  assignee_open_at_T    same "open" count for the issue's assignee
Creation-time text stats (issue_slim)
  log_title_len, log_desc_len, desc_has_code

Leakage tests at the end: max |corr| with label, and an explicit assertion
that no counted event is at/after T.
"""

import numpy as np
import pandas as pd

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"
OUT = f"{FEATURES_DIR}/phase8_activity_features.csv"
SCALE = 10**10
import os


def raw_path(*names):
    """First existing file among candidates (Mac layout: data/raw/*.csv; cloud: *.gz / data/stage_tmp)."""
    for n in names:
        if os.path.exists(n):
            return n
    raise FileNotFoundError(names)


def load_issue_slim():
    p = [f"{RAW_DIR}/issue_slim.csv.gz", "data/stage_tmp/issue_slim.csv.gz"]
    for c in p:
        if os.path.exists(c):
            return pd.read_csv(c)
    df = pd.read_csv(f"{RAW_DIR}/issue.csv", usecols=["ID", "Project_ID", "Assignee_ID", "Creation_Date",
                                                       "Resolution_Date", "Title", "Description_Text"])
    df["title_len"] = df["Title"].fillna("").str.len()
    df["desc_len"] = df["Description_Text"].fillna("").str.len()
    df["desc_has_code"] = df["Description_Text"].fillna("").str.contains(
        r"\{code|\{noformat|```|Exception|at [a-z]+\.[a-z]", regex=True).astype(int)
    return df.drop(columns=["Title", "Description_Text"])  # key = entity * SCALE + seconds-since-epoch (seconds < 2e9)

STAGE = {"todo": 0, "reopened": 0, "in_progress": 1, "waiting": 1, "review": 2, "testing": 3, "done": 4}


def bucket(s):
    s = str(s).lower()
    if "reopen" in s:
        return "reopened"
    if any(w in s for w in ["closed", "resolved", "done", "won't", "wont", "fixed", "complete", "invalid", "duplicate"]):
        return "done"
    if any(w in s for w in ["review", " cr", "cr ", "reviewable", "quality"]):
        return "review"
    if any(w in s for w in ["test", "verif", "qa"]):
        return "testing"
    if any(w in s for w in ["wait", "await", "block", "hold", "input", "pending", "feedback", "info"]):
        return "waiting"
    if any(w in s for w in ["progress", "develop", "investigat", "implement", "working", "coding"]):
        return "in_progress"
    if any(w in s for w in ["open", "to do", "todo", "new", "backlog", "triage", "gather", "schedul",
                            "accept", "selected", "ready", "planned", "unconfirmed", "confirmed"]):
        return "todo"
    return "other"


def secs(ts):
    dt = pd.to_datetime(pd.Series(ts).reset_index(drop=True), format="ISO8601")
    return ((dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def count_before(ev_keys, q_keys, base_keys):
    """# events with key in [base, q) for sorted ev_keys -> (count, index of last)."""
    hi = np.searchsorted(ev_keys, q_keys, side="left")
    lo = np.searchsorted(ev_keys, base_keys, side="left")
    return hi - lo, hi - 1


def main():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv",
                     usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    issue = load_issue_slim()[["ID", "Project_ID", "Assignee_ID", "Creation_Date", "Resolution_Date",
                                "title_len", "desc_len", "desc_has_code"]].copy()
    issue["c_s"] = secs(issue["Creation_Date"])
    issue["r_s"] = np.where(issue["Resolution_Date"].notna(),
                            secs(issue["Resolution_Date"].fillna("2100-01-01 00:00:00")), np.iinfo(np.int64).max // 4)
    ck = ck.merge(issue[["ID", "Project_ID", "Assignee_ID", "c_s", "title_len", "desc_len", "desc_has_code"]],
                  left_on="Issue_ID", right_on="ID", how="left")
    assert ck["c_s"].notna().all()
    ck["T"] = ck["c_s"] + np.floor(ck["elapsed_minutes"] * 60).astype(np.int64)
    iid = ck["Issue_ID"].to_numpy(np.int64)
    T = ck["T"].to_numpy(np.int64)
    q = iid * SCALE + T
    base = iid * SCALE
    print(f"checkpoint rows: {len(ck)}")

    # ---------------- comments ----------------
    cm = pd.read_csv(f"{RAW_DIR}/comment.csv")
    cm = cm[cm["Issue_ID"].isin(set(iid))].copy()
    # 751 TAWOS comment rows use a 2-digit year ("14-02-05 21:04:00" = 2014-02-05)
    short = cm["Creation_Date"].str.match(r"^\d{2}-\d{2}-\d{2} ")
    cm.loc[short, "Creation_Date"] = "20" + cm.loc[short, "Creation_Date"]
    print(f"  fixed {short.sum()} two-digit-year comment dates")
    cm["t"] = secs(cm["Creation_Date"])
    cm = cm.sort_values(["Issue_ID", "t"], kind="mergesort").reset_index(drop=True)
    cm["first_by_author"] = (~cm.duplicated(["Issue_ID", "Author_ID"])).astype(int)
    ck_keys = cm["Issue_ID"].to_numpy(np.int64) * SCALE + cm["t"].to_numpy(np.int64)
    n_c, last = count_before(ck_keys, q, base)
    cum_auth = np.concatenate([[0], np.cumsum(cm["first_by_author"].to_numpy())])
    lo = np.searchsorted(ck_keys, base, side="left")
    ck["n_comments"] = n_c
    ck["n_comment_authors"] = cum_auth[last + 1] - cum_auth[lo]
    n7, _ = count_before(ck_keys, q, np.maximum(base, q - 7 * 86400))
    ck["comments_last_7d"] = n7
    last_t = np.where(n_c > 0, cm["t"].to_numpy()[np.clip(last, 0, None)], ck["c_s"].to_numpy())
    ck["log_min_since_last_comment"] = np.log1p(np.maximum(T - last_t, 0) / 60)
    assert (last_t[n_c > 0] < T[n_c > 0]).all(), "comment at/after checkpoint counted"
    print(f"  comments: mean per row {n_c.mean():.2f}, rows with >=1: {(n_c > 0).mean():.3f}")

    # ---------------- status history ----------------
    cl = pd.read_csv(raw_path(f"{RAW_DIR}/change_log_status.csv", f"{RAW_DIR}/change_log_status.csv.gz"))
    cl = cl[cl["From_String"].notna() & cl["To_String"].notna() & cl["Issue_ID"].isin(set(iid))].copy()
    cl["t"] = secs(cl["Creation_Date"])
    cl = cl.sort_values(["Issue_ID", "t"], kind="mergesort").reset_index(drop=True)
    cl["to_b"] = cl["To_String"].map(bucket)
    cl["from_b"] = cl["From_String"].map(bucket)
    cl["is_reopen"] = (cl["to_b"] == "reopened").astype(int)
    cl["is_back"] = (cl["to_b"].map(STAGE).fillna(-1) < cl["from_b"].map(STAGE).fillna(-1)).astype(int)
    cl["is_prog"] = (cl["to_b"] == "in_progress").astype(int)
    cl["is_rev"] = cl["to_b"].isin(["review", "testing"]).astype(int)
    keys = cl["Issue_ID"].to_numpy(np.int64) * SCALE + cl["t"].to_numpy(np.int64)
    n_t, last = count_before(keys, q, base)
    lo = np.searchsorted(keys, base, side="left")
    ck["n_transitions"] = n_t
    for col, name in [("is_reopen", "n_reopens"), ("is_back", "n_back_moves"),
                      ("is_prog", "ever_in_progress"), ("is_rev", "ever_in_review")]:
        cum = np.concatenate([[0], np.cumsum(cl[col].to_numpy())])
        v = cum[last + 1] - cum[lo]
        ck[name] = v if name.startswith("n_") else (v > 0).astype(int)
    has_hist = np.isin(iid, cl["Issue_ID"].unique())
    first_from = cl.groupby("Issue_ID")["from_b"].first()
    cur = np.where(n_t > 0, cl["to_b"].to_numpy()[np.clip(last, 0, None)],
                   ck["Issue_ID"].map(first_from).fillna("no_history").to_numpy())
    cur = np.where(has_hist, cur, "no_history")
    ck["status_bucket"] = cur
    last_ct = np.where(n_t > 0, cl["t"].to_numpy()[np.clip(last, 0, None)], ck["c_s"].to_numpy())
    ck["log_min_since_last_change"] = np.log1p(np.maximum(T - last_ct, 0) / 60)
    last_act = np.maximum(last_t, last_ct)
    ck["log_min_since_last_activity"] = np.log1p(np.maximum(T - last_act, 0) / 60)
    assert (last_ct[n_t > 0] < T[n_t > 0]).all(), "status change at/after checkpoint counted"
    print("  status_bucket distribution:\n" + ck["status_bucket"].value_counts(normalize=True).round(3).to_string())

    # ---------------- workload at T ----------------
    def open_at(entity_issue_col, entity_ck):
        df = issue[issue[entity_issue_col].notna()]
        e = df[entity_issue_col].astype(np.int64).to_numpy()
        ck_e = ck[entity_ck]
        valid = ck_e.notna().to_numpy()
        ee = np.where(valid, ck_e.fillna(-1).astype(np.int64).to_numpy(), 0)
        c_keys = np.sort(e * SCALE + df["c_s"].to_numpy())
        has_r = df["r_s"].to_numpy() < np.iinfo(np.int64).max // 8
        r_keys = np.sort(e[has_r] * SCALE + df["r_s"].to_numpy()[has_r])
        qq, bb = ee * SCALE + T, ee * SCALE
        created, _ = count_before(c_keys, qq, bb)
        resolved, _ = count_before(r_keys, qq, bb)
        res30, _ = count_before(r_keys, qq, np.maximum(bb, qq - 30 * 86400))
        return np.where(valid, created - resolved, np.nan), np.where(valid, res30, np.nan)

    ck["project_open_at_T"], ck["project_resolved_30d"] = open_at("Project_ID", "Project_ID")
    ck["assignee_open_at_T"], _ = open_at("Assignee_ID", "Assignee_ID")
    for c in ["project_open_at_T", "project_resolved_30d", "assignee_open_at_T"]:
        ck[c] = np.log1p(ck[c].clip(lower=0))

    ck["log_title_len"] = np.log1p(ck["title_len"])
    ck["log_desc_len"] = np.log1p(ck["desc_len"])

    new_cols = ["n_comments", "n_comment_authors", "comments_last_7d", "log_min_since_last_comment",
                "n_transitions", "n_reopens", "n_back_moves", "ever_in_progress", "ever_in_review",
                "status_bucket", "log_min_since_last_change", "log_min_since_last_activity",
                "project_open_at_T", "project_resolved_30d", "assignee_open_at_T",
                "log_title_len", "log_desc_len", "desc_has_code"]
    out = ck[["Issue_ID", "checkpoint_index"] + new_cols]
    out.to_csv(OUT, index=False)
    print(f"saved {out.shape} -> {OUT}")

    # ---------------- leakage test ----------------
    lab = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2"])
    chk = out.merge(lab, left_on="Issue_ID", right_on="ID")
    num = [c for c in new_cols if c != "status_bucket"]
    corr = chk[num].corrwith(chk["delayed_v2"]).abs().sort_values(ascending=False)
    print("\n|corr| with delayed_v2 (pooled rows):\n" + corr.round(3).to_string())
    print("\ndelay rate by status_bucket:\n" +
          chk.groupby("status_bucket")["delayed_v2"].agg(["mean", "size"]).round(3).to_string())
    assert corr.max() < 0.95, "LEAKAGE: feature near-perfectly correlated with label"
    print("\nLeakage test: PASS (no event at/after T; max |corr| %.3f)" % corr.max())


if __name__ == "__main__":
    main()
