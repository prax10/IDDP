"""
Task N1: v7 features. Every value for a checkpoint row uses only information timestamped strictly
before the checkpoint moment T = Creation_Date + elapsed_minutes.

A. Links AT T (replaces the static n_links / n_blocking_links, which count the issue's FINAL links
   from Issue_Link, including links added after T)
   Rebuilt from Change_Log "Link" events ("This issue <phrase> <KEY>", removals in From_String).
     n_links_at_T, n_blockedby_at_T, n_blocks_at_T, n_dup_at_T, log_min_since_link_change
   Jira keys are mapped to issue IDs by matching (issue, phrase) against Issue_Link, so a blocker's
   own status at T can be read:
     n_open_blockers_at_T     blockers linked before T and still unresolved at T
     n_blockers_resolved_7d   blockers resolved in [T-7d, T)
     any_open_blocker
B. Fix version AT T (the closest thing to a deadline TAWOS has)
   Current fix version = the most recently added version still assigned at T.
     has_fixver_at_T, n_fixver_at_T, fixver_placeholder (name has no digit: "Backlog", "Future", ...)
   Stats of that version over ALL issues, counted only from events before T:
     log_fixver_assigned      issues assigned to it before T
     fixver_done_share        of those, share assigned AND resolved before T
     log_fixver_resolved_14d  its issues resolved in [T-14d, T)
     log_fixver_age_days      time since the version was first used
     log_fixver_days_left_est open issues / recent resolution rate (a velocity-based "release ETA")
C. Assignee AT T (the base features use the FINAL assignee, who may only arrive after T)
   assignee_at_T reconstructed from the assignee change log (first event's From_String before the
   first change, last To_String after). Recomputed on it:
     assignee_rate_known_T, assignee_n_known_T    smoothed late rate of their issues resolved before T
     a_rate_90d, a_n_res_30d, a_n_res_90d         recent form
     a_last5_late                                  late share of their last 5 resolved issues
     log_a_days_since_res                          time since they last resolved anything
     is_unassigned_T
"""
import numpy as np
import pandas as pd

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"
OUT = f"{FEATURES_DIR}/phase8_v7_features.csv"
SCALE = 10**10
FAR = np.int64(4 * 10**9)
DAY = 86400

BLOCKED_BY = {"is blocked by", "depends on", "has to be done after", "is caused by", "is a regression caused by"}
BLOCKS = {"blocks", "is depended on by", "has to be done before", "causes", "caused a regression"}
DUP = {"duplicates", "is duplicated by"}


def secs(s):
    s = pd.Series(s).reset_index(drop=True).astype(str)
    short = s.str.match(r"^\d{2}-\d{2}-\d{2} ")
    s[short] = "20" + s[short]
    dt = pd.to_datetime(s.where(s != "nan"), format="ISO8601", errors="coerce")
    out = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return out.fillna(-1).to_numpy(np.int64)


def pair_events(ev, key_cols):
    """ev: rows with key_cols, t, add(bool). Match the k-th add of a key to its k-th removal.
    Returns key_cols + a (add time) + d (removal time, FAR if never removed)."""
    ev = ev.sort_values(key_cols + ["t"])
    adds = ev[ev["add"]].copy()
    rems = ev[~ev["add"]].copy()
    adds["k"] = adds.groupby(key_cols).cumcount()
    rems["k"] = rems.groupby(key_cols).cumcount()
    p = adds.merge(rems[key_cols + ["k", "t"]].rename(columns={"t": "d"}), on=key_cols + ["k"], how="left")
    p["d"] = p["d"].fillna(FAR).astype(np.int64)
    return p.rename(columns={"t": "a"})


def rows_x_pairs(R, P, on):
    """Join checkpoint rows R (with row id 'r', 'T') to pairs P on `on`."""
    return R.merge(P, on=on, how="inner")


def main():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    iss = pd.read_csv(f"{RAW_DIR}/issue_slim.csv.gz")
    iss["c_s"] = secs(iss["Creation_Date"])
    iss["r_s"] = np.where(iss["Resolution_Date"].notna(), secs(iss["Resolution_Date"]), FAR)
    rmap = iss.set_index("ID")["r_s"]
    ck = ck.merge(iss[["ID", "Project_ID", "Assignee_ID", "c_s"]], left_on="Issue_ID", right_on="ID", how="left")
    T = ck["c_s"].to_numpy(np.int64) + np.floor(ck["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    ck["T"] = T
    ck["r"] = np.arange(len(ck))
    iid = ck["Issue_ID"].to_numpy(np.int64)
    F = pd.DataFrame({"Issue_ID": iid, "checkpoint_index": ck["checkpoint_index"].to_numpy()})
    R = ck[["r", "Issue_ID", "T", "Project_ID"]]
    print(f"rows {len(F)}", flush=True)

    fl = pd.read_csv(f"{RAW_DIR}/change_log_fields.csv.gz")

    # ---------------- A. links at T ----------------
    lk = fl[fl["Field"] == "Link"].reset_index(drop=True)
    lk["t"] = secs(lk["Creation_Date"])
    lk["add"] = lk["To_String"].notna()
    txt = lk["To_String"].where(lk["add"], lk["From_String"]).fillna("")
    body = txt.str.replace(r"^This issue\s+", "", regex=True)
    lk["key"] = body.str.extract(r"(\S+)\s*$")[0]
    lk["phrase"] = body.str.replace(r"\s+\S+\s*$", "", regex=True).str.strip().str.lower()
    lk = lk[lk["key"].notna() & (lk["t"] > 0)]
    lk["cls"] = np.select([lk["phrase"].isin(BLOCKED_BY), lk["phrase"].isin(BLOCKS), lk["phrase"].isin(DUP)],
                          ["blocked_by", "blocks", "dup"], "other")
    # key -> issue ID: (issue, phrase) pairs with exactly one key in the log and exactly one Issue_Link row
    il = pd.read_csv(f"{RAW_DIR}/issue_link.csv")
    il = il[il["Issue_ID"].notna() & il["Target_Issue_ID"].notna()].copy()
    il["Issue_ID"] = il["Issue_ID"].astype(np.int64)
    il["phrase"] = il["Description"].str.strip().str.lower()
    a1 = lk.groupby(["Issue_ID", "phrase"])["key"].nunique().rename("nk").reset_index()
    a1 = a1.merge(lk[["Issue_ID", "phrase", "key"]].drop_duplicates(), on=["Issue_ID", "phrase"])
    a2 = il.groupby(["Issue_ID", "phrase"])["Target_Issue_ID"].agg(["nunique", "first"]).reset_index()
    m = a1[a1.nk == 1].merge(a2[a2["nunique"] == 1], on=["Issue_ID", "phrase"])
    km = m.groupby("key")["first"].agg(["nunique", "first"])
    keymap = km.loc[km["nunique"] == 1, "first"].astype(np.int64)
    lk["tgt"] = lk["key"].map(keymap)
    print(f"  A: {len(lk)} link events, {len(keymap)} keys mapped, events with mapped target {lk['tgt'].notna().mean():.3f}", flush=True)

    P = pair_events(lk[["Issue_ID", "key", "cls", "tgt", "t", "add"]], ["Issue_ID", "key", "cls"])
    P = P[P["Issue_ID"].isin(set(iid))]
    J = rows_x_pairs(R[["r", "Issue_ID", "T"]], P, "Issue_ID")
    act = (J["a"] < J["T"]) & (J["d"] >= J["T"])
    Ja = J[act]
    for name, c in [("n_links_at_T", None), ("n_blockedby_at_T", "blocked_by"), ("n_blocks_at_T", "blocks"), ("n_dup_at_T", "dup")]:
        sub = Ja if c is None else Ja[Ja["cls"] == c]
        F[name] = np.bincount(sub["r"], minlength=len(F))
    Jb = Ja[(Ja["cls"] == "blocked_by") & Ja["tgt"].notna()].copy()
    Jb["tr"] = Jb["tgt"].map(rmap).fillna(FAR).astype(np.int64)
    F["n_open_blockers_at_T"] = np.bincount(Jb.loc[Jb["tr"] >= Jb["T"], "r"], minlength=len(F))
    F["n_blockers_resolved_7d"] = np.bincount(Jb.loc[(Jb["tr"] < Jb["T"]) & (Jb["tr"] >= Jb["T"] - 7 * DAY), "r"], minlength=len(F))
    F["any_open_blocker"] = (F["n_open_blockers_at_T"] > 0).astype(int)
    # last link change before T (adds or removals)
    ev_t = np.concatenate([J.loc[J["a"] < J["T"], ["r", "a"]].to_numpy(),
                           J.loc[J["d"] < J["T"], ["r", "d"]].to_numpy()]) if len(J) else np.zeros((0, 2), np.int64)
    last = np.full(len(F), -1, np.int64)
    if len(ev_t):
        np.maximum.at(last, ev_t[:, 0].astype(np.int64), ev_t[:, 1].astype(np.int64))
    F["log_min_since_link_change"] = np.where(last > 0, np.log1p(np.maximum(T - last, 0) / 60), np.nan)
    assert (Ja["a"] < Ja["T"]).all()
    print(f"  A: rows with a link at T {np.mean(F.n_links_at_T > 0):.3f}, blocked-by {np.mean(F.n_blockedby_at_T > 0):.4f}, "
          f"open blocker {F.any_open_blocker.mean():.4f}", flush=True)
    del J, Ja, Jb, P

    # ---------------- B. fix version at T ----------------
    fv = fl[fl["Field"] == "Fix Version"].reset_index(drop=True)
    fv["t"] = secs(fv["Creation_Date"])
    fv["add"] = fv["To_String"].notna()
    fv["ver"] = fv["To_String"].where(fv["add"], fv["From_String"])
    fv = fv[fv["ver"].notna() & (fv["t"] > 0)]
    fv = fv.merge(iss[["ID", "Project_ID"]], left_on="Issue_ID", right_on="ID")
    fv["vid"] = fv.groupby(["Project_ID", "ver"]).ngroup().astype(np.int64)
    VP = pair_events(fv[["Issue_ID", "vid", "ver", "t", "add"]], ["Issue_ID", "vid"])
    VP = VP.drop(columns=["k"])
    # this issue's versions active at T -> the most recently added one
    J = rows_x_pairs(R[["r", "Issue_ID", "T"]], VP[VP["Issue_ID"].isin(set(iid))], "Issue_ID")
    J = J[(J["a"] < J["T"]) & (J["d"] >= J["T"])]
    F["n_fixver_at_T"] = np.bincount(J["r"], minlength=len(F))
    F["has_fixver_at_T"] = (F["n_fixver_at_T"] > 0).astype(int)
    cur = J.sort_values(["r", "a"]).groupby("r").tail(1).set_index("r")
    vid_row = np.full(len(F), -1, np.int64)
    vid_row[cur.index.to_numpy()] = cur["vid"].to_numpy()
    vname = pd.Series(index=cur.index, data=cur["ver"].astype(str).to_numpy())
    ph = np.zeros(len(F), int)
    ph[vname.index.to_numpy()] = (~vname.str.contains(r"\d", regex=True)).astype(int).to_numpy()
    F["fixver_placeholder"] = ph
    # version-level event arrays (all issues): assignment time a; "done" time = max(a, resolution)
    VP["rs"] = VP["Issue_ID"].map(rmap).fillna(FAR).astype(np.int64)
    first_a = VP.groupby(["vid", "Issue_ID"])["a"].min().reset_index()
    first_a["rs"] = first_a["Issue_ID"].map(rmap).fillna(FAR).astype(np.int64)
    first_a["done"] = np.maximum(first_a["a"], first_a["rs"])
    ka = np.sort(first_a["vid"].to_numpy() * SCALE + first_a["a"].to_numpy())
    kd = np.sort(first_a["vid"].to_numpy() * SCALE + np.minimum(first_a["done"].to_numpy(), FAR))
    vfirst = first_a.groupby("vid")["a"].min()
    ok = vid_row >= 0
    v = np.where(ok, vid_row, 0)
    q, b = v * SCALE + T, v * SCALE
    n_ass = np.searchsorted(ka, q, "left") - np.searchsorted(ka, b, "left")
    n_done = np.searchsorted(kd, q, "left") - np.searchsorted(kd, b, "left")
    n_d14 = np.searchsorted(kd, q, "left") - np.searchsorted(kd, np.maximum(b, q - 14 * DAY), "left")
    n_d30 = np.searchsorted(kd, q, "left") - np.searchsorted(kd, np.maximum(b, q - 30 * DAY), "left")
    age = (T - pd.Series(v).map(vfirst).fillna(0).to_numpy(np.int64)) / DAY
    open_ = np.maximum(n_ass - n_done, 0)
    eta = open_ / (n_d30 / 30.0 + 0.1)
    F["log_fixver_assigned"] = np.where(ok, np.log1p(n_ass), np.nan)
    F["fixver_done_share"] = np.where(ok, n_done / np.maximum(n_ass, 1), np.nan)
    F["log_fixver_resolved_14d"] = np.where(ok, np.log1p(n_d14), np.nan)
    F["log_fixver_age_days"] = np.where(ok, np.log1p(np.maximum(age, 0)), np.nan)
    F["log_fixver_days_left_est"] = np.where(ok, np.log1p(eta), np.nan)
    print(f"  B: rows with a fix version at T {F.has_fixver_at_T.mean():.3f}, placeholder {F.fixver_placeholder.mean():.3f}", flush=True)
    del J, VP, first_a, fl

    # ---------------- C. assignee at T ----------------
    pe = pd.read_csv(f"{RAW_DIR}/change_log_people.csv")
    pe = pe[pe["Field"].str.lower().eq("assignee")].reset_index(drop=True)
    pe["t"] = secs(pe["Creation_Date"])
    pe = pe[pe["t"] > 0].sort_values(["Issue_ID", "t"]).reset_index(drop=True)
    keys = pe["Issue_ID"].to_numpy(np.int64) * SCALE + pe["t"].to_numpy()
    hi = np.searchsorted(keys, iid * SCALE + T, "left")
    lo = np.searchsorted(keys, iid * SCALE, "left")
    nxt = np.searchsorted(keys, iid * SCALE + SCALE - 1, "left")
    to_s = pd.to_numeric(pe["To_String"], errors="coerce").to_numpy()
    fr_s = pd.to_numeric(pe["From_String"], errors="coerce").to_numpy()
    has_before = hi > lo
    has_any = nxt > lo
    final_a = ck["Assignee_ID"].to_numpy(float)
    a_T = np.where(has_before, to_s[np.clip(hi - 1, 0, len(pe) - 1)],
                   np.where(has_any, fr_s[np.clip(lo, 0, len(pe) - 1)], final_a))
    F["is_unassigned_T"] = np.isnan(a_T).astype(int)
    print(f"  C: assignee at T differs from final assignee on {np.mean((a_T != final_a) & ~(np.isnan(a_T) & np.isnan(final_a))):.3f} of rows; "
          f"unassigned at T {np.isnan(a_T).mean():.3f} (final {np.isnan(final_a).mean():.3f})", flush=True)

    lab = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2"])
    res = iss.merge(lab, on="ID")
    res = res[res["Assignee_ID"].notna()]
    e = res["Assignee_ID"].astype(np.int64).to_numpy()
    rs = res["r_s"].to_numpy(np.int64)
    d = res["delayed_v2"].to_numpy()
    o = np.lexsort((rs, e))
    k = e[o] * SCALE + rs[o]
    rs_sorted = rs[o]
    cumd = np.concatenate([[0], np.cumsum(d[o])])
    okA = ~np.isnan(a_T)
    ee = np.where(okA, np.nan_to_num(a_T), 0).astype(np.int64)
    qq, bb = ee * SCALE + T, ee * SCALE

    def win(days=None):
        lo_ = np.searchsorted(k, bb if days is None else np.maximum(bb, qq - days * DAY), "left")
        hi_ = np.searchsorted(k, qq, "left")
        return hi_ - lo_, cumd[hi_] - cumd[lo_], hi_, lo_

    n_all, s_all, hi_, lo_all = win()
    F["assignee_rate_known_T"] = np.where(okA, (s_all + 5.0) / (n_all + 10.0), np.nan)
    F["assignee_n_known_T"] = np.where(okA, np.log1p(n_all), 0)
    n90, s90, _, _ = win(90)
    F["a_rate_90d"] = np.where(okA, (s90 + 2.5) / (n90 + 5.0), np.nan)
    F["a_n_res_90d"] = np.where(okA, np.log1p(n90), 0)
    n30, _, _, _ = win(30)
    F["a_n_res_30d"] = np.where(okA, np.log1p(n30), 0)
    lo5 = np.maximum(hi_ - 5, lo_all)
    n5 = hi_ - lo5
    F["a_last5_late"] = np.where(okA & (n5 > 0), (cumd[hi_] - cumd[lo5]) / np.maximum(n5, 1), np.nan)
    last_res = np.where(n_all > 0, rs_sorted[np.clip(hi_ - 1, 0, len(rs_sorted) - 1)], -1)
    F["log_a_days_since_res"] = np.where(okA & (n_all > 0), np.log1p(np.maximum(T - last_res, 0) / DAY), np.nan)
    assert (last_res[n_all > 0] < T[n_all > 0]).all()
    print("  C: done", flush=True)

    F.to_csv(OUT, index=False)
    print(f"saved {F.shape} -> {OUT}", flush=True)

    chk = F.merge(lab, left_on="Issue_ID", right_on="ID")
    cols = [c for c in F.columns if c not in ("Issue_ID", "checkpoint_index")]
    corr = chk[cols].corrwith(chk["delayed_v2"]).abs().sort_values(ascending=False)
    print("\n|corr| with delayed_v2:\n" + corr.round(3).to_string())
    assert corr.max() < 0.95
    print("Leakage test: PASS")


if __name__ == "__main__":
    main()
