"""
Task J4: change-history + "known-at-T" features for the dynamic model (-> v5).

Every feature for a checkpoint row uses only information timestamped strictly
before the checkpoint moment T = Creation_Date + elapsed_minutes.

A. Reassignment (Change_Log PEOPLE / assignee)
   n_assignee_changes, ever_reassigned, assigned_at_T,
   log_min_since_assign_change, log_min_to_first_assign (censored at T)
B. Sprint / Scrum (Change_Log OTHER / Sprint + Sprint table)
   in_sprint_at_T, n_sprint_changes, n_sprint_carryovers,
   days_to_sprint_end (negative = sprint already over), sprint_overdue,
   sprint_len_days
C. Story points over time (Change_Log STORY_POINT)
   n_sp_changes, sp_at_T  (value as of T -- the base feature Story_Point is the FINAL value)
D. Other field churn before T (Change_Log OTHER)
   n_priority_chg, n_type_chg, n_rank_chg, n_link_chg, n_fixver_chg,
   n_epic_chg, n_estimate_chg, n_label_chg, n_component_chg,
   n_field_chg_total, n_field_chg_last_7d
E. Known-at-T delay rates (outcomes of OTHER issues that had resolved before T,
   smoothed toward 0.5 with m=10 pseudo-issues)
   proj_rate_known, proj_rate_known_90d, proj_n_resolved_90d,
   assignee_rate_known, assignee_n_known, reporter_rate_known
   (fixes a subtle issue in the Phase 6 assignee_prior_delay_rate, which uses
    prior issues by CREATION order even if their outcome was not yet known)
F. Position relative to the label threshold
   log_elapsed_over_median  = log(elapsed / project median used by the label)
   past_median              = 1 if the issue is already older than that median
                              (then it is delayed by definition -- known at T,
                              not leakage, but it makes some rows trivially
                              positive; evaluation also reports AUC on the
                              undetermined rows only)
"""

import numpy as np
import pandas as pd

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"
OUT = f"{FEATURES_DIR}/phase8_change_history_features.csv"
SCALE = 10**10
FAR = np.int64(4 * 10**9)


def secs(s):
    s = pd.Series(s).reset_index(drop=True).astype(str)
    short = s.str.match(r"^\d{2}-\d{2}-\d{2} ")
    s[short] = "20" + s[short]
    dt = pd.to_datetime(s.where(s != "nan"), format="ISO8601", errors="coerce")
    out = ((dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
    return out.fillna(-1).to_numpy(np.int64)


def index_events(issue_ids, times):
    order = np.lexsort((times, issue_ids))
    keys = issue_ids[order].astype(np.int64) * SCALE + times[order]
    return order, keys


def before(keys, q, base):
    hi = np.searchsorted(keys, q, side="left")
    lo = np.searchsorted(keys, base, side="left")
    return hi - lo, hi - 1, lo


def main():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv",
                     usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    iss = pd.read_csv(f"{RAW_DIR}/issue_slim.csv.gz")
    iss["c_s"] = secs(iss["Creation_Date"])
    iss["r_s"] = np.where(iss["Resolution_Date"].notna(), secs(iss["Resolution_Date"]), FAR)
    ck = ck.merge(iss[["ID", "Project_ID", "Assignee_ID", "Reporter_ID", "c_s"]],
                  left_on="Issue_ID", right_on="ID", how="left")
    T = (ck["c_s"].to_numpy(np.int64) + np.floor(ck["elapsed_minutes"].to_numpy() * 60).astype(np.int64))
    iid = ck["Issue_ID"].to_numpy(np.int64)
    q, base = iid * SCALE + T, iid * SCALE
    c_s = ck["c_s"].to_numpy(np.int64)
    ids = set(iid)
    F = pd.DataFrame({"Issue_ID": iid, "checkpoint_index": ck["checkpoint_index"].to_numpy()})
    print(f"rows: {len(F)}", flush=True)

    # ---------------- A. reassignment ----------------
    pe = pd.read_csv(f"{RAW_DIR}/change_log_people.csv")
    pe = pe[pe["Field"].str.lower().eq("assignee") & pe["Issue_ID"].isin(ids)].reset_index(drop=True)
    pe["t"] = secs(pe["Creation_Date"])
    o, keys = index_events(pe["Issue_ID"].to_numpy(np.int64), pe["t"].to_numpy())
    pe = pe.iloc[o].reset_index(drop=True)
    n, last, lo = before(keys, q, base)
    F["n_assignee_changes"] = n
    reas = np.concatenate([[0], np.cumsum(pe["From_String"].notna().to_numpy().astype(int))])
    F["ever_reassigned"] = ((reas[last + 1] - reas[lo]) > 0).astype(int)
    lastto = pe["To_String"].notna().to_numpy()
    F["assigned_at_T"] = np.where(n > 0, lastto[np.clip(last, 0, None)], 0).astype(int)
    lt = np.where(n > 0, pe["t"].to_numpy()[np.clip(last, 0, None)], c_s)
    F["log_min_since_assign_change"] = np.log1p(np.maximum(T - lt, 0) / 60)
    first_t = np.where(n > 0, pe["t"].to_numpy()[np.clip(lo, 0, len(pe) - 1)], T)
    F["log_min_to_first_assign"] = np.log1p(np.maximum(first_t - c_s, 0) / 60)
    assert (lt[n > 0] < T[n > 0]).all()
    print(f"  A: reassignment rows with >=1 change {np.mean(n > 0):.3f}", flush=True)

    # ---------------- B. sprint ----------------
    sc = pd.read_csv(f"{RAW_DIR}/change_log_sprint.csv")
    sc = sc[sc["Issue_ID"].isin(ids)].reset_index(drop=True)
    sc["t"] = secs(sc["Creation_Date"])
    o, keys = index_events(sc["Issue_ID"].to_numpy(np.int64), sc["t"].to_numpy())
    sc = sc.iloc[o].reset_index(drop=True)
    n, last, lo = before(keys, q, base)
    F["n_sprint_changes"] = n
    carry = (sc["From_String"].notna() & sc["To_String"].notna()).astype(int).to_numpy()
    cum = np.concatenate([[0], np.cumsum(carry)])
    F["n_sprint_carryovers"] = cum[last + 1] - cum[lo]
    cur_name = np.where(n > 0, sc["To_String"].to_numpy(dtype=object)[np.clip(last, 0, None)], None)
    cur_name = pd.Series(cur_name).where(pd.notna(cur_name), None)
    cur_name = cur_name.map(lambda s: s.split(",")[-1].strip() if isinstance(s, str) else None)
    F["in_sprint_at_T"] = cur_name.notna().astype(int).to_numpy()
    spr = pd.read_csv(f"{RAW_DIR}/sprint.csv")
    spr["s_s"], spr["e_s"] = secs(spr["Start_Date"]), secs(spr["End_Date"])
    spr = spr.drop_duplicates(["Project_ID", "Name"])
    look = spr.set_index(["Project_ID", "Name"])[["s_s", "e_s"]]
    key_df = pd.DataFrame({"Project_ID": ck["Project_ID"].to_numpy(), "Name": cur_name.to_numpy()})
    j = key_df.merge(look, left_on=["Project_ID", "Name"], right_index=True, how="left")
    s_s, e_s = j["s_s"].to_numpy(float), j["e_s"].to_numpy(float)
    valid = (s_s > 0) & (e_s > 0)
    F["days_to_sprint_end"] = np.where(valid, (e_s - T) / 86400, np.nan)
    F["sprint_overdue"] = np.where(valid, (T > e_s).astype(int), 0)
    F["sprint_len_days"] = np.where(valid, (e_s - s_s) / 86400, np.nan)
    F["sprint_started"] = np.where(valid, (T >= s_s).astype(int), 0)
    print(f"  B: in sprint at T {F['in_sprint_at_T'].mean():.3f}; matched to Sprint table {valid.mean():.3f}", flush=True)

    # ---------------- C. story points ----------------
    sp = pd.read_csv(f"{RAW_DIR}/change_log_other.csv")
    sp = sp[(sp["Change_Type"] == "STORY_POINT") & sp["Issue_ID"].isin(ids)].reset_index(drop=True)
    sp["t"] = secs(sp["Creation_Date"])
    o, keys = index_events(sp["Issue_ID"].to_numpy(np.int64), sp["t"].to_numpy())
    sp = sp.iloc[o].reset_index(drop=True)
    n, last, lo = before(keys, q, base)
    F["n_sp_changes"] = n
    val = pd.to_numeric(sp["To_String"], errors="coerce").to_numpy()
    F["sp_at_T"] = np.where(n > 0, val[np.clip(last, 0, None)], np.nan)
    print(f"  C: story point history rows {np.mean(n > 0):.3f}", flush=True)

    # ---------------- D. other field churn ----------------
    fl = pd.read_csv(f"{RAW_DIR}/change_log_fields.csv.gz", usecols=["Issue_ID", "Field", "Creation_Date"])
    fl = fl[fl["Issue_ID"].isin(ids)].reset_index(drop=True)
    fl["Field"] = fl["Field"].str.lower()
    fl["t"] = secs(fl["Creation_Date"])
    groups = {"n_priority_chg": ["priority"], "n_type_chg": ["issuetype"], "n_rank_chg": ["rank"],
              "n_link_chg": ["link"], "n_fixver_chg": ["fix version"], "n_epic_chg": ["epic link"],
              "n_estimate_chg": ["timeestimate", "timeoriginalestimate"], "n_label_chg": ["labels"],
              "n_component_chg": ["component"]}
    o, keys = index_events(fl["Issue_ID"].to_numpy(np.int64), fl["t"].to_numpy())
    fl = fl.iloc[o].reset_index(drop=True)
    n, last, lo = before(keys, q, base)
    F["n_field_chg_total"] = n
    n7, _, _ = before(keys, q, np.maximum(base, q - 7 * 86400))
    F["n_field_chg_last_7d"] = n7
    for name, fields in groups.items():
        cum = np.concatenate([[0], np.cumsum(fl["Field"].isin(fields).to_numpy().astype(int))])
        F[name] = cum[last + 1] - cum[lo]
    print("  D: field churn done", flush=True)

    # ---------------- E. known-at-T delay rates ----------------
    lab = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2"])
    res = iss.merge(lab, on="ID")  # resolved, labelled population
    m, p0 = 10.0, 0.5

    def known_rate(entity_col_issue, entity_ck, window_days=None):
        r = res[res[entity_col_issue].notna()]
        e = r[entity_col_issue].astype(np.int64).to_numpy()
        rs = r["r_s"].to_numpy(np.int64)
        d = r["delayed_v2"].to_numpy()
        order = np.lexsort((rs, e))
        k = e[order] * SCALE + rs[order]
        cumd = np.concatenate([[0], np.cumsum(d[order])])
        ce = ck[entity_ck]
        ok = ce.notna().to_numpy()
        ee = np.where(ok, ce.fillna(0).astype(np.int64).to_numpy(), 0)
        qq, bb = ee * SCALE + T, ee * SCALE
        if window_days:
            bb = np.maximum(bb, qq - window_days * 86400)
        hi = np.searchsorted(k, qq, side="left")
        lo_ = np.searchsorted(k, bb, side="left")
        cnt, s = hi - lo_, cumd[hi] - cumd[lo_]
        rate = (s + m * p0) / (cnt + m)
        return np.where(ok, rate, np.nan), np.where(ok, cnt, 0)

    F["proj_rate_known"], _ = known_rate("Project_ID", "Project_ID")
    F["proj_rate_known_90d"], F["proj_n_resolved_90d"] = known_rate("Project_ID", "Project_ID", 90)
    F["assignee_rate_known"], F["assignee_n_known"] = known_rate("Assignee_ID", "Assignee_ID")
    F["reporter_rate_known"], _ = known_rate("Reporter_ID", "Reporter_ID")
    F["proj_n_resolved_90d"] = np.log1p(F["proj_n_resolved_90d"])
    F["assignee_n_known"] = np.log1p(F["assignee_n_known"])
    print("  E: known-at-T rates done", flush=True)

    # ---------------- F. relative to label threshold ----------------
    sp_ = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    rr = iss.merge(sp_[["ID", "split"]], on="ID")
    rr = rr.merge(pd.read_csv(f"{RAW_DIR}/issue_slim.csv.gz", usecols=["ID"]), on="ID")
    rr["res_min"] = (rr["r_s"] - rr["c_s"]) / 60
    med = rr[rr["split"].isin(["train", "val"])].groupby("Project_ID")["res_min"].median()
    pm = ck["Project_ID"].map(med).to_numpy()
    F["log_elapsed_over_median"] = np.log(np.maximum(ck["elapsed_minutes"].to_numpy(), 1e-3) / pm)
    F["past_median"] = (ck["elapsed_minutes"].to_numpy() > pm).astype(int)
    print(f"  F: rows already past project median: {F['past_median'].mean():.3f}", flush=True)

    F.to_csv(OUT, index=False)
    print(f"saved {F.shape} -> {OUT}")

    # ---------------- leakage test ----------------
    chk = F.merge(lab, left_on="Issue_ID", right_on="ID")
    cols = [c for c in F.columns if c not in ("Issue_ID", "checkpoint_index")]
    corr = chk[cols].corrwith(chk["delayed_v2"]).abs().sort_values(ascending=False)
    print("\n|corr| with delayed_v2:\n" + corr.round(3).to_string())
    print("\ndelay rate when past_median=1:", chk.loc[chk.past_median == 1, "delayed_v2"].mean().round(4),
          " (must be 1.0 by construction)")
    assert corr.drop("past_median").max() < 0.95
    print("Leakage test: PASS")


if __name__ == "__main__":
    main()
