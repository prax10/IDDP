"""
Task O1: v8 features. Everything strictly before the checkpoint moment T (or before the issue's creation).

A. Priority / type AT T. The base features Priority_Normalized and Type_Normalized are the FINAL values;
   ~95k issues changed priority and ~81k changed type. Rebuilt from the change log like the assignee:
   last To_String before T, else the first event's From_String, else the final value.
     priority_T, type_T, n_priority_up (raises before T)
B. Similar past issues (text kNN). For each issue, the 20 most similar issues (SBERT cosine) in the same
   project that were RESOLVED BEFORE THIS ISSUE WAS CREATED, so their outcomes were known.
     knn_late_rate (similarity-weighted), knn_log_ratio (mean log2 duration / project median),
     knn_top_sim, knn_n
C. Epic AT T (Epic Link change log): progress of the epic the issue belongs to at T, as for fix versions.
     has_epic_T, log_epic_children, epic_done_share, log_epic_resolved_14d, log_epic_age_days, log_epic_days_left_est
D. Component AT T: smoothed late rate of the current component over issues resolved before T.
     has_component_T, comp_rate_known, log_comp_n_known
E. Reporter recent form: rep_rate_90d, log_rep_n_res_90d
"""
import numpy as np
import pandas as pd
from task_n1_v7_features import secs, pair_events, SCALE, FAR, DAY

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"
OUT = f"{FEATURES_DIR}/phase8_v8_features.csv"
PRIORITY_MAP = {"Trivial": 1, "Trivial - P5": 1, "Lowest": 1, "Minor": 2, "Minor - P4": 2, "Low": 2,
                "Major": 3, "Major - P3": 3, "Medium": 3, "Critical": 4, "Critical - P2": 4, "High": 4,
                "Blocker": 5, "Blocker - P1": 5, "Highest": 5}
K = 20


def value_at_T(ev, iid, T, final):
    """ev sorted by (Issue_ID, t) with From_String/To_String. Value of the field at T."""
    keys = ev["Issue_ID"].to_numpy(np.int64) * SCALE + ev["t"].to_numpy()
    hi = np.searchsorted(keys, iid * SCALE + T, "left")
    lo = np.searchsorted(keys, iid * SCALE, "left")
    nx = np.searchsorted(keys, iid * SCALE + SCALE - 1, "left")
    to_s, fr_s = ev["To_String"].to_numpy(object), ev["From_String"].to_numpy(object)
    n = len(ev) - 1
    return np.where(hi > lo, to_s[np.clip(hi - 1, 0, n)], np.where(nx > lo, fr_s[np.clip(lo, 0, n)], final)), hi, lo


def group_progress(ev_pairs, rmap, vid_row, T, prefix, F):
    """Progress stats of a group (fix version / epic) at T. ev_pairs: vid, Issue_ID, a."""
    fa = ev_pairs.groupby(["vid", "Issue_ID"])["a"].min().reset_index()
    fa["rs"] = fa["Issue_ID"].map(rmap).fillna(FAR).astype(np.int64)
    fa["done"] = np.minimum(np.maximum(fa["a"], fa["rs"]), FAR)
    ka = np.sort(fa["vid"].to_numpy() * SCALE + fa["a"].to_numpy())
    kd = np.sort(fa["vid"].to_numpy() * SCALE + fa["done"].to_numpy())
    vfirst = fa.groupby("vid")["a"].min()
    ok = vid_row >= 0
    v = np.where(ok, vid_row, 0)
    q, b = v * SCALE + T, v * SCALE
    cnt = lambda k, lo: np.searchsorted(k, q, "left") - np.searchsorted(k, lo, "left")
    n_ass, n_done = cnt(ka, b), cnt(kd, b)
    n14, n30 = cnt(kd, np.maximum(b, q - 14 * DAY)), cnt(kd, np.maximum(b, q - 30 * DAY))
    age = (T - pd.Series(v).map(vfirst).fillna(0).to_numpy(np.int64)) / DAY
    open_ = np.maximum(n_ass - n_done, 0)
    F[f"has_{prefix}_T"] = ok.astype(int)
    F[f"log_{prefix}_children"] = np.where(ok, np.log1p(n_ass), np.nan)
    F[f"{prefix}_done_share"] = np.where(ok, n_done / np.maximum(n_ass, 1), np.nan)
    F[f"log_{prefix}_resolved_14d"] = np.where(ok, np.log1p(n14), np.nan)
    F[f"log_{prefix}_age_days"] = np.where(ok, np.log1p(np.maximum(age, 0)), np.nan)
    F[f"log_{prefix}_days_left_est"] = np.where(ok, np.log1p(open_ / (n30 / 30.0 + 0.1)), np.nan)


def current_group(ev, R, iid_set):
    """Most recently added value still active at T, per checkpoint row -> vid (or -1)."""
    P = pair_events(ev[["Issue_ID", "vid", "t", "add"]], ["Issue_ID", "vid"]).drop(columns=["k"])
    J = R[["r", "Issue_ID", "T"]].merge(P[P["Issue_ID"].isin(iid_set)], on="Issue_ID")
    J = J[(J["a"] < J["T"]) & (J["d"] >= J["T"])]
    cur = J.sort_values(["r", "a"]).groupby("r").tail(1)
    out = np.full(len(R), -1, np.int64)
    out[cur["r"].to_numpy()] = cur["vid"].to_numpy()
    return out, P


def main():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    iss = pd.read_csv(f"{RAW_DIR}/issue_slim.csv.gz")
    iss["c_s"] = secs(iss["Creation_Date"])
    iss["r_s"] = np.where(iss["Resolution_Date"].notna(), secs(iss["Resolution_Date"]), FAR)
    rmap = iss.set_index("ID")["r_s"]
    ck = ck.merge(iss[["ID", "Project_ID", "Priority", "Type", "Reporter_ID", "c_s"]], left_on="Issue_ID", right_on="ID", how="left")
    T = ck["c_s"].to_numpy(np.int64) + np.floor(ck["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    ck["T"], ck["r"] = T, np.arange(len(ck))
    iid = ck["Issue_ID"].to_numpy(np.int64)
    R = ck[["r", "Issue_ID", "T"]]
    F = pd.DataFrame({"Issue_ID": iid, "checkpoint_index": ck["checkpoint_index"].to_numpy()})
    lab = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2", "Type_Normalized"])
    fl = pd.read_csv(f"{RAW_DIR}/change_log_fields.csv.gz")
    fl["t"] = secs(fl["Creation_Date"])
    fl = fl[fl["t"] > 0]
    print(f"rows {len(F)}", flush=True)

    # ---------- A. priority / type at T ----------
    pr = fl[fl["Field"].str.lower() == "priority"].sort_values(["Issue_ID", "t"]).reset_index(drop=True)
    pT, hi, lo = value_at_T(pr, iid, T, ck["Priority"].to_numpy(object))
    F["priority_T"] = pd.Series(pT).map(PRIORITY_MAP).astype("float32").to_numpy()
    up = (pr["To_String"].map(PRIORITY_MAP) > pr["From_String"].map(PRIORITY_MAP)).astype(int).to_numpy()
    cum = np.concatenate([[0], np.cumsum(up)])
    F["n_priority_up"] = cum[hi] - cum[lo]
    fin = pd.Series(ck["Priority"].to_numpy(object)).map(PRIORITY_MAP).to_numpy(float)
    print(f"  A: priority at T differs from final on {np.mean((F['priority_T'].to_numpy() != fin) & ~(np.isnan(fin) & F['priority_T'].isna().to_numpy())):.3f} of rows", flush=True)
    ty = fl[fl["Field"] == "issuetype"].sort_values(["Issue_ID", "t"]).reset_index(drop=True)
    tT, _, _ = value_at_T(ty, iid, T, ck["Type"].to_numpy(object))
    keep = set(lab["Type_Normalized"].dropna().unique())
    F["type_T"] = pd.Series(tT).where(pd.Series(tT).isin(keep), "Other").to_numpy()
    print(f"  A: type at T differs from final on {np.mean(pd.Series(tT).to_numpy() != ck['Type'].to_numpy(object)):.3f} of rows", flush=True)

    # ---------- B. similar past issues ----------
    ids = np.load(f"{FEATURES_DIR}/sbert_raw_ids.npy")
    E = np.load(f"{FEATURES_DIR}/sbert_raw.npy").astype(np.float32)
    meta = pd.DataFrame({"ID": ids}).merge(iss[["ID", "Project_ID", "c_s", "r_s"]], on="ID", how="left").merge(lab[["ID", "delayed_v2"]], on="ID")
    sp = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv", usecols=["ID", "split"])
    meta = meta.merge(sp, on="ID", how="left")
    meta["dur"] = (meta["r_s"] - meta["c_s"]) / 60.0
    med = meta[meta.split.isin(["train", "val"])].groupby("Project_ID")["dur"].median()
    meta["lr"] = np.log2(np.maximum(meta["dur"], 1) / meta["Project_ID"].map(med))
    assert (meta["ID"].to_numpy() == ids).all()
    kn = {c: np.full(len(ids), np.nan, np.float32) for c in ["knn_late_rate", "knn_log_ratio", "knn_top_sim", "knn_n"]}
    for pid, g in meta.groupby("Project_ID"):
        ix = g.index.to_numpy()
        Ep = E[ix]
        cs, rs = g["c_s"].to_numpy(), g["r_s"].to_numpy()
        y, lr = g["delayed_v2"].to_numpy(np.float32), g["lr"].to_numpy(np.float32)
        for s in range(0, len(ix), 2000):
            S = Ep[s:s + 2000] @ Ep.T
            S[rs[None, :] >= cs[s:s + 2000, None]] = -np.inf          # only issues resolved before creation
            kk = min(K, S.shape[1])
            top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            ts = np.take_along_axis(S, top, 1)
            valid = np.isfinite(ts)
            w = np.where(valid, np.maximum(ts, 0) + 1e-3, 0)
            n = valid.sum(1)
            q = ix[s:s + 2000]
            kn["knn_n"][q] = n
            with np.errstate(invalid="ignore", divide="ignore"):
                kn["knn_late_rate"][q] = np.where(n > 0, (w * y[top]).sum(1) / w.sum(1), np.nan)
                kn["knn_log_ratio"][q] = np.where(n > 0, (w * lr[top]).sum(1) / w.sum(1), np.nan)
                kn["knn_top_sim"][q] = np.where(n > 0, np.where(valid, ts, -1).max(1), np.nan)
        print(f"    project {pid}: {len(ix)} issues", flush=True)
    knn = pd.DataFrame({"Issue_ID": ids, **kn})
    knn["knn_n"] = np.log1p(knn["knn_n"])
    F = F.merge(knn, on="Issue_ID", how="left")
    print(f"  B: rows with >=1 earlier similar issue {F['knn_late_rate'].notna().mean():.3f}", flush=True)

    # ---------- C. epic at T ----------
    ep = fl[fl["Field"] == "Epic Link"].reset_index(drop=True)
    ep["add"] = ep["To_String"].notna()
    ep["val"] = ep["To_String"].where(ep["add"], ep["From_String"])
    ep = ep[ep["val"].notna()]
    ep["vid"] = ep.groupby("val").ngroup().astype(np.int64)
    vid_row, P = current_group(ep, R, set(iid))
    group_progress(P[["vid", "Issue_ID", "a"]], rmap, vid_row, T, "epic", F)
    print(f"  C: rows in an epic at T {F['has_epic_T'].mean():.3f}", flush=True)

    # ---------- D. component at T ----------
    co = fl[fl["Field"] == "Component"].reset_index(drop=True)
    co["add"] = co["To_String"].notna()
    co["val"] = co["To_String"].where(co["add"], co["From_String"])
    co = co[co["val"].notna()].merge(iss[["ID", "Project_ID"]], left_on="Issue_ID", right_on="ID")
    co["vid"] = co.groupby(["Project_ID", "val"]).ngroup().astype(np.int64)
    vid_row, P = current_group(co, R, set(iid))
    fa = P.groupby(["vid", "Issue_ID"])["a"].min().reset_index().merge(lab[["ID", "delayed_v2"]], left_on="Issue_ID", right_on="ID")
    fa["rs"] = fa["Issue_ID"].map(rmap).astype(np.int64)
    fa["done"] = np.maximum(fa["a"], fa["rs"])
    fa = fa.sort_values(["vid", "done"])
    kd = fa["vid"].to_numpy() * SCALE + fa["done"].to_numpy()
    cumd = np.concatenate([[0], np.cumsum(fa["delayed_v2"].to_numpy())])
    ok = vid_row >= 0
    v = np.where(ok, vid_row, 0)
    hi_, lo_ = np.searchsorted(kd, v * SCALE + T, "left"), np.searchsorted(kd, v * SCALE, "left")
    n, s_ = hi_ - lo_, cumd[hi_] - cumd[lo_]
    F["has_component_T"] = ok.astype(int)
    F["comp_rate_known"] = np.where(ok, (s_ + 5.0) / (n + 10.0), np.nan)
    F["log_comp_n_known"] = np.where(ok, np.log1p(n), np.nan)
    print(f"  D: rows with a component at T {ok.mean():.3f}", flush=True)

    # ---------- E. reporter recent form ----------
    res = iss.merge(lab[["ID", "delayed_v2"]], on="ID")
    res = res[res["Reporter_ID"].notna()]
    e = res["Reporter_ID"].astype(np.int64).to_numpy()
    rs = res["r_s"].to_numpy(np.int64)
    o = np.lexsort((rs, e))
    k = e[o] * SCALE + rs[o]
    cd = np.concatenate([[0], np.cumsum(res["delayed_v2"].to_numpy()[o])])
    okr = ck["Reporter_ID"].notna().to_numpy()
    ee = np.where(okr, ck["Reporter_ID"].fillna(0).astype(np.int64).to_numpy(), 0)
    qq = ee * SCALE + T
    hi_, lo_ = np.searchsorted(k, qq, "left"), np.searchsorted(k, np.maximum(ee * SCALE, qq - 90 * DAY), "left")
    F["rep_rate_90d"] = np.where(okr, (cd[hi_] - cd[lo_] + 2.5) / (hi_ - lo_ + 5.0), np.nan)
    F["log_rep_n_res_90d"] = np.where(okr, np.log1p(hi_ - lo_), 0)

    for c in F.columns:
        if F[c].dtype == "float64":
            F[c] = F[c].astype("float32")
    F.to_csv(OUT, index=False)
    print(f"saved {F.shape} -> {OUT}", flush=True)
    chk = F.merge(lab[["ID", "delayed_v2"]], left_on="Issue_ID", right_on="ID")
    cols = [c for c in F.columns if c not in ("Issue_ID", "checkpoint_index", "type_T")]
    corr = chk[cols].corrwith(chk["delayed_v2"]).abs().sort_values(ascending=False)
    print("\n|corr| with delayed_v2:\n" + corr.round(3).to_string())
    assert corr.max() < 0.95
    print("Leakage test: PASS")


if __name__ == "__main__":
    main()
