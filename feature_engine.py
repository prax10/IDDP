"""
feature_engine.py -- compute every dynamic feature for ANY (issue, moment T) pair.

Shared by: Scrum mode (task_j5), the v4/v5 app (app_v4.py), and anything that
needs features "as of now". Logic is identical to task_i1_activity_features.py
and task_j4_change_history_features.py (verified by test_feature_engine()).

Usage:
    fe = FeatureEngine()                       # loads raw caches once (~1 min)
    X = fe.at(issue_ids, T_seconds)            # DataFrame, one row per (issue, T)

All features use only events with timestamp strictly < T.
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
FEATURES_DIR = "data/features"
SCALE = 10**10
FAR = np.int64(4 * 10**9)
STAGE = {"todo": 0, "reopened": 0, "in_progress": 1, "waiting": 1, "review": 2, "testing": 3, "done": 4}

ACT_COLS = ["n_comments", "n_comment_authors", "comments_last_7d", "log_min_since_last_comment",
            "n_transitions", "n_reopens", "n_back_moves", "ever_in_progress", "ever_in_review",
            "status_bucket", "log_min_since_last_change", "log_min_since_last_activity",
            "project_open_at_T", "project_resolved_30d", "assignee_open_at_T",
            "log_title_len", "log_desc_len", "desc_has_code"]


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


def secs(s):
    s = pd.Series(s).reset_index(drop=True).astype(str)
    short = s.str.match(r"^\d{2}-\d{2}-\d{2} ")
    s[short] = "20" + s[short]
    dt = pd.to_datetime(s.where(~s.isin(["nan", "None", "NaT"])), format="ISO8601", errors="coerce")
    return ((dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).fillna(-1).to_numpy(np.int64)


def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)


class _Events:
    """Per-issue sorted event stream with fast 'how many before T' queries."""

    def __init__(self, issue_ids, times, cols=None):
        order = np.lexsort((times, issue_ids))
        self.iid = np.asarray(issue_ids)[order].astype(np.int64)
        self.t = np.asarray(times)[order].astype(np.int64)
        self.keys = self.iid * SCALE + self.t
        self.cols = {k: np.asarray(v)[order] for k, v in (cols or {}).items()}
        self.cum = {}

    def query(self, iid, T, window=None):
        q, base = iid * SCALE + T, iid * SCALE
        if window:
            base = np.maximum(base, q - window)
        hi = np.searchsorted(self.keys, q, side="left")
        lo = np.searchsorted(self.keys, base, side="left")
        return hi - lo, hi - 1, lo

    def csum(self, name, last, lo):
        if name not in self.cum:
            self.cum[name] = np.concatenate([[0], np.cumsum(self.cols[name].astype(np.int64))])
        c = self.cum[name]
        return c[last + 1] - c[lo]

    def last_val(self, name, n, last, default):
        v = self.cols[name] if name != "t" else self.t
        return np.where(n > 0, v[np.clip(last, 0, max(len(v) - 1, 0))] if len(v) else default, default)


class FeatureEngine:
    def __init__(self, verbose=True):
        log = print if verbose else (lambda *a, **k: None)
        iss = pd.read_csv(_first(f"{RAW_DIR}/issue_slim.csv.gz", "data/stage_tmp/issue_slim.csv.gz"))
        iss["c_s"] = secs(iss["Creation_Date"])
        iss["r_s"] = np.where(iss["Resolution_Date"].notna(), secs(iss["Resolution_Date"]), FAR)
        self.iss = iss.set_index("ID", drop=False)
        log("  issues loaded", flush=True)

        cm = pd.read_csv(f"{RAW_DIR}/comment.csv")
        cm["t"] = secs(cm["Creation_Date"])
        cm = cm.sort_values(["Issue_ID", "t"], kind="mergesort")
        first_auth = (~cm.duplicated(["Issue_ID", "Author_ID"])).astype(int).to_numpy()
        self.cm = _Events(cm["Issue_ID"].to_numpy(), cm["t"].to_numpy(), {"first_auth": first_auth})

        cl = pd.read_csv(_first(f"{RAW_DIR}/change_log_status.csv", f"{RAW_DIR}/change_log_status.csv.gz"))
        cl = cl[cl["From_String"].notna() & cl["To_String"].notna()].copy()
        cl["t"] = secs(cl["Creation_Date"])
        to_b, from_b = cl["To_String"].map(bucket), cl["From_String"].map(bucket)
        self.cl = _Events(cl["Issue_ID"].to_numpy(), cl["t"].to_numpy(), {
            "to_b": to_b.to_numpy(object), "from_b": from_b.to_numpy(object),
            "is_reopen": (to_b == "reopened").to_numpy(int),
            "is_back": (to_b.map(STAGE).fillna(-1) < from_b.map(STAGE).fillna(-1)).to_numpy(int),
            "is_prog": (to_b == "in_progress").to_numpy(int),
            "is_rev": to_b.isin(["review", "testing"]).to_numpy(int)})
        # status before the first genuine transition (known from creation)
        first = pd.DataFrame({"iid": self.cl.iid, "fb": self.cl.cols["from_b"]}).drop_duplicates("iid")
        self.first_from = first.set_index("iid")["fb"]
        log("  comments + status history loaded", flush=True)

        pe = pd.read_csv(f"{RAW_DIR}/change_log_people.csv")
        pe = pe[pe["Field"].str.lower().eq("assignee")]
        pe_t = secs(pe["Creation_Date"])
        self.pe = _Events(pe["Issue_ID"].to_numpy(), pe_t, {
            "reas": pe["From_String"].notna().to_numpy(int), "assigned": pe["To_String"].notna().to_numpy(int)})

        sc = pd.read_csv(f"{RAW_DIR}/change_log_sprint.csv")
        sc_t = secs(sc["Creation_Date"])
        names = sc["To_String"].map(lambda s: s.split(",")[-1].strip() if isinstance(s, str) else None)
        self.sc = _Events(sc["Issue_ID"].to_numpy(), sc_t, {
            "carry": (sc["From_String"].notna() & sc["To_String"].notna()).to_numpy(int),
            "name": names.to_numpy(object)})
        spr = pd.read_csv(f"{RAW_DIR}/sprint.csv")
        spr["s_s"], spr["e_s"], spr["done_s"] = secs(spr["Start_Date"]), secs(spr["End_Date"]), secs(spr["Complete_Date"])
        self.sprint_by_proj = spr.drop_duplicates(["Project_ID", "Name"]).set_index(["Project_ID", "Name"])
        uniq = spr[~spr.duplicated("Name", keep=False)]
        self.sprint_by_name = uniq.set_index("Name")
        self.spr = spr

        sp = pd.read_csv(f"{RAW_DIR}/change_log_other.csv")
        sp = sp[sp["Change_Type"] == "STORY_POINT"]
        self.sp = _Events(sp["Issue_ID"].to_numpy(), secs(sp["Creation_Date"]),
                          {"val": pd.to_numeric(sp["To_String"], errors="coerce").to_numpy(float)})

        fl = pd.read_csv(f"{RAW_DIR}/change_log_fields.csv.gz", usecols=["Issue_ID", "Field", "Creation_Date"])
        fl["Field"] = fl["Field"].str.lower()
        self.field_groups = {"n_priority_chg": ["priority"], "n_type_chg": ["issuetype"], "n_rank_chg": ["rank"],
                             "n_link_chg": ["link"], "n_fixver_chg": ["fix version"], "n_epic_chg": ["epic link"],
                             "n_estimate_chg": ["timeestimate", "timeoriginalestimate"],
                             "n_label_chg": ["labels"], "n_component_chg": ["component"]}
        self.fl = _Events(fl["Issue_ID"].to_numpy(), secs(fl["Creation_Date"]),
                          {k: fl["Field"].isin(v).to_numpy(int) for k, v in self.field_groups.items()})
        log("  change logs loaded", flush=True)

        lab = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2"])
        self.res = iss.merge(lab, on="ID")
        split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
        rr = iss.merge(split[["ID", "split"]], on="ID")
        rr = rr[rr["split"].isin(["train", "val"])]
        self.proj_median_min = ((rr["r_s"] - rr["c_s"]) / 60).groupby(rr["Project_ID"]).median()
        self._prep_rates()
        self._prep_open()
        log("  FeatureEngine ready", flush=True)

    # ---------- precomputed sorted arrays for entity-level queries ----------
    def _prep_rates(self):
        self.rate_idx = {}
        for col in ["Project_ID", "Assignee_ID", "Reporter_ID"]:
            r = self.res[self.res[col].notna()]
            e = r[col].astype(np.int64).to_numpy()
            rs = r["r_s"].to_numpy(np.int64)
            o = np.lexsort((rs, e))
            self.rate_idx[col] = (e[o] * SCALE + rs[o], np.concatenate([[0], np.cumsum(r["delayed_v2"].to_numpy()[o])]))

    def _prep_open(self):
        self.open_idx = {}
        for col in ["Project_ID", "Assignee_ID"]:
            df = self.iss[self.iss[col].notna()]
            e = df[col].astype(np.int64).to_numpy()
            c = np.sort(e * SCALE + df["c_s"].to_numpy())
            has = df["r_s"].to_numpy() < FAR
            r = np.sort(e[has] * SCALE + df["r_s"].to_numpy()[has])
            self.open_idx[col] = (c, r)

    def _entity(self, iid, col):
        v = self.iss[col].reindex(iid).to_numpy()
        ok = ~pd.isna(v)
        return np.where(ok, np.nan_to_num(v.astype(float), nan=0).astype(np.int64), 0), ok

    def _rate(self, iid, T, col, window=None, m=10.0, p0=0.5):
        keys, cum = self.rate_idx[col]
        e, ok = self._entity(iid, col)
        q, b = e * SCALE + T, e * SCALE
        if window:
            b = np.maximum(b, q - window)
        hi, lo = np.searchsorted(keys, q, "left"), np.searchsorted(keys, b, "left")
        n, s = hi - lo, cum[hi] - cum[lo]
        return np.where(ok, (s + m * p0) / (n + m), np.nan), np.where(ok, n, 0)

    def _open(self, iid, T, col):
        c, r = self.open_idx[col]
        e, ok = self._entity(iid, col)
        q, b = e * SCALE + T, e * SCALE
        created = np.searchsorted(c, q, "left") - np.searchsorted(c, b, "left")
        resolved = np.searchsorted(r, q, "left") - np.searchsorted(r, b, "left")
        r30 = np.searchsorted(r, q, "left") - np.searchsorted(r, np.maximum(b, q - 30 * 86400), "left")
        return np.where(ok, created - resolved, np.nan), np.where(ok, r30, np.nan)

    # ---------- main entry ----------
    def at(self, issue_ids, T):
        iid = np.asarray(issue_ids, dtype=np.int64)
        T = np.asarray(T, dtype=np.int64)
        info = self.iss.reindex(iid)
        c_s = info["c_s"].to_numpy(np.int64)
        F = {}

        n, last, lo = self.cm.query(iid, T)
        F["n_comments"] = n
        F["n_comment_authors"] = self.cm.csum("first_auth", last, lo)
        F["comments_last_7d"] = self.cm.query(iid, T, 7 * 86400)[0]
        lt_c = self.cm.last_val("t", n, last, c_s)
        F["log_min_since_last_comment"] = np.log1p(np.maximum(T - lt_c, 0) / 60)

        n, last, lo = self.cl.query(iid, T)
        F["n_transitions"] = n
        F["n_reopens"] = self.cl.csum("is_reopen", last, lo)
        F["n_back_moves"] = self.cl.csum("is_back", last, lo)
        F["ever_in_progress"] = (self.cl.csum("is_prog", last, lo) > 0).astype(int)
        F["ever_in_review"] = (self.cl.csum("is_rev", last, lo) > 0).astype(int)
        ff = pd.Series(iid).map(self.first_from).fillna("no_history").to_numpy(object)
        F["status_bucket"] = self.cl.last_val("to_b", n, last, ff)
        lt_s = self.cl.last_val("t", n, last, c_s)
        F["log_min_since_last_change"] = np.log1p(np.maximum(T - lt_s, 0) / 60)
        F["log_min_since_last_activity"] = np.log1p(np.maximum(T - np.maximum(lt_c, lt_s), 0) / 60)

        po, pr30 = self._open(iid, T, "Project_ID")
        ao, _ = self._open(iid, T, "Assignee_ID")
        F["project_open_at_T"] = np.log1p(np.clip(po, 0, None))
        F["project_resolved_30d"] = np.log1p(np.clip(pr30, 0, None))
        F["assignee_open_at_T"] = np.log1p(np.clip(ao, 0, None))
        F["log_title_len"] = np.log1p(info["title_len"].to_numpy(float))
        F["log_desc_len"] = np.log1p(info["desc_len"].to_numpy(float))
        F["desc_has_code"] = info["desc_has_code"].to_numpy()

        # reassignment
        n, last, lo = self.pe.query(iid, T)
        F["n_assignee_changes"] = n
        F["ever_reassigned"] = (self.pe.csum("reas", last, lo) > 0).astype(int)
        F["assigned_at_T"] = self.pe.last_val("assigned", n, last, 0).astype(int)
        lt_a = self.pe.last_val("t", n, last, c_s)
        F["log_min_since_assign_change"] = np.log1p(np.maximum(T - lt_a, 0) / 60)
        first_t = np.where(n > 0, self.pe.t[np.clip(lo, 0, len(self.pe.t) - 1)], T)
        F["log_min_to_first_assign"] = np.log1p(np.maximum(first_t - c_s, 0) / 60)

        # sprint
        n, last, lo = self.sc.query(iid, T)
        F["n_sprint_changes"] = n
        F["n_sprint_carryovers"] = self.sc.csum("carry", last, lo)
        name = self.sc.last_val("name", n, last, None)
        F["in_sprint_at_T"] = pd.notna(name).astype(int)
        s_s, e_s = self.sprint_window(info["Project_ID"].to_numpy(), name)
        valid = (s_s > 0) & (e_s > 0)
        F["days_to_sprint_end"] = np.where(valid, (e_s - T) / 86400, np.nan)
        F["sprint_overdue"] = np.where(valid, (T > e_s).astype(int), 0)
        F["sprint_len_days"] = np.where(valid, (e_s - s_s) / 86400, np.nan)
        F["sprint_started"] = np.where(valid, (T >= s_s).astype(int), 0)

        n, last, lo = self.sp.query(iid, T)
        F["n_sp_changes"] = n
        F["sp_at_T"] = self.sp.last_val("val", n, last, np.nan)

        n, last, lo = self.fl.query(iid, T)
        F["n_field_chg_total"] = n
        F["n_field_chg_last_7d"] = self.fl.query(iid, T, 7 * 86400)[0]
        for g in self.field_groups:
            F[g] = self.fl.csum(g, last, lo)

        F["proj_rate_known"], _ = self._rate(iid, T, "Project_ID")
        F["proj_rate_known_90d"], n90 = self._rate(iid, T, "Project_ID", 90 * 86400)
        F["proj_n_resolved_90d"] = np.log1p(n90)
        F["assignee_rate_known"], na = self._rate(iid, T, "Assignee_ID")
        F["assignee_n_known"] = np.log1p(na)
        F["reporter_rate_known"], _ = self._rate(iid, T, "Reporter_ID")

        elapsed_min = (T - c_s) / 60
        pm = info["Project_ID"].map(self.proj_median_min).to_numpy(float)
        F["log_elapsed_over_median"] = np.log(np.maximum(elapsed_min, 1e-3) / pm)
        F["past_median"] = (elapsed_min > pm).astype(int)
        out = pd.DataFrame(F)
        out.insert(0, "Issue_ID", iid)
        out.insert(1, "T", T)
        return out

    def sprint_window(self, project_ids, names, with_done=False):
        names = pd.Series(names, dtype=object)
        key = pd.DataFrame({"Project_ID": project_ids, "Name": names})
        cols = ["s_s", "e_s", "done_s"]
        j = key.merge(self.sprint_by_proj[cols], left_on=["Project_ID", "Name"], right_index=True, how="left")
        miss = j["s_s"].isna() & names.notna().to_numpy()
        if miss.any():
            j2 = key[miss].merge(self.sprint_by_name[cols], left_on="Name", right_index=True, how="left")
            j.loc[miss, cols] = j2[cols].to_numpy()
        out = (j["s_s"].fillna(-1).to_numpy(float), j["e_s"].fillna(-1).to_numpy(float))
        return out + (j["done_s"].fillna(-1).to_numpy(float),) if with_done else out


def test_feature_engine(n=20000, seed=0):
    """Recompute features for random checkpoint rows and compare to the cached tables."""
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "elapsed_minutes"])
    s = ck.sample(n, random_state=seed).reset_index(drop=True)
    fe = FeatureEngine()
    c_s = fe.iss["c_s"].reindex(s["Issue_ID"]).to_numpy()
    T = c_s + np.floor(s["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    X = fe.at(s["Issue_ID"], T)
    X["checkpoint_index"] = s["checkpoint_index"].to_numpy()
    ref = pd.read_csv(f"{FEATURES_DIR}/phase8_activity_features.csv").merge(
        pd.read_csv(f"{FEATURES_DIR}/phase8_change_history_features.csv"), on=["Issue_ID", "checkpoint_index"])
    m = X.merge(ref, on=["Issue_ID", "checkpoint_index"], suffixes=("", "_ref"))
    bad = {}
    for c in [c for c in X.columns if c + "_ref" in m.columns]:
        a, b = m[c], m[c + "_ref"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            mis = (a.astype(str) != b.astype(str)).mean()
        else:
            mis = (~np.isclose(a.astype(float), b.astype(float), equal_nan=True, atol=1e-6)).mean()
        if mis > 0:
            bad[c] = mis
    print(f"compared {len(m)} rows; mismatching columns (share of rows): {bad or 'none'}")
    return bad


if __name__ == "__main__":
    test_feature_engine()
