"""
Backend for app/app_v5.py -- everything the UI needs, no Streamlit imports,
so it can be tested headless (python app_v5_backend.py).

  Backend()                         loads raw caches, static features, live + Scrum models
  .open_issues(project, date)       every issue open in the project at `date`, scored by the
                                    live dynamic model, with top reasons (TreeSHAP)
  .trajectory(issue_id, date)       weekly risk history of one issue up to `date`
  .active_sprints(project, date)    sprints running at `date`
  .sprint_issues(sprint_key, date)  committed issues still open in that sprint, with spill risk
All features are computed from events strictly before `date` (FeatureEngine).
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engine import FeatureEngine, FAR
from scrum_backend import STAGE_NAMES
from h_common import static_cols, CATEGORICAL_COLS

MODELS_DIR = "models"

FRIENDLY = {
    "log_min_since_last_change": "time since the status last moved",
    "log_min_since_last_activity": "time since any activity",
    "log_min_since_last_comment": "time since the last comment",
    "ever_in_review": "whether it has reached review/testing",
    "ever_in_progress": "whether work has started",
    "status_bucket": "its current workflow stage",
    "n_transitions": "number of status moves",
    "comments_last_7d": "comments in the last week",
    "n_comments": "number of comments", "n_comment_authors": "people discussing it",
    "elapsed_minutes": "how long it has been open",
    "log_elapsed_over_median": "age vs the project's typical duration",
    "past_median": "already older than the project's typical duration",
    "assignee_prior_delay_rate": "assignee's track record", "assignee_rate_known": "assignee's recent track record",
    "assignee_n_known": "assignee's experience", "reporter_rate_known": "reporter's track record",
    "proj_rate_known": "project's track record", "proj_rate_known_90d": "project's last-90-day track record",
    "Project_ID": "the project", "Type_Normalized": "issue type", "Priority_Normalized": "priority",
    "Story_Point": "story points", "sp_at_T": "current story points",
    "log_min_since_assign_change": "time since (re)assignment", "n_assignee_changes": "number of reassignments",
    "assigned_at_T": "whether someone is assigned", "log_min_to_first_assign": "how long until first assigned",
    "n_field_chg_last_7d": "edits in the last week", "n_field_chg_total": "number of edits",
    "n_links": "number of links", "n_blocking_links": "blocking links",
    "in_sprint_at_T": "whether it is in a sprint", "n_sprint_carryovers": "sprint carry-overs",
    "days_to_sprint_end": "days left in its sprint",
    "project_open_at_T": "project backlog size", "assignee_open_at_T": "assignee's open workload",
}


def friendly(c):
    if c.startswith("text_pc_"):
        return "what the description says"
    return FRIENDLY.get(c, c.replace("_", " "))


def _ensemble(prefix, seeds):
    return [joblib.load(f"{MODELS_DIR}/{prefix}_seed{s}.joblib") for s in seeds]


class Backend:
    def __init__(self, live_variant="v5b"):
        self.fe = FeatureEngine(verbose=False)
        self.live_meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_{live_variant}_meta.json"))
        self.live = _ensemble(f"xgb_dynamic_live_{live_variant}", self.live_meta["seeds"])
        self.cols = self.live_meta["features"]
        try:
            self.scrum_meta = json.load(open(f"{MODELS_DIR}/xgb_scrum_spillover_meta.json"))
            self.scrum = _ensemble("xgb_scrum_spillover", self.scrum_meta["seeds"])
        except FileNotFoundError:
            self.scrum_meta, self.scrum = None, None
        feat = pd.read_csv("data/features/phase6_feature_table_v3.csv")
        self.static = feat.set_index("ID")
        try:
            titles = pd.read_csv("data/raw/issue.csv", usecols=["ID", "Title"]).set_index("ID")["Title"]
        except (FileNotFoundError, ValueError):
            titles = pd.Series(dtype=str)
        self.titles = titles.astype(str).str.strip('"')
        self._scrum_pop = None

    # ------------------------------------------------------------------ helpers
    def _design(self, ids, T):
        ids = np.asarray(ids, dtype=np.int64)
        X = self.fe.at(ids, T)
        st = self.static.reindex(ids)
        for c in static_cols(30):
            X[c] = st[c].to_numpy()
        X["elapsed_minutes"] = (np.asarray(T) - self.fe.iss["c_s"].reindex(ids).to_numpy()) / 60
        cats = self.live_meta.get("categories", {})
        for c in self.cols:
            if c in cats:
                if c == "Project_ID":   # trained with integer categories
                    X[c] = pd.Categorical(X[c].astype(int), categories=[int(v) for v in cats[c]])
                else:
                    X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
            elif X[c].dtype == object:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def _predict(self, models, X, cols):
        return np.mean([m.predict_proba(X[cols])[:, 1] for m in models], axis=0)

    def _reasons(self, X, k=3):
        dm = xgb.DMatrix(X[self.cols], enable_categorical=True)
        contrib = self.live[0].get_booster().predict(dm, pred_contribs=True)[:, :-1]
        C = pd.DataFrame(contrib, columns=self.cols)
        text = [c for c in self.cols if c.startswith("text_pc_")]
        C["__text__"] = C[text].sum(axis=1)
        C = C.drop(columns=text)
        out = []
        for i in range(len(C)):
            row = C.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            out.append([(friendly("text_pc_0") if n == "__text__" else friendly(n), float(v)) for n, v in top.items()])
        return out, C

    # ------------------------------------------------------------------ portfolio
    def projects(self):
        iss = self.fe.iss
        have = iss[iss["ID"].isin(self.static.index)]
        g = have.groupby("Project_ID").agg(n=("ID", "size"), first=("c_s", "min"), last=("c_s", "max"))
        return g

    def open_issues(self, project, date):
        T = int(pd.Timestamp(date).timestamp())
        iss = self.fe.iss
        m = (iss["Project_ID"] == project) & (iss["c_s"] < T) & (iss["r_s"] > T) & iss["ID"].isin(self.static.index)
        ids = iss.loc[m, "ID"].to_numpy()
        if len(ids) == 0:
            return pd.DataFrame()
        X = self._design(ids, np.full(len(ids), T))
        risk = self._predict(self.live, X, self.cols)
        reasons, _ = self._reasons(X)
        thr = self.live_meta["prior_match_threshold_global"]
        df = pd.DataFrame({
            "Issue_ID": ids, "Title": self.titles.reindex(ids).fillna("").to_numpy(),
            "age_days": X["elapsed_minutes"].to_numpy() / 1440,
            "stage": X["status_bucket"].astype(str).map(STAGE_NAMES).fillna(X["status_bucket"].astype(str)).to_numpy(),
            "assignee": iss["Assignee_ID"].reindex(ids).to_numpy(), "risk": risk,
            "flag": np.where(X["past_median"].to_numpy() == 1, "already overdue",
                             np.where(risk >= thr, "likely late", "on track")),
            "overdue": X["past_median"].to_numpy() == 1,
            "why": [", ".join(f"{n} ({'↑' if v > 0 else '↓'})" for n, v in r) for r in reasons],
        })
        res = iss["r_s"].reindex(ids).to_numpy()
        df["actually_late"] = np.where(res < FAR, (self.static.reindex(ids)["delayed_v2"].to_numpy()), np.nan)
        return df.sort_values("risk", ascending=False).reset_index(drop=True)

    def trajectory(self, issue_id, date, max_points=40):
        c = int(self.fe.iss.loc[issue_id, "c_s"])
        r = int(self.fe.iss.loc[issue_id, "r_s"])
        end = min(int(pd.Timestamp(date).timestamp()), r - 60)
        if end <= c + 3600:
            return pd.DataFrame(columns=["date", "risk"])
        Ts = np.unique(np.linspace(c + 3600, end, num=max_points).astype(np.int64))
        X = self._design(np.full(len(Ts), issue_id), Ts)
        return pd.DataFrame({"date": pd.to_datetime(Ts, unit="s"), "risk": self._predict(self.live, X, self.cols)})

    def explain_issue(self, issue_id, date):
        T = int(pd.Timestamp(date).timestamp())
        X = self._design([issue_id], [T])
        _, C = self._reasons(X, k=8)
        row = C.iloc[0]
        row = row.reindex(row.abs().sort_values(ascending=False).index)[:8]
        return pd.DataFrame({"factor": [friendly("text_pc_0") if n == "__text__" else friendly(n) for n in row.index],
                             "push": row.to_numpy()})

    # ------------------------------------------------------------------ scrum
    def _pop(self):
        if self._scrum_pop is None:
            from task_j5_scrum_mode import build_population
            self._scrum_pop = build_population(self.fe)
        return self._scrum_pop

    def active_sprints(self, project, date):
        T = int(pd.Timestamp(date).timestamp())
        pop = self._pop()
        p = pop[(pop["Project_ID"] == project) & (pop["s_s"] <= T) & (pop["e_s"] > T)]
        return p.groupby("sprint_key").agg(name=("name", "first"), start=("s_s", "min"), end=("planned_e_s", "max"),
                                           committed=("Issue_ID", "size")).reset_index()

    def sprint_issues(self, sprint_key, date):
        if self.scrum is None:
            return pd.DataFrame()
        from task_j5_scrum_mode import sprint_context
        T = int(pd.Timestamp(date).timestamp())
        pop = self._pop()
        r = pop[(pop["sprint_key"] == sprint_key) & (pop["join_t"] <= T) & (pop["r_s"] > T) & (pop["leave_t"] > T)].copy()
        if r.empty:
            return pd.DataFrame()
        r["T"] = T
        r["progress_actual"] = (T - r["s_s"]) / (r["planned_e_s"] - r["s_s"])
        r["days_left"] = (r["planned_e_s"] - T) / 86400
        r["sprint_len"] = (r["planned_e_s"] - r["s_s"]) / 86400
        r["joined_late"] = (r["t"] > r["s_s"]).astype(int)
        r["issue_age_days"] = (T - r["c_s"]) / 86400
        r = sprint_context(pop, r)
        X = self.fe.at(r["Issue_ID"].to_numpy(), r["T"].to_numpy())
        for c in X.columns:
            if c not in ("Issue_ID", "T") and c not in r.columns:
                r[c] = X[c].to_numpy()
        info = self.fe.iss.reindex(r["Issue_ID"])
        r["Priority"] = info["Priority"].fillna("Unknown").astype(str).to_numpy()
        r["Type"] = info["Type"].fillna("Unknown").astype(str).to_numpy()
        r["Project_cat"] = r["Project_ID"].astype(int)
        cols = self.scrum_meta["features"]
        cats = self.scrum_meta.get("categories", {})
        for c in ["Priority", "Type", "status_bucket", "Project_cat"]:
            vals = r[c].astype(int) if c == "Project_cat" else r[c].astype(str)
            r[c] = pd.Categorical(vals, categories=cats.get(c)) if c in cats else vals.astype("category")
        try:
            risk = np.mean([m.predict_proba(r[cols])[:, 1] for m in self.scrum], axis=0)
        except Exception:
            risk = np.full(len(r), np.nan)
        prog = min([0.0, 0.25, 0.5, 0.75], key=lambda p: abs(p - float(r["progress_actual"].iloc[0])))
        thr = self.scrum_meta["prior_match_thresholds_by_progress"][str(prog)]
        return pd.DataFrame({"Issue_ID": r["Issue_ID"].to_numpy(),
                             "Title": self.titles.reindex(r["Issue_ID"]).fillna("").to_numpy(),
                             "stage": r["status_bucket"].astype(str).to_numpy(), "spill_risk": risk,
                             "flag": np.where(risk >= thr, "likely to spill", "on track"),
                             "actually_spilled": r["spilled"].to_numpy()}).sort_values("spill_risk", ascending=False)


if __name__ == "__main__":
    b = Backend()
    pr = b.projects().sort_values("n", ascending=False)
    p = int(pr.index[0])
    date = pd.to_datetime(pr.loc[p, "last"], unit="s") - pd.Timedelta(days=200)
    df = b.open_issues(p, date)
    print(f"project {p} on {date.date()}: {len(df)} open issues")
    print(df.head(8).to_string())
    if len(df):
        from sklearn.metrics import roc_auc_score
        d = df.dropna(subset=["actually_late"])
        if d["actually_late"].nunique() == 2:
            print("AUC on this snapshot:", round(roc_auc_score(d["actually_late"], d["risk"]), 3))
        print(b.trajectory(int(df.Issue_ID.iloc[0]), date).tail())
        print(b.explain_issue(int(df.Issue_ID.iloc[0]), date))
    sp = b.active_sprints(p, date)
    print(sp)
    if len(sp):
        print(b.sprint_issues(sp.sprint_key.iloc[0], date).head(10))
