"""
Scrum-mode backend for app/app_v5.py (no Streamlit imports; testable headless).

In Scrum mode the PM has declared the project runs sprints, so the deadline of
every committed issue is its sprint's end. Everything here is about that
deadline:

  sprints(project)                  all sprints of the project (from the Sprint table + change log)
  snapshot(sprint_key, T)           committed issues still open at T with spill risk + reasons,
                                    and sprint KPIs (done / open / removed / expected spills)
  burndown(sprint_key, T)           daily open-commitment count from sprint start to T, and the
                                    model's projected spill count at T
  issue_trajectory(iid, sprint, T)  one issue's spill risk through the sprint
  history(project, before_T)        past sprints: committed, actually spilled, and the spill count the
                                    model predicted at sprint start (expected = sum of risks)
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from task_j5_scrum_mode import build_population, sprint_context

MODELS_DIR = "models"
PROGRESS = [0.0, 0.25, 0.5, 0.75]
STAGE_NAMES = {"no_history": "not started (no status moves yet)", "todo": "to do", "in_progress": "in progress",
               "review": "in review", "testing": "testing", "waiting": "waiting / blocked", "done": "done (not closed)",
               "reopened": "reopened", "other": "other"}

FRIENDLY = {
    "log_min_since_last_change": "time since the status last moved",
    "log_min_since_last_activity": "time since any activity",
    "log_min_since_last_comment": "time since the last comment",
    "status_bucket": "its current workflow stage", "ever_in_review": "whether it reached review",
    "days_left": "days left in the sprint", "progress_actual": "how far into the sprint we are",
    "sprint_resolved_share_at_T": "share of the sprint already done",
    "sprint_load_at_T": "how many issues the sprint holds",
    "team_spill_rate_known": "team's spill rate in past sprints", "carried_in": "carried in from an earlier sprint",
    "joined_late": "added after the sprint started", "n_sprint_carryovers": "previous carry-overs",
    "n_sprint_changes": "sprint moves", "assignee_n_known": "assignee's experience",
    "assignee_rate_known": "assignee's track record", "sp_at_T": "story points",
    "Project_cat": "the project", "Type": "issue type", "Priority": "priority",
    "issue_age_days": "how old the issue is", "n_field_chg_last_7d": "edits in the last week",
    "log_min_since_assign_change": "time since (re)assignment",
    "team_sprint_history_n": "how many past sprints the team has run", "n_transitions": "number of status moves",
    "ever_in_progress": "whether work has started", "n_comments": "number of comments",
    "comments_last_7d": "comments in the last week", "assigned_at_T": "whether someone is assigned",
}


def friendly(c):
    return FRIENDLY.get(c, c.replace("_", " "))


class ScrumBackend:
    def __init__(self, fe, titles):
        self.fe = fe
        self.titles = titles
        self.meta = json.load(open(f"{MODELS_DIR}/xgb_scrum_spillover_meta.json"))
        self.models = [joblib.load(f"{MODELS_DIR}/xgb_scrum_spillover_seed{s}.joblib") for s in self.meta["seeds"]]
        self.cols = self.meta["features"]
        self.cats = self.meta.get("categories", {})
        self.pop = build_population(fe)
        try:   # Task L5: split-conformal residual quantiles of the per-sprint spill count
            self.spill_q = json.load(open(f"{MODELS_DIR}/conformal_calibration.json"))["scrum_sprint_count"]
        except (FileNotFoundError, KeyError):
            self.spill_q = None

    def spill_interval(self, expected_open, n_open, removed, progress):
        """90% interval for the number of committed issues that will spill (Task L5)."""
        if self.spill_q is None:
            return None
        key = min(self.spill_q, key=lambda p: abs(float(p) - progress))
        q = self.spill_q[key]
        lo = np.clip(np.floor(expected_open + q["residual_q05"]), 0, n_open)
        hi = np.clip(np.ceil(expected_open + q["residual_q95"]), 0, n_open)
        return int(lo + removed), int(hi + removed)

    # ------------------------------------------------------------------ core scoring
    def _score(self, r):
        """r: pop rows with a T column. Returns r with features, risk and top reasons."""
        r = r.copy()
        T = r["T"].to_numpy()
        r["progress_actual"] = (T - r["s_s"]) / (r["planned_e_s"] - r["s_s"])
        r["days_left"] = (r["planned_e_s"] - T) / 86400
        r["sprint_len"] = (r["planned_e_s"] - r["s_s"]) / 86400
        r["joined_late"] = (r["t"] > r["s_s"]).astype(int)
        r["issue_age_days"] = (T - r["c_s"]) / 86400
        r = sprint_context(self.pop, r)
        X = self.fe.at(r["Issue_ID"].to_numpy(), T)
        for c in X.columns:
            if c not in ("Issue_ID", "T") and c not in r.columns:
                r[c] = X[c].to_numpy()
        info = self.fe.iss.reindex(r["Issue_ID"])
        r["Priority"] = info["Priority"].fillna("Unknown").astype(str).to_numpy()
        r["Type"] = info["Type"].fillna("Unknown").astype(str).to_numpy()
        r["Project_cat"] = r["Project_ID"].astype(int)
        for c in ["Priority", "Type", "status_bucket", "Project_cat"]:
            vals = r[c].astype(int) if c == "Project_cat" else r[c].astype(str)
            r[c] = pd.Categorical(vals, categories=self.cats.get(c))
        r["risk"] = np.mean([m.predict_proba(r[self.cols])[:, 1] for m in self.models], axis=0)
        return r

    def _reasons(self, r, k=3):
        dm = xgb.DMatrix(r[self.cols], enable_categorical=True)
        C = pd.DataFrame(self.models[0].get_booster().predict(dm, pred_contribs=True)[:, :-1], columns=self.cols)
        out = []
        for i in range(len(C)):
            row = C.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            out.append([(friendly(n), float(v)) for n, v in top.items()])
        return out, C

    def _thr(self, progress):
        p = min(PROGRESS, key=lambda x: abs(x - progress))
        return self.meta["prior_match_thresholds_by_progress"][str(p)]

    # ------------------------------------------------------------------ public API
    def projects(self):
        g = self.pop.groupby("Project_ID").agg(sprints=("sprint_key", "nunique"), issues=("Issue_ID", "size"))
        return g[g.sprints >= 5].sort_values("issues", ascending=False)

    def sprints(self, project):
        p = self.pop[self.pop["Project_ID"] == project]
        g = p.groupby("sprint_key").agg(name=("name", "first"), start=("s_s", "min"), planned_end=("planned_e_s", "max"),
                                        close=("e_s", "max"), committed=("Issue_ID", "size"),
                                        spilled=("spilled", "sum")).reset_index()
        return g.sort_values("start", ascending=False).reset_index(drop=True)

    def snapshot(self, sprint_key, T):
        T = int(T)
        s = self.pop[self.pop["sprint_key"] == sprint_key]
        joined = s[s["join_t"] <= T]
        done = joined[joined["r_s"] <= T]
        removed = joined[(joined["r_s"] > T) & (joined["leave_t"] <= T)]
        open_ = joined[(joined["r_s"] > T) & (joined["leave_t"] > T)].copy()
        kpi = {"committed": int(len(joined)), "done": int(len(done)), "removed": int(len(removed)),
               "open": int(len(open_)), "expected_spills": 0.0, "days_left": float((s["planned_e_s"].max() - T) / 86400),
               "progress": float((T - s["s_s"].min()) / max(s["planned_e_s"].max() - s["s_s"].min(), 1))}
        if open_.empty:
            return kpi, pd.DataFrame()
        open_["T"] = T
        r = self._score(open_)
        reasons, _ = self._reasons(r)
        thr = self._thr(kpi["progress"])
        kpi["expected_spills"] = float(r["risk"].sum() + len(removed))   # removed-and-open issues have spilled
        kpi["spill_interval"] = self.spill_interval(float(r["risk"].sum()), len(open_), len(removed), kpi["progress"])
        kpi["flagged"] = int((r["risk"] >= thr).sum())
        tbl = pd.DataFrame({
            "Issue_ID": r["Issue_ID"].to_numpy(), "Title": self.titles.reindex(r["Issue_ID"]).fillna("").to_numpy(),
            "stage": r["status_bucket"].astype(str).map(STAGE_NAMES).fillna(r["status_bucket"].astype(str)).to_numpy(),
            "story_points": r["sp_at_T"].to_numpy(),
            "spill_risk": r["risk"].to_numpy(), "flag": np.where(r["risk"] >= thr, "likely to spill", "on track"),
            "why": [", ".join(f"{n} ({'↑' if v > 0 else '↓'})" for n, v in x) for x in reasons],
            "actually_spilled": r["spilled"].to_numpy()})
        return kpi, tbl.sort_values("spill_risk", ascending=False).reset_index(drop=True)

    def burndown(self, sprint_key, T):
        s = self.pop[self.pop["sprint_key"] == sprint_key]
        start = int(s["s_s"].min())
        end = int(s["planned_e_s"].max())
        days = pd.date_range(pd.to_datetime(start, unit="s").normalize(), pd.to_datetime(min(T, end), unit="s"), freq="D")
        rows = []
        for d in days:
            t = int(d.timestamp()) + 86399
            t = min(t, int(T))
            joined = s[s["join_t"] <= t]
            rows.append({"date": d, "open_commitments": int(((joined["r_s"] > t) & (joined["leave_t"] > t)).sum()),
                         "committed_so_far": int(len(joined))})
        return pd.DataFrame(rows)

    def issue_trajectory(self, issue_id, sprint_key, T, points=12):
        s = self.pop[(self.pop["sprint_key"] == sprint_key) & (self.pop["Issue_ID"] == issue_id)]
        if s.empty:
            return pd.DataFrame(columns=["date", "spill_risk"])
        row = s.iloc[0]
        lo, hi = int(row["join_t"]) + 60, int(min(T, row["r_s"] - 60, row["leave_t"] - 60, row["e_s"] - 60))
        if hi <= lo:
            return pd.DataFrame(columns=["date", "spill_risk"])
        Ts = np.unique(np.linspace(lo, hi, points).astype(np.int64))
        r = pd.concat([s] * len(Ts), ignore_index=True)
        r["T"] = Ts
        r = self._score(r)
        return pd.DataFrame({"date": pd.to_datetime(Ts, unit="s"), "spill_risk": r["risk"].to_numpy()})

    def explain(self, issue_id, sprint_key, T, k=8):
        s = self.pop[(self.pop["sprint_key"] == sprint_key) & (self.pop["Issue_ID"] == issue_id)].copy()
        s["T"] = int(T)
        r = self._score(s)
        _, C = self._reasons(r, k)
        row = C.iloc[0]
        row = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
        return pd.DataFrame({"factor": [friendly(n) for n in row.index], "push": row.to_numpy()})

    def history(self, project, before_T, max_sprints=40):
        """Past closed sprints: predicted (at sprint start) vs actual spill counts."""
        sp = self.sprints(project)
        sp = sp[sp["close"] < before_T].head(max_sprints)
        if sp.empty:
            return sp
        p = self.pop[self.pop["sprint_key"].isin(sp["sprint_key"])].copy()
        # snapshot at the start of each sprint (or when the issue joined), like the model's progress-0 checkpoint
        p["T"] = np.maximum(p["s_s"], p["join_t"] + 60).astype(np.int64)
        p = p[(p["r_s"] > p["T"]) & (p["leave_t"] > p["T"]) & (p["T"] < p["e_s"])]
        r = self._score(p)
        agg = r.groupby("sprint_key").agg(predicted_spills=("risk", "sum"), open_at_start=("risk", "size"),
                                          actual_spills_open_at_start=("spilled", "sum")).reset_index()
        out = sp.merge(agg, on="sprint_key", how="left")
        out["start_date"] = pd.to_datetime(out["start"], unit="s").dt.date
        out["completion_rate"] = 1 - out["spilled"] / out["committed"]
        return out.sort_values("start").reset_index(drop=True)


if __name__ == "__main__":
    from feature_engine import FeatureEngine
    fe = FeatureEngine(verbose=False)
    sb = ScrumBackend(fe, pd.Series(dtype=str))
    pr = sb.projects()
    print(pr.head())
    proj = int(pr.index[0])
    sps = sb.sprints(proj)
    print(sps.head())
    key = sps.iloc[3]["sprint_key"]
    row = sps.iloc[3]
    T = int(row["start"] + 0.5 * (row["planned_end"] - row["start"]))
    kpi, tbl = sb.snapshot(key, T)
    print(kpi)
    print(tbl.head(8).to_string())
    print(sb.burndown(key, T).tail())
    if len(tbl):
        print(sb.issue_trajectory(int(tbl.Issue_ID.iloc[0]), key, T))
        print(sb.explain(int(tbl.Issue_ID.iloc[0]), key, T))
    h = sb.history(proj, T)
    print(h[["name", "start_date", "committed", "spilled", "predicted_spills", "actual_spills_open_at_start"]].tail(10))
    print("corr predicted vs actual (history):", np.corrcoef(h["predicted_spills"].fillna(0), h["actual_spills_open_at_start"].fillna(0))[0, 1])
