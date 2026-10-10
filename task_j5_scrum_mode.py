"""
Task J5: SCRUM MODE -- predict sprint spillover for sprint-tracked issues.

Question a Scrum team actually asks: "will this issue be finished by the end
of the sprint it was committed to?"

Population
  every issue whose FIRST sprint assignment (Change_Log, Field='Sprint') can be
  matched to a CLOSED sprint in the Sprint table with valid start/end dates
  (project+name match, unique-name fallback). Unresolved issues are INCLUDED:
  an issue still open at sprint end has spilled, so this label does not suffer
  from the resolved-only censoring found in J1.
Label
  spilled = not resolved by the sprint's close + 1 day grace, where close =
  min(planned End_Date, actual Complete_Date). Using the actual close matters:
  a third of sprints were completed before their planned end, and Jira logs
  the carry-over of unfinished issues at completion -- a checkpoint placed
  after that moment would see the carry-over (a leak). Checkpoints therefore
  always sit strictly before the real close.
Checkpoints (sprint-progress milestones, Kula-style)
  p in {0, 0.25, 0.5, 0.75} of the PLANNED sprint window; T = max(start + p*len,
  time the issue joined the sprint); a row exists only if the issue is still
  unresolved at T and T is before sprint end.
Features at T
  FeatureEngine.at(): status/comment/activity/reassignment/field-churn/known-rate
  features, plus sprint features: progress, days_left, sprint_len, joined_late,
  sprint_load_at_T, sprint_resolved_share_at_T (burndown), sprint_carried_in,
  team_spill_rate_known (project's earlier closed sprints, outcomes known by T)
  and Priority / Type / Project as categories.
Split: chronological by sprint start, 70 / 10 / 20 of sprints.
Models: team-history baseline (logistic on team_spill_rate_known + progress) vs
XGBoost (multi-seed val selection, 5 seeds). Paired bootstrap CIs.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from feature_engine import FeatureEngine, secs, SCALE, FAR
from h_common import REPORTS_DIR, MODELS_DIR, SEEDS, seeded, paired_bootstrap_seeds, seed_metric_panel, verdict

LABEL = "spilled"
PROGRESS = [0.0, 0.25, 0.5, 0.75]
GRID = [{"learning_rate": lr, "max_depth": d} for lr in [0.05, 0.1] for d in [4, 6, 8]]
SEL_SEEDS = [0, 1, 2]


def build_population(fe):
    """Committed sprint = the first sprint the issue was actually IN while that sprint was running.

    Sprint-field changes give a membership timeline per issue (membership after an event = the last sprint
    named in To_String). Planning moves before a sprint starts (issue put in sprint A, moved to B before A
    began) do NOT count as a commitment to A -- an earlier version did count them, which made 'spilled from
    A' trivially predictable from the later move."""
    sc = pd.read_csv("data/raw/change_log_sprint.csv")
    sc["t"] = secs(sc["Creation_Date"])
    sc = sc.sort_values(["Issue_ID", "t"], kind="mergesort").reset_index(drop=True)
    sc["name"] = sc["To_String"].map(lambda v: v.split(",")[-1].strip() if isinstance(v, str) else None)
    sc["next_t"] = sc.groupby("Issue_ID")["t"].shift(-1).fillna(FAR).astype(np.int64)
    sc = sc[sc["name"].notna()].copy()
    sc["Project_ID"] = fe.iss["Project_ID"].reindex(sc["Issue_ID"]).to_numpy()
    s_s, e_s, done_s = fe.sprint_window(sc["Project_ID"].to_numpy(), sc["name"].to_numpy(), with_done=True)
    sc["s_s"], sc["planned_e_s"] = s_s, e_s
    sc["e_s"] = np.where(done_s > s_s, np.minimum(e_s, done_s), e_s)   # real close
    spr = fe.spr
    closed_pn = set(zip(spr.loc[spr.State == "CLOSED", "Project_ID"], spr.loc[spr.State == "CLOSED", "Name"]))
    closed_n = set(spr.loc[(spr.State == "CLOSED") & ~spr["Name"].duplicated(keep=False), "Name"])
    sc["closed"] = [((p, n) in closed_pn) or (n in closed_n) for p, n in zip(sc["Project_ID"], sc["name"])]
    # membership interval [t, next_t) must overlap the running sprint [s_s, e_s)
    ok = (sc.s_s > 0) & (sc.e_s > sc.s_s) & sc.closed & (sc.t < sc.e_s) & (sc.next_t > sc.s_s) & \
         ((sc.planned_e_s - sc.s_s) < 60 * 86400)
    cand = sc[ok]
    first = cand.drop_duplicates("Issue_ID").copy()
    first["join_t"] = np.maximum(first["t"], first["s_s"]).astype(np.int64)   # in sprint from here
    first["leave_t"] = first["next_t"]                                         # until the next sprint change
    first["carried_in"] = first["From_String"].notna().astype(int)
    info = fe.iss.reindex(first["Issue_ID"])
    first["c_s"], first["r_s"] = info["c_s"].to_numpy(), info["r_s"].to_numpy()
    proj_end = fe.iss.groupby("Project_ID")["c_s"].max().combine(
        fe.iss[fe.iss.r_s < FAR].groupby("Project_ID")["r_s"].max(), max)
    first["proj_end"] = first["Project_ID"].map(proj_end)
    pop = first[(first.e_s + 86400 < first.proj_end) & (first.r_s > first.join_t)].copy()
    pop["t"] = pop["join_t"]
    pop[LABEL] = (pop["r_s"] > pop["e_s"] + 86400).astype(int)
    pop["sprint_key"] = pop["Project_ID"].astype(int).astype(str) + "|" + pop["name"]
    print(f"sprint-tracked issues: {sc['Issue_ID'].nunique()}; committed to a matched closed sprint: {len(pop)}; "
          f"spill rate {pop[LABEL].mean():.3f}", flush=True)
    return pop


def sprint_context(pop, rows):
    """Sprint load / burndown / team history at each row's T (only info before T)."""
    # membership events: issue joins sprint at t
    pop = pop.reset_index(drop=True)
    kid = {k: i for i, k in enumerate(pop["sprint_key"].unique())}
    pk = pop["sprint_key"].map(kid).to_numpy(np.int64)
    join_keys = np.sort(pk * SCALE + pop["t"].to_numpy(np.int64))
    res_keys = np.sort(pk * SCALE + np.minimum(pop["r_s"].to_numpy(np.int64), FAR))
    rk = rows["sprint_key"].map(kid).to_numpy(np.int64)
    T = rows["T"].to_numpy(np.int64)
    q, b = rk * SCALE + T, rk * SCALE
    load = np.searchsorted(join_keys, q) - np.searchsorted(join_keys, b)
    resolved = np.searchsorted(res_keys, q) - np.searchsorted(res_keys, b)
    rows["sprint_load_at_T"] = load
    rows["sprint_resolved_share_at_T"] = resolved / np.maximum(load, 1)
    # team history: outcomes of the project's earlier sprints become known at their end + 1 day
    proj = pop["Project_ID"].astype(np.int64).to_numpy()
    known_t = (pop["e_s"] + 86400).astype(np.int64).to_numpy()
    o = np.lexsort((known_t, proj))
    keys = proj[o] * SCALE + known_t[o]
    cum = np.concatenate([[0], np.cumsum(pop[LABEL].to_numpy()[o])])
    rp = rows["Project_ID"].astype(np.int64).to_numpy()
    hi, lo = np.searchsorted(keys, rp * SCALE + T), np.searchsorted(keys, rp * SCALE)
    n, s = hi - lo, cum[hi] - cum[lo]
    rows["team_spill_rate_known"] = (s + 5 * 0.5) / (n + 5)
    rows["team_sprint_history_n"] = np.log1p(n)
    return rows


def build_rows(fe, pop):
    out = []
    for p in PROGRESS:
        T = np.maximum.reduce([pop["s_s"] + p * (pop["planned_e_s"] - pop["s_s"]), pop["t"] + 60,
                               pop["c_s"] + 60]).astype(np.int64)
        keep = (pop["r_s"].to_numpy() > T) & (T < pop["e_s"].to_numpy()) & (T < pop["leave_t"].to_numpy())
        r = pop[keep].copy()
        r["T"] = T[keep]
        r["progress"] = p
        out.append(r)
    rows = pd.concat(out, ignore_index=True)
    rows["progress_actual"] = (rows["T"] - rows["s_s"]) / (rows["planned_e_s"] - rows["s_s"])
    rows["sprint_len"] = (rows["planned_e_s"] - rows["s_s"]) / 86400
    rows["days_left"] = (rows["planned_e_s"] - rows["T"]) / 86400  # planned end is what the team sees at T
    rows["joined_late"] = (rows["t"] > rows["s_s"]).astype(int)
    rows["issue_age_days"] = (rows["T"] - rows["c_s"]) / 86400
    rows = sprint_context(pop, rows)
    X = fe.at(rows["Issue_ID"].to_numpy(), rows["T"].to_numpy())
    drop = {"Issue_ID", "T", "past_median", "log_elapsed_over_median",  # tied to the other label
            "days_to_sprint_end", "sprint_overdue", "sprint_len_days", "sprint_started"}  # duplicated by sprint feats
    for c in X.columns:
        if c not in drop:
            rows[c] = X[c].to_numpy()
    info = fe.iss.reindex(rows["Issue_ID"])
    rows["Priority"] = info["Priority"].fillna("Unknown").astype(str).to_numpy()
    rows["Type"] = info["Type"].fillna("Unknown").astype(str).to_numpy()
    for c in ["Priority", "Type", "status_bucket"]:
        rows[c] = rows[c].astype("category")
    rows["Project_cat"] = rows["Project_ID"].astype(int).astype("category")
    # chronological split by sprint start
    sk = rows.drop_duplicates("sprint_key")[["sprint_key", "s_s"]].sort_values("s_s")
    n = len(sk)
    sk["split"] = ["train"] * int(0.7 * n) + ["val"] * (int(0.8 * n) - int(0.7 * n)) + ["test"] * (n - int(0.8 * n))
    rows = rows.merge(sk[["sprint_key", "split"]], on="sprint_key")
    return rows


FEATS_SPRINT = ["progress_actual", "days_left", "sprint_len", "joined_late", "issue_age_days", "carried_in",
                "sprint_load_at_T", "sprint_resolved_share_at_T", "team_spill_rate_known", "team_sprint_history_n"]


def main():
    fe = FeatureEngine()
    pop = build_population(fe)
    rows = build_rows(fe, pop)
    rows.to_csv("data/features/scrum_rows.csv.gz", index=False, compression="gzip")
    eng = ["n_comments", "n_comment_authors", "comments_last_7d", "log_min_since_last_comment", "n_transitions",
           "n_reopens", "n_back_moves", "ever_in_progress", "ever_in_review", "status_bucket",
           "log_min_since_last_change", "log_min_since_last_activity", "project_open_at_T", "project_resolved_30d",
           "assignee_open_at_T", "log_title_len", "log_desc_len", "desc_has_code", "n_assignee_changes",
           "ever_reassigned", "assigned_at_T", "log_min_since_assign_change", "log_min_to_first_assign",
           "n_sprint_changes", "n_sprint_carryovers", "n_sp_changes", "sp_at_T", "n_field_chg_total",
           "n_field_chg_last_7d", "n_priority_chg", "n_type_chg", "n_rank_chg", "n_link_chg", "n_fixver_chg",
           "n_epic_chg", "n_estimate_chg", "n_label_chg", "n_component_chg", "proj_rate_known",
           "proj_rate_known_90d", "proj_n_resolved_90d", "assignee_rate_known", "assignee_n_known",
           "reporter_rate_known"]
    FULL = FEATS_SPRINT + eng + ["Priority", "Type", "Project_cat"]
    print(rows.groupby("split").agg(rows=("Issue_ID", "size"), issues=("Issue_ID", "nunique"),
                                    sprints=("sprint_key", "nunique"), spill=(LABEL, "mean")).to_string(), flush=True)
    print(rows.groupby("progress")[LABEL].agg(["size", "mean"]).round(3).to_string(), flush=True)

    tr, va = rows[rows.split == "train"], rows[rows.split == "val"]
    tv, te = rows[rows.split.isin(["train", "val"])], rows[rows.split == "test"].reset_index(drop=True)
    y, prog = te[LABEL].to_numpy(), te["progress"].to_numpy()

    # baseline: team history + sprint progress only
    bcols = ["team_spill_rate_known", "progress_actual", "days_left"]
    lr = LogisticRegression(max_iter=1000).fit(tv[bcols].fillna(0.5), tv[LABEL])
    P_base = [lr.predict_proba(te[bcols].fillna(0.5))[:, 1]]

    # XGBoost full, multi-seed selection
    best, g_rows = None, []
    for cfg in GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(tr[FULL], tr[LABEL], eval_set=[(va[FULL], va[LABEL])], verbose=False)
            aucs.append(roc_auc_score(va[LABEL], m.predict_proba(va[FULL])[:, 1]))
            iters.append(m.best_iteration + 1)
        g_rows.append({**cfg, "val_auc": np.mean(aucs), "trees": int(np.median(iters))})
        print(f"  {cfg}: val {np.mean(aucs):.4f}", flush=True)
    g = pd.DataFrame(g_rows).sort_values("val_auc", ascending=False)
    g.to_csv(f"{REPORTS_DIR}/j5_scrum_tuning_grid.csv", index=False)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b.learning_rate), "max_depth": int(b.max_depth), "n_estimators": max(int(b.trees), 10)}
    P_full, models = [], []
    for s in SEEDS:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(tv[FULL], tv[LABEL])
        P_full.append(m.predict_proba(te[FULL])[:, 1])
        models.append(m)
        joblib.dump(m, f"{MODELS_DIR}/xgb_scrum_spillover_seed{s}.joblib")
    # thresholds: match historical spill share per progress milestone (train+val in-sample)
    p_tv = np.mean([m.predict_proba(tv[FULL])[:, 1] for m in models], axis=0)
    thr = {str(p): float(np.quantile(p_tv[tv.progress.to_numpy() == p], 1 - tv.loc[tv.progress == p, LABEL].mean()))
           for p in PROGRESS}
    json.dump({"features": FULL, "config": cfg, "seeds": SEEDS, "label": "not resolved by sprint end + 1 day",
               "prior_match_thresholds_by_progress": thr},
              open(f"{MODELS_DIR}/xgb_scrum_spillover_meta.json", "w"), indent=2)

    panel, eff = [], []
    for sc in ["all"] + PROGRESS:
        msk = np.ones(len(y), bool) if sc == "all" else prog == sc
        for name, P in [("team_history_baseline", P_base), ("scrum_xgb", P_full)]:
            panel.append(seed_metric_panel(y[msk], [p[msk] for p in P], model=name, progress=sc))
        bs = paired_bootstrap_seeds(y[msk], [p[msk] for p in P_full], [p[msk] for p in P_base], n_boot=1000)
        eff.append({"progress": sc, **bs, "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
        # accuracy at prior-matched threshold
        if sc != "all":
            pf = np.mean([p[msk] for p in P_full], axis=0)
            yh = (pf >= thr[str(sc)]).astype(int)
            panel[-1]["acc_prior_thr"] = (yh == y[msk]).mean()
            from sklearn.metrics import balanced_accuracy_score
            panel[-1]["bal_acc_prior_thr"] = balanced_accuracy_score(y[msk], yh)
    panel = pd.DataFrame(panel)
    panel.to_csv(f"{REPORTS_DIR}/j5_scrum_metrics.csv", index=False)
    pd.DataFrame(eff).to_csv(f"{REPORTS_DIR}/j5_scrum_vs_baseline.csv", index=False)
    print("\nScrum mode, test:")
    print(panel[["model", "progress", "N", "pos_rate", "auc", "accuracy", "majority_accuracy", "balanced_accuracy",
                 "f1", "brier"] + [c for c in ["acc_prior_thr", "bal_acc_prior_thr"] if c in panel]].round(4).to_string(index=False))
    print("\nXGB minus team-history baseline:")
    print(pd.DataFrame(eff)[["progress", "d_auc", "d_auc_lo", "d_auc_hi"]].round(4).to_string(index=False))
    imp = pd.Series(models[0].get_booster().get_score(importance_type="gain"))
    (imp / imp.sum()).sort_values(ascending=False).to_csv(f"{REPORTS_DIR}/j5_scrum_gain_importance.csv")
    print("\nTop features:\n" + (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_string())
    print(f"\nselected config {cfg}; thresholds {thr}")


if __name__ == "__main__":
    main()
