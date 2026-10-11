"""
Task J3 (future-work #4): time-respecting cross-project generalisation.

Two asymmetries in Phase 9 (task_b3_phase9.py):
  (1) cross-project trained on other projects across ALL time, including issues
      created after the held-out project's test issues;
  (2) cross-project was scored on ALL of the held-out project's issues while
      within-project was scored on its last 20% only -- different test sets.
Fix: for every project p, ONE test set = p's chronologically last 20% of issues.
  cutoff_p = creation time of p's first test issue
  within   : train on p's own first 80%                       (as before)
  cross    : train on OTHER projects' issues created < cutoff_p   (no future, no p)
  pooled   : train on ALL projects' issues created < cutoff_p     (realistic deployment)
Run for the static model (Phase 7 features) and the v4 dynamic model
(checkpoint rows, 43 + 18 activity features). Static: tuned config, 2 seeds; dynamic: 1 seed
(runtime). Phase 9 used a 400-tree static config; the tuned 42-tree config is used here for
speed and is the paper's selected static model.
Wilcoxon signed-rank on paired per-project AUCs.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from h_common import LABEL, REPORTS_DIR, static_cols, seeded, load_static
from task_i2_model_upgrades import load, BASE, ACT

STATIC_CFG = dict(n_estimators=42, max_depth=4, learning_rate=0.2)    # the paper's tuned static config (B4)
DYN_CFG = dict(n_estimators=32, max_depth=8, learning_rate=0.05)       # v4 config
SEEDS2 = [0, 1]   # static
SEEDS1 = [0]      # dynamic (1M-row fits x 4 arms x 39 projects)


def fit_pred(cfg, tr, te, cols, seeds=SEEDS2):
    ps = []
    for s in seeds:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(tr[cols], tr[LABEL])
        ps.append(m.predict_proba(te[cols])[:, 1])
    return np.mean(ps, axis=0)


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def run(df, issue_level, cols, cfg, tag, seeds=SEEDS2):
    rows = []
    projects = sorted(df["Project_ID"].astype(int).unique())
    iss = issue_level.sort_values("Creation_Date", kind="mergesort")
    for i, p in enumerate(projects, 1):
        pi = iss[iss["Project_ID"].astype(int) == p]
        n = len(pi)
        n_tr = int(round(n * 0.8))
        test_ids = set(pi["ID"].iloc[n_tr:])
        own_train_ids = set(pi["ID"].iloc[:n_tr])
        cutoff = pi["Creation_Date"].iloc[n_tr] if n_tr < n else pd.NaT
        is_p = df["Project_ID"].astype(int) == p
        te = df[df["ID"].isin(test_ids)]
        r = {"Project_ID": p, "n_issues": n, "n_test_rows": len(te), "test_delay_rate": te[LABEL].mean() if len(te) else np.nan}
        if len(te) < 30 or te[LABEL].nunique() < 2:
            r["skipped"] = "test too small / one class"
            rows.append(r)
            continue
        before = df["Creation_Date"] < cutoff
        arms = {
            "within": df[df["ID"].isin(own_train_ids)],
            "cross_time_respecting": df[(~is_p) & before],
            "pooled_time_respecting": df[before],
            "cross_all_time": df[~is_p],          # the original Phase 9 setup, same test set
        }
        for name, tr in arms.items():
            if len(tr) < 50 or tr[LABEL].nunique() < 2:
                r[f"{name}_auc"] = np.nan
                continue
            r[f"{name}_auc"] = auc(te[LABEL].to_numpy(), fit_pred(cfg, tr, te, cols, seeds))
            r[f"{name}_n_train"] = len(tr)
        rows.append(r)
        print(f"  [{tag} {i:2d}/{len(projects)}] P{p}: " + " ".join(
            f"{k.replace('_auc','')}={v:.3f}" for k, v in r.items() if k.endswith("_auc") and pd.notna(v)), flush=True)
    return pd.DataFrame(rows)


def summarise(res, tag, out):
    lines = [f"=== {tag} ==="]
    arms = ["within", "cross_time_respecting", "pooled_time_respecting", "cross_all_time"]
    for a in arms:
        s = res[f"{a}_auc"].dropna()
        lines.append(f"  {a:24s} mean={s.mean():.4f} median={s.median():.4f} n={len(s)}")
    for a, b in [("cross_time_respecting", "within"), ("pooled_time_respecting", "within"),
                 ("cross_all_time", "within"), ("cross_all_time", "cross_time_respecting")]:
        d = res[[f"{a}_auc", f"{b}_auc"]].dropna()
        wins = (d[f"{a}_auc"] > d[f"{b}_auc"]).sum()
        try:
            pval = wilcoxon(d[f"{a}_auc"], d[f"{b}_auc"]).pvalue
        except ValueError:
            pval = np.nan
        lines.append(f"  {a} beats {b}: {wins}/{len(d)} projects; mean diff "
                     f"{(d[f'{a}_auc'] - d[f'{b}_auc']).mean():+.4f}; Wilcoxon p={pval:.4g}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    out.append(txt)


def main():
    out = []
    split = pd.read_csv("data/features/phase8_split.csv", parse_dates=["Creation_Date"])
    st = load_static().merge(split[["ID", "Creation_Date"]], on="ID")
    res_s = run(st, st[["ID", "Project_ID", "Creation_Date"]], static_cols(30), STATIC_CFG, "static")
    res_s.to_csv(f"{REPORTS_DIR}/j3_lopo_static.csv", index=False)
    summarise(res_s, "STATIC model (one test set per project = its last 20%)", out)

    ck = load()
    ck["ID"] = ck["Issue_ID"]
    res_d = run(ck, st[["ID", "Project_ID", "Creation_Date"]], BASE + ACT, DYN_CFG, "dyn-v4", seeds=SEEDS1)
    res_d.to_csv(f"{REPORTS_DIR}/j3_lopo_dynamic_v4.csv", index=False)
    summarise(res_d, "DYNAMIC v4 model (checkpoint rows of the same test issues)", out)
    open(f"{REPORTS_DIR}/j3_summary.txt", "w").write("\n\n".join(out))


if __name__ == "__main__":
    main()
