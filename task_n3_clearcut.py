"""
Task N3: secondary "clear-cut delay" task. NOT a replacement for the main label.

Label: an issue is CLEARLY LATE if resolution time > 1.5 x its project's train+val median, CLEARLY ON TIME if
< median / 1.5. Issues in between are dropped (from training and from evaluation). Same features, splits,
recency weights and v7 config as the main model; 5 seeds; test scored once.

Honesty check reported alongside: an open row whose elapsed time already exceeds median/1.5 can only belong
to a clearly-late issue (a clearly-on-time issue would have been resolved by then), so those rows are decided
by construction. AUC is therefore also reported on UNDECIDED rows (elapsed < median/1.5).
"""
import sys, json
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from h_common import REPORTS_DIR, MODELS_DIR, seeded
from task_i2_model_upgrades import weights
import task_n2_v7_model as n2

R = 1.5


def main(variant):
    cols = n2.VARIANTS[variant]
    cfg, hl = n2.CFG, n2.HL
    if len(sys.argv) > 2:
        j = json.load(open(sys.argv[2]))
        cfg, hl = j["config"], j["half_life"]
    ck = n2.data()
    iss = pd.read_csv("data/raw/issue_slim.csv.gz", usecols=["ID", "Project_ID", "Creation_Date", "Resolution_Date"],
                      parse_dates=["Creation_Date", "Resolution_Date"], date_format="ISO8601")
    sp = pd.read_csv("data/features/phase8_split.csv", usecols=["ID", "split"])
    r = iss.merge(sp, on="ID")
    r["res_min"] = (r["Resolution_Date"] - r["Creation_Date"]).dt.total_seconds() / 60
    med = r[r.split.isin(["train", "val"])].groupby("Project_ID")["res_min"].median()
    r["ratio"] = r["res_min"] / r["Project_ID"].map(med)
    r["clear"] = np.where(r["ratio"] > R, 1, np.where(r["ratio"] < 1 / R, 0, -1))
    ck = ck.merge(r[["ID", "clear"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    # sanity: clear label agrees with the main label where defined
    d = ck[ck.clear >= 0]
    print(f"agreement with main label on clear rows: {(d['clear'] == d['delayed_v2']).mean():.4f}", flush=True)
    ck = ck[ck.clear >= 0].reset_index(drop=True)
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    yt = te["clear"].to_numpy()
    undecided = (te["log_elapsed_over_median"].to_numpy() < np.log(1 / R))
    P = []
    for s in range(5):
        m = xgb.XGBClassifier(**{**seeded(s), **cfg, "n_jobs": 2})
        m.fit(tv[cols], tv["clear"], sample_weight=weights(tv, hl, tv["age_years"].min()))
        P.append(m.predict_proba(te[cols])[:, 1])
        print(f"  seed {s} done", flush=True)
    p = np.mean(P, 0)
    c = te["checkpoint_index"].to_numpy()
    rows = []
    for scope, msk in [("all clear-cut rows", np.ones(len(te), bool)), ("undecided rows", undecided)]:
        row = {"scope": scope, "rows": int(msk.sum()), "late_share": yt[msk].mean(),
               "auc_seed_mean": np.mean([roc_auc_score(yt[msk], q[msk]) for q in P]),
               "auc_ensemble": roc_auc_score(yt[msk], p[msk])}
        for k in (1, 3, 5, 8, 10):
            mk = msk & (c == k)
            row[f"M{k}"] = roc_auc_score(yt[mk], p[mk]) if mk.sum() > 50 and 0 < yt[mk].mean() < 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, "variant", variant)
    out.to_csv(f"{REPORTS_DIR}/n3_clearcut_{variant}.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"test issues kept: {te.Issue_ID.nunique()} of {ck.Issue_ID.nunique()} clear-cut issues overall")


if __name__ == "__main__":
    main(sys.argv[1])
