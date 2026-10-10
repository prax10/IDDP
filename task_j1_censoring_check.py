"""
Task J1: is the late-checkpoint advantage (and the Task H "+0.13 from adding
val data" effect) an artifact of RIGHT-CENSORING at the TAWOS snapshot end?

Mechanism under test: only RESOLVED issues are modelled. An issue created
shortly before a project's snapshot end can only appear in the data if it
resolved quickly -- long-running recent issues are missing. The test split
is the most recent 20%, so it is the most exposed.

Definitions
  project_end   = latest Creation/Resolution timestamp in that project (all 458k issues)
  horizon_p90   = 90th pct resolution time of that project's TRAIN issues
  safe          = issue created >= horizon_p90 before project_end
                  (an issue this old had time to resolve even if slow)
  at-risk       = not safe

Experiments (3 seeds each, configs from Task I)
  1. how exposed is each split; delay rate safe vs at-risk
  2. test AUC on all / safe / at-risk rows for: static, baseline dynamic, v4 dynamic
  3. Task H data effect (train+val minus train-only) on safe vs at-risk test rows
  4. v4 trained on SAFE rows only (train+val), scored on safe test rows
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import (LABEL, REPORTS_DIR, TUNED_STATIC, static_cols, seeded, load_static,
                      paired_bootstrap_seeds, verdict)
from task_i2_model_upgrades import load, BASE, ACT, weights

SEEDS3 = [0, 1, 2]
CFG_A = dict(learning_rate=0.05, max_depth=8, n_estimators=29)
CFG_B = dict(learning_rate=0.05, max_depth=8, n_estimators=32)
HL = 4.0


def safe_flags():
    iss = pd.read_csv("data/raw/issue_slim.csv.gz", usecols=["ID", "Project_ID", "Creation_Date", "Resolution_Date"],
                      parse_dates=["Creation_Date", "Resolution_Date"], date_format="ISO8601")
    iss["end"] = iss.groupby("Project_ID")[["Creation_Date", "Resolution_Date"]].transform("max").max(axis=1)
    sp = pd.read_csv("data/features/phase8_split.csv")
    r = iss.merge(sp[["ID", "split"]], on="ID")
    r["res_days"] = (r["Resolution_Date"] - r["Creation_Date"]).dt.total_seconds() / 86400
    p90 = r[r["split"] == "train"].groupby("Project_ID")["res_days"].quantile(0.9)
    r["horizon_days"] = r["Project_ID"].map(p90)
    r["days_to_end"] = (r["end"] - r["Creation_Date"]).dt.total_seconds() / 86400
    r["safe"] = r["days_to_end"] >= r["horizon_days"]
    return r[["ID", "safe", "days_to_end", "horizon_days"]]


def fit(cfg, df, cols, X, w=None):
    out = []
    for s in SEEDS3:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(df[cols], df[LABEL], sample_weight=w)
        out.append(m.predict_proba(X[cols])[:, 1])
    return out


def auc_seedmean(y, P):
    return float(np.mean([roc_auc_score(y, p) for p in P]))


def main():
    sf = safe_flags()
    ck = load().merge(sf.rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    st = load_static().merge(sf, on="ID", how="left")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- 1. exposure
    log("=== 1. Exposure to censoring (issue level) ===")
    expo = st.groupby("split").agg(n=("ID", "size"), safe_share=("safe", "mean"))
    expo["delay_rate_safe"] = st[st.safe].groupby("split")[LABEL].mean()
    expo["delay_rate_atrisk"] = st[~st.safe].groupby("split")[LABEL].mean()
    log(expo.round(3).to_string())
    expo.to_csv(f"{REPORTS_DIR}/j1_exposure.csv")

    train = ck[ck.split == "train"]
    tv = ck[ck.split.isin(["train", "val"])]
    test = ck[ck.split == "test"].reset_index(drop=True)
    y = test[LABEL].to_numpy()
    c = test["checkpoint_index"].to_numpy()
    safe = test["safe"].to_numpy(bool)

    # ---- 2. models trained normally
    log("\n=== 2. Fitting models (3 seeds each) ===")
    st_tv = st[st.split.isin(["train", "val"])]
    P = {
        "static|tv": fit(TUNED_STATIC, st_tv, static_cols(30), test),
        "baseline|tv": fit(CFG_A, tv, BASE, test),
        "baseline|train": fit(CFG_A, train, BASE, test),
        "v4|tv": fit(CFG_B, tv, BASE + ACT, test, w=weights(tv, HL, tv["age_years"].min())),
        "v4|train": fit(CFG_B, train, BASE + ACT, test, w=weights(train, HL, train["age_years"].min())),
    }
    tv_safe = tv[tv["safe"]]
    P["v4|tv_safe_only"] = fit(CFG_B, tv_safe, BASE + ACT, test, w=weights(tv_safe, HL, tv_safe["age_years"].min()))

    rows = []
    for name, PP in P.items():
        for sub, mask in [("all", np.ones(len(y), bool)), ("safe", safe), ("at_risk", ~safe)]:
            for sc in ["all"] + [f"M{t}" for t in (1, 3, 5, 10)]:
                m = mask if sc == "all" else mask & (c == int(sc[1:]))
                if m.sum() < 200 or len(np.unique(y[m])) < 2:
                    continue
                rows.append({"model": name, "test_subset": sub, "scope": sc, "N": int(m.sum()),
                             "pos_rate": y[m].mean(), "auc": auc_seedmean(y[m], [p[m] for p in PP])})
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{REPORTS_DIR}/j1_auc_by_subset.csv", index=False)
    wide = tab.pivot_table(index=["test_subset", "scope"], columns="model", values="auc")
    log("\n=== 2. Test AUC by censoring subset ===")
    log(wide.round(4).to_string())
    ns = tab.drop_duplicates(["test_subset", "scope"])[["test_subset", "scope", "N", "pos_rate"]]
    log("\nrows / delay rate per subset:\n" + ns.round(3).to_string(index=False))

    # ---- 3. data effect and dynamic-vs-static, with CIs, on safe vs at-risk
    log("\n=== 3. Key contrasts with 95% CIs ===")
    eff = []
    contrasts = [
        ("v4 minus static", "v4|tv", "static|tv"),
        ("baseline-dynamic minus static", "baseline|tv", "static|tv"),
        ("val-data effect (baseline)", "baseline|tv", "baseline|train"),
        ("val-data effect (v4)", "v4|tv", "v4|train"),
        ("v4 minus baseline-dynamic", "v4|tv", "baseline|tv"),
    ]
    for sub, mask in [("all", np.ones(len(y), bool)), ("safe", safe), ("at_risk", ~safe)]:
        for sc in ["all", "M1", "M10"]:
            m = mask if sc == "all" else mask & (c == int(sc[1:]))
            if m.sum() < 200:
                continue
            for lab, a, b in contrasts:
                bs = paired_bootstrap_seeds(y[m], [p[m] for p in P[a]], [p[m] for p in P[b]], n_boot=500, seed=1)
                eff.append({"test_subset": sub, "scope": sc, "contrast": lab, **bs,
                            "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
    eff = pd.DataFrame(eff)
    eff.to_csv(f"{REPORTS_DIR}/j1_contrasts.csv", index=False)
    for _, r in eff.iterrows():
        log(f"  [{r['test_subset']:7s} {r['scope']:3s}] {r['contrast']:30s} {r['d_auc']:+.4f} "
            f"[{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]")

    with open(f"{REPORTS_DIR}/j1_run_summary.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
