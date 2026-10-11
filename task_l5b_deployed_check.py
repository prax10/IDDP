"""
Task L5b: do the L5 calibration constants still hold for the models the APP uses?

L5 calibrated on VAL with train-only models (the clean protocol). The app's deployed
models are refit on train+val, so the same constants are applied to slightly different
scores. This script measures test coverage exactly as the app computes it:
  1. flow: deployed live v5b ensemble + L5 Mondrian thresholds
  2. scrum: deployed spillover ensemble + L5 residual quantiles, with the app's
     floor/ceil rounding and clipping to [0, n_open]
(The CQR "how long" interval in the app uses the train-only CQR model itself, so its
L5 test coverage applies unchanged.)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from h_common import LABEL, REPORTS_DIR, MODELS_DIR
from task_j4_v5_model import load_all


LIVE = os.environ.get("LIVE", "v5b")


def main():
    conf = json.load(open(f"{MODELS_DIR}/conformal_calibration" + ("" if LIVE == "v5b" else f"_{LIVE}") + ".json"))
    rows = []

    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_{LIVE}_meta.json"))
    te = load_all()
    te = te[te.split == "test"].reset_index(drop=True)
    models = [joblib.load(f"{MODELS_DIR}/xgb_dynamic_live_{LIVE}_seed{s}.joblib") for s in meta["seeds"]]
    p = np.mean([m.predict_proba(te[meta["features"]])[:, 1] for m in models], 0)
    y = te[LABEL].to_numpy()
    c = conf["flow_mondrian"]
    inc1, inc0 = (1 - p) <= c["q_late"], p <= c["q_on_time"]
    cov = np.where(y == 1, inc1, inc0)
    single = inc1 ^ inc0
    ck = te["checkpoint_index"].to_numpy()
    for sc in ["all", 1, 3, 5, 8, 10]:
        m = np.ones(len(y), bool) if sc == "all" else ck == sc
        rows.append({"part": "flow sets, deployed model", "scope": sc if sc == "all" else f"M{sc}",
                     "coverage": cov[m].mean(), "coverage_late": cov[m & (y == 1)].mean(),
                     "coverage_on_time": cov[m & (y == 0)].mean(), "confident_share": single[m].mean(),
                     "confident_accuracy": (np.where(inc1 & ~inc0, 1, 0)[m & single] == y[m & single]).mean(),
                     "confident_late_precision": y[m & inc1 & ~inc0].mean()})
    del te

    smeta = json.load(open(f"{MODELS_DIR}/xgb_scrum_spillover_meta.json"))
    rws = pd.read_csv("data/features/scrum_rows.csv.gz")
    rws = rws[rws.split == "test"].copy()
    cats = smeta.get("categories", {})
    for col in ["Priority", "Type", "status_bucket"]:
        rws[col] = pd.Categorical(rws[col].astype(str), categories=cats.get(col)) if col in cats else rws[col].astype(str).astype("category")
    rws["Project_cat"] = pd.Categorical(rws["Project_ID"].astype(int), categories=cats.get("Project_cat")) \
        if "Project_cat" in cats else rws["Project_ID"].astype(int).astype("category")
    sm = [joblib.load(f"{MODELS_DIR}/xgb_scrum_spillover_seed{s}.joblib") for s in smeta["seeds"]]
    rws["p"] = np.mean([m.predict_proba(rws[smeta["features"]])[:, 1] for m in sm], 0)
    g = rws.groupby(["sprint_key", "progress"]).agg(E=("p", "sum"), A=("spilled", "sum"), n=("p", "size")).reset_index()
    g = g[g.n >= 3]
    for prog, t in g.groupby("progress"):
        q = conf["scrum_sprint_count"][min(conf["scrum_sprint_count"], key=lambda k: abs(float(k) - prog))]
        lo = np.clip(np.floor(t.E + q["residual_q05"]), 0, t.n)
        hi = np.clip(np.ceil(t.E + q["residual_q95"]), 0, t.n)
        rows.append({"part": "scrum spill-count range, deployed model (as shown in app)", "scope": f"progress {prog}",
                     "coverage": ((t.A >= lo) & (t.A <= hi)).mean(), "median_width_issues": float(np.median(hi - lo)),
                     "n_test_sprints": len(t), "MAE_point": float((t.A - t.E).abs().mean())})
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l5b_deployed_coverage" + ("" if LIVE == "v5b" else f"_{LIVE}") + ".csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
