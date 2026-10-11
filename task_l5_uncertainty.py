"""
Task L5: uncertainty with coverage guarantees (conformal prediction).

Protocol for every part: models are trained on TRAIN only, calibrated on VAL,
and coverage is measured on TEST (the most recent period, so drift is included).

1. Flow mode, late / on time -- Mondrian (class-conditional) conformal sets, alpha = 0.10
     nonconformity s = 1 - p(true class); a class enters the set if 1 - p_class <= q_class
     -> set {late}, {on time} (a confident call) or {late, on time} (= "uncertain")
     reported: coverage overall / per class / per checkpoint, share of confident calls,
     accuracy of confident calls
2. "How late" -- conformalised quantile regression (CQR) on rrd = log2(duration / project median)
     80% and 90% intervals; coverage, median width (log2 and days remaining)
3. Scrum mode, spills per sprint -- split-conformal on (actual - expected) spill counts of VAL
     sprints, per sprint-progress milestone; 90% intervals; test coverage. A naive
     Poisson-binomial interval (treating issues as independent) is reported for comparison.
Calibration constants are saved to models/conformal_calibration.json for the app.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

ALPHA = 0.10
LIVE = os.environ.get("LIVE", "v5b")
REG = dict(subsample=0.8, colsample_bytree=0.8, enable_categorical=True, tree_method="hist")


def conformal_q(scores, alpha):
    n = len(scores)
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def main():
    out, rows = {}, []
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_{LIVE}_meta.json"))
    cols = meta["features"]
    cfg = dict(meta["config"])
    ck = load_all()
    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    te = ck[ck.split == "test"].reset_index(drop=True)
    w = weights(tr, 4.0, tr["age_years"].min())

    # ---------------- 1. Mondrian conformal classification
    pv, pt = [], []
    for s in [0, 1, 2]:
        m = xgb.XGBClassifier(**{**seeded(s), **cfg})
        m.fit(tr[cols], tr[LABEL], sample_weight=w)
        pv.append(m.predict_proba(va[cols])[:, 1])
        pt.append(m.predict_proba(te[cols])[:, 1])
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    yv, yt = va[LABEL].to_numpy(), te[LABEL].to_numpy()
    q1 = conformal_q(1 - pv[yv == 1], ALPHA)        # late
    q0 = conformal_q(pv[yv == 0], ALPHA)            # on time: 1 - p(on time) = p(late)
    inc1 = (1 - pt) <= q1
    inc0 = pt <= q0
    covered = np.where(yt == 1, inc1, inc0)
    single = inc1 ^ inc0
    pred_single = np.where(inc1 & ~inc0, 1, 0)
    c = te["checkpoint_index"].to_numpy()
    for sc in ["all"] + [f"M{t}" for t in (1, 3, 5, 8, 10)]:
        msk = np.ones(len(yt), bool) if sc == "all" else c == int(sc[1:])
        rows.append({"part": "flow conformal sets (90%)", "scope": sc,
                     "coverage": covered[msk].mean(),
                     "coverage_late": covered[msk & (yt == 1)].mean(),
                     "coverage_on_time": covered[msk & (yt == 0)].mean(),
                     "confident_share": single[msk].mean(),
                     "confident_accuracy": (pred_single[msk & single] == yt[msk & single]).mean(),
                     "empty_share": (~inc1 & ~inc0)[msk].mean()})
    out["flow_mondrian"] = {"alpha": ALPHA, "q_late": float(q1), "q_on_time": float(q0),
                            "rule": "late in set if 1-p<=q_late; on time in set if p<=q_on_time"}
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

    # ---------------- 2. CQR for rrd
    f = pd.read_csv("data/features/phase6_feature_table_v3.csv", usecols=["ID", "Project_ID", "Resolution_Time_Minutes"])
    sp = pd.read_csv("data/features/phase8_split.csv", usecols=["ID", "split"])
    f = f.merge(sp, on="ID")
    med = f[f.split.isin(["train", "val"])].groupby("Project_ID")["Resolution_Time_Minutes"].median()
    for d in (tr, va, te):
        d["rrd"] = np.log2(np.maximum(d["Resolution_Time_Minutes"].astype(float), 1) /
                           d["Project_ID"].astype(int).map(med).astype(float))
    rcfg = dict(learning_rate=0.1, max_depth=6, n_estimators=268)
    for level in (0.8, 0.9):
        a = (1 - level) / 2
        qm = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([a, 1 - a]), **rcfg, **REG,
                              random_state=0)
        qm.fit(tr[cols], tr["rrd"], sample_weight=w)
        Qv = np.sort(qm.predict(va[cols]), axis=1)
        Qt = np.sort(qm.predict(te[cols]), axis=1)
        E = np.maximum(Qv[:, 0] - va["rrd"].to_numpy(), va["rrd"].to_numpy() - Qv[:, 1])
        qE = conformal_q(E, 1 - level)
        lo, hi = Qt[:, 0] - qE, Qt[:, 1] + qE
        y = te["rrd"].to_numpy()
        raw_cov = ((y >= Qt[:, 0]) & (y <= Qt[:, 1])).mean()
        cov = ((y >= lo) & (y <= hi)).mean()
        medd = te["Project_ID"].astype(int).map(med).astype(float).to_numpy() / 1440
        el = te["elapsed_minutes"].astype(float).to_numpy() / 1440
        rem_lo = np.maximum(medd * 2 ** lo - el, 0)
        rem_hi = np.maximum(medd * 2 ** hi - el, 0)
        rows.append({"part": f"how-late interval ({int(level*100)}%, CQR)", "scope": "all",
                     "coverage": cov, "coverage_before_conformal": raw_cov,
                     "median_width_log2": np.median(hi - lo),
                     "median_remaining_days_interval": f"{np.median(rem_lo):.1f}-{np.median(rem_hi):.1f}"})
        out[f"cqr_{int(level*100)}"] = {"quantiles": [a, 1 - a], "adjustment_log2": float(qE)}
        joblib.dump(qm, f"{MODELS_DIR}/xgb_regression_rrd_cqr{int(level*100)}_trainonly.joblib")
        print(rows[-1], flush=True)

    # ---------------- 3. Scrum sprint-level spill counts
    smeta = json.load(open(f"{MODELS_DIR}/xgb_scrum_spillover_meta.json"))
    rws = pd.read_csv("data/features/scrum_rows.csv.gz")
    for col in ["Priority", "Type", "status_bucket"]:
        rws[col] = rws[col].astype(str).astype("category")
    rws["Project_cat"] = rws["Project_ID"].astype(int).astype("category")
    str_, sva, ste = rws[rws.split == "train"], rws[rws.split == "val"], rws[rws.split == "test"]
    ps_v, ps_t = [], []
    for s in [0, 1, 2]:
        m = xgb.XGBClassifier(**smeta["config"], **seeded(s))
        m.fit(str_[smeta["features"]], str_["spilled"])
        ps_v.append(m.predict_proba(sva[smeta["features"]])[:, 1])
        ps_t.append(m.predict_proba(ste[smeta["features"]])[:, 1])
    sva = sva.assign(p=np.mean(ps_v, 0))
    ste = ste.assign(p=np.mean(ps_t, 0))
    agg = lambda d: d.groupby(["sprint_key", "progress"]).agg(E=("p", "sum"), V=("p", lambda x: (x * (1 - x)).sum()),
                                                              A=("spilled", "sum"), n=("p", "size")).reset_index()
    gv, gt = agg(sva), agg(ste)
    gv, gt = gv[gv.n >= 3], gt[gt.n >= 3]
    out["scrum_sprint_count"] = {}
    for prog in sorted(gt.progress.unique()):
        v, t = gv[gv.progress == prog], gt[gt.progress == prog]
        r = v["A"] - v["E"]
        lo_q, hi_q = np.quantile(r, 0.05), np.quantile(r, 0.95)
        cov = ((t["A"] >= t["E"] + lo_q) & (t["A"] <= t["E"] + hi_q)).mean()
        z = norm.ppf(0.95)
        pb_cov = ((t["A"] >= t["E"] - z * np.sqrt(t["V"])) & (t["A"] <= t["E"] + z * np.sqrt(t["V"]))).mean()
        rows.append({"part": "scrum spills per sprint (90%)", "scope": f"progress {prog}", "coverage": cov,
                     "coverage_poisson_binomial": pb_cov, "median_width_issues": hi_q - lo_q,
                     "n_test_sprints": len(t)})
        out["scrum_sprint_count"][str(prog)] = {"residual_q05": float(lo_q), "residual_q95": float(hi_q)}
        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l5_conformal_coverage" + ("" if LIVE == "v5b" else f"_{LIVE}") + ".csv", index=False)
    json.dump(out, open(f"{MODELS_DIR}/conformal_calibration" + ("" if LIVE == "v5b" else f"_{LIVE}") + ".json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
