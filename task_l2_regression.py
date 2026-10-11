"""
Task L2 (future-work #8): REGRESSION -- predict how late, not just late / not late.

Target (per checkpoint row):
    rrd = log2( Resolution_Time / project median )      # relative resolution duration
          (median from train+val, the same threshold as the binary label: rrd > 0 <=> delayed_v2 = 1)
  -> predicted total duration  = median * 2^rrd_hat
  -> predicted remaining days  = total - elapsed (floored at 0)
This is the issue-level analogue of Kula et al.'s continuous BRE target, so it is
evaluated with their metrics:
    MAE and Standardised Accuracy  SA = (1 - MAE / MAE_random) x 100
    (MAE_random: predict by drawing a random TRAINING target, 1,000 draws -- Shepperd & MacDonell)
plus MAE in days on total and remaining duration, median absolute error in days,
and the sign agreement with the binary label (AUC of rrd_hat vs delayed_v2).
Uncertainty: XGBoost multi-quantile regression (q = 0.1, 0.5, 0.9) -> 80% interval
coverage on test.
Models: v5b live features, recency weights, config chosen on val (3 configs), 3 seeds.
Baselines: "project median" (rrd = 0) and a static (creation-time) regressor.
"""

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded, static_cols, TUNED_STATIC
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

SEEDS3 = [0, 1, 2]
GRID = [{"learning_rate": 0.1, "max_depth": 6}, {"learning_rate": 0.1, "max_depth": 8},
        {"learning_rate": 0.05, "max_depth": 8}]
REG = dict(subsample=0.8, colsample_bytree=0.8, enable_categorical=True, tree_method="hist")


def sa(y, pred, y_pool, rng, draws=1000):
    mae = np.abs(y - pred).mean()
    mae_rand = np.mean([np.abs(y - rng.choice(y_pool, size=len(y))).mean() for _ in range(draws // 10)])
    return mae, 100 * (1 - mae / mae_rand)


def main():
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    cols = meta["features"]
    ck = load_all()
    f = pd.read_csv("data/features/phase6_feature_table_v3.csv", usecols=["ID", "Project_ID", "Resolution_Time_Minutes"])
    sp = pd.read_csv("data/features/phase8_split.csv", usecols=["ID", "split"])
    f = f.merge(sp, on="ID")
    med = f[f.split.isin(["train", "val"])].groupby("Project_ID")["Resolution_Time_Minutes"].median()
    if "Resolution_Time_Minutes" not in ck.columns:   # load_all() already carries it from the v3 feature table
        ck = ck.merge(f[["ID", "Resolution_Time_Minutes"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    ck["proj_median_min"] = ck["Project_ID"].astype(int).map(med).astype(float)
    ck["rrd"] = np.log2(np.maximum(ck["Resolution_Time_Minutes"], 1) / ck["proj_median_min"])
    assert ((ck["rrd"] > 0).astype(int) == ck[LABEL]).mean() > 0.999

    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    w_tr, w_tv = weights(tr, 4.0, tr["age_years"].min()), weights(tv, 4.0, tv["age_years"].min())

    best = None
    for cfg in GRID:
        m = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=40, objective="reg:absoluteerror",
                             **cfg, **REG, random_state=0)
        m.fit(tr[cols], tr["rrd"], sample_weight=w_tr, eval_set=[(va[cols], va["rrd"])], verbose=False)
        mae = np.abs(m.predict(va[cols]) - va["rrd"]).mean()
        print(f"  {cfg}: val MAE {mae:.4f} (log2 units), trees {m.best_iteration + 1}", flush=True)
        if best is None or mae < best[0]:
            best = (mae, {**cfg, "n_estimators": m.best_iteration + 1})
    cfg = best[1]
    P = []
    for s in SEEDS3:
        m = xgb.XGBRegressor(objective="reg:absoluteerror", **cfg, **REG, random_state=s)
        m.fit(tv[cols], tv["rrd"], sample_weight=w_tv)
        P.append(m.predict(te[cols]))
        joblib.dump(m, f"{MODELS_DIR}/xgb_regression_rrd_v5b_seed{s}.joblib")
    pred = np.mean(P, axis=0)

    # quantile model for intervals
    q = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.5, 0.9]),
                         **cfg, **REG, random_state=0)
    q.fit(tv[cols], tv["rrd"], sample_weight=w_tv)
    Q = np.sort(q.predict(te[cols]), axis=1)
    joblib.dump(q, f"{MODELS_DIR}/xgb_regression_rrd_v5b_quantiles.joblib")

    # static regressor baseline (creation-time features)
    st = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=300, learning_rate=0.05, max_depth=6,
                          **REG, random_state=0)
    st.fit(tv[static_cols(30)], tv["rrd"])
    pred_static = st.predict(te[static_cols(30)])

    rng = np.random.default_rng(0)
    y = te["rrd"].to_numpy()
    c = te["checkpoint_index"].to_numpy()
    pool = tr["rrd"].to_numpy()
    medd = te["proj_median_min"].to_numpy() / 1440
    elapsed = te["elapsed_minutes"].astype(float).to_numpy() / 1440
    true_total = te["Resolution_Time_Minutes"].to_numpy() / 1440
    rows = []
    for sc in ["all"] + [f"M{t}" for t in range(1, 11)]:
        msk = np.ones(len(y), bool) if sc == "all" else c == int(sc[1:])
        r = {"scope": sc, "N": int(msk.sum())}
        for name, p in [("v5b_regression", pred), ("static_regression", pred_static), ("project_median", np.zeros_like(y))]:
            mae, s = sa(y[msk], p[msk], pool, rng)
            r[f"{name}_MAE_log2"] = mae
            r[f"{name}_SA"] = s
        tot_hat = np.maximum(medd * 2 ** pred, elapsed)          # can't finish before now
        rem_hat, rem = np.maximum(tot_hat - elapsed, 0), np.maximum(true_total - elapsed, 0)
        r["MAE_total_days"] = np.abs(tot_hat - true_total)[msk].mean()
        r["MedAE_total_days"] = np.median(np.abs(tot_hat - true_total)[msk])
        r["MedAE_remaining_days"] = np.median(np.abs(rem_hat - rem)[msk])
        r["median_true_remaining_days"] = np.median(rem[msk])
        r["AUC_sign_vs_label"] = roc_auc_score(te[LABEL].to_numpy()[msk], pred[msk])
        r["coverage_80pct_interval"] = ((y[msk] >= Q[msk, 0]) & (y[msk] <= Q[msk, 2])).mean()
        r["median_interval_width_log2"] = np.median(Q[msk, 2] - Q[msk, 0])
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l2_regression_metrics.csv", index=False)
    json.dump({"features": cols, "config": cfg, "target": "log2(Resolution_Time / train+val project median)",
               "seeds": SEEDS3, "quantile_model": "xgb_regression_rrd_v5b_quantiles.joblib (q=0.1,0.5,0.9)"},
              open(f"{MODELS_DIR}/xgb_regression_rrd_v5b_meta.json", "w"), indent=2)
    print(f"\nselected {cfg}")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
