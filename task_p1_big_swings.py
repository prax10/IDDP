"""
Task P1: big-swing learners on the v8 feature set, TRAIN -> VAL only (test untouched).
  xgb_v8          reference (v7/N4 config, 130 trees)
  xgb_slow        learning rate 0.02, up to 800 trees, best iteration on val reported
  lgbm            LightGBM, 255 leaves, lr 0.03, up to 800 rounds
  catboost        CatBoost, depth 8, lr 0.08, up to 1500 iterations, native categoricals
  avg(xgb, cat)   equal-weight blend
"""
import time, sys
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from h_common import LABEL, seeded
from task_i2_model_upgrades import weights
import task_n2_v7_model as n2, task_o2_v8_model as o2

COLS = o2.VARIANTS["v8"]
CAT = [c for c in COLS if c in ("Project_ID", "Priority_Normalized", "Type_Normalized", "status_bucket", "type_T")]

if __name__ == "__main__":
    which = sys.argv[1:] or ["xgb_v8", "xgb_slow", "lgbm", "catboost"]
    ck = o2.data()
    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    del ck
    y, yv = tr[LABEL].to_numpy(), va[LABEL].to_numpy()
    w = weights(tr, o2.HL, tr["age_years"].min())
    P, rows = {}, []

    def log(name, p, extra=""):
        a = roc_auc_score(yv, p)
        c = va["checkpoint_index"].to_numpy()
        rows.append({"model": name, "val_auc": a, "M1": roc_auc_score(yv[c == 1], p[c == 1]), "M10": roc_auc_score(yv[c == 10], p[c == 10]), "note": extra})
        print(f"{name:14s} val AUC {a:.4f}  {extra}", flush=True)
        pd.DataFrame(rows).to_csv("reports/p1_big_swings_val.csv", index=False)

    if "xgb_v8" in which:
        P["xgb"] = n2.fit(tr, COLS, 0, o2.CFG, o2.HL).predict_proba(va[COLS])[:, 1]
        log("xgb_v8", P["xgb"])
    if "xgb_slow" in which:
        t0 = time.time()
        m = xgb.XGBClassifier(**{**seeded(0), **o2.CFG, "learning_rate": 0.02, "n_estimators": 800, "eval_metric": "auc", "n_jobs": 2})
        m.fit(tr[COLS], y, sample_weight=w, eval_set=[(va[COLS], yv)], verbose=False)
        cur = np.array(m.evals_result()["validation_0"]["auc"])
        log("xgb_slow", m.predict_proba(va[COLS], iteration_range=(0, int(cur.argmax()) + 1))[:, 1],
            f"best {cur.argmax()+1} trees (optimistic: picked on val), {time.time()-t0:.0f}s")
    if "lgbm" in which:
        t0 = time.time()
        Xt, Xv = tr[COLS].copy(), va[COLS].copy()
        m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=255, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=2)
        m.fit(Xt, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc")
        cur = np.array(m.evals_result_["valid_0"]["auc"])
        P["lgbm"] = m.predict_proba(Xv, num_iteration=int(cur.argmax()) + 1)[:, 1]
        log("lgbm", P["lgbm"], f"best {cur.argmax()+1} rounds (optimistic), {time.time()-t0:.0f}s")
    if "catboost" in which:
        t0 = time.time()
        Xt, Xv = tr[COLS].copy(), va[COLS].copy()
        for c in CAT:
            Xt[c] = Xt[c].astype(str); Xv[c] = Xv[c].astype(str)
        m = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.08, eval_metric="AUC", thread_count=2,
                               random_seed=0, verbose=250, od_type="Iter", od_wait=150)
        m.fit(Pool(Xt, y, cat_features=CAT, weight=w), eval_set=Pool(Xv, yv, cat_features=CAT), use_best_model=True)
        P["cat"] = m.predict_proba(Xv)[:, 1]
        log("catboost", P["cat"], f"best {m.get_best_iteration()+1} iters (optimistic), {time.time()-t0:.0f}s")
    if "xgb" in P and "cat" in P:
        log("avg(xgb,cat)", (P["xgb"] + P["cat"]) / 2)
    if "xgb" in P and "lgbm" in P:
        log("avg(xgb,lgbm)", (P["xgb"] + P["lgbm"]) / 2)
