"""
Task P2: CatBoost vs XGBoost on the v8 features, judged on the 3 rolling-origin windows of train+val
(same protocol as M1/N4; test untouched). Iteration count chosen on the mean window curve.
"""
import json, time
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from h_common import LABEL
from task_i2_model_upgrades import weights
import task_m1_retune as m1, task_o2_v8_model as o2
from task_p1_big_swings import COLS, CAT

MAXIT = 1200
CFGS = [("cat d8 lr.05", {"depth": 8, "learning_rate": 0.05}), ("cat d10 lr.05", {"depth": 10, "learning_rate": 0.05})]

if __name__ == "__main__":
    ck = o2.data()
    tv = ck[ck.split.isin(["train", "val"])].reset_index(drop=True)
    del ck
    F = m1.folds(tv)
    X = tv[COLS].copy()
    for c in CAT:
        X[c] = X[c].astype(str)
    y = tv[LABEL].to_numpy()
    rows = []
    # XGB reference on the same windows with the v8 config
    m1.HL = o2.HL
    xm, _ = m1.run({k: v for k, v in o2.CFG.items() if k != "n_estimators"}, tv, COLS, F, 300, "xgb v8")
    rows.append({"model": "xgb v8 (d8 mcw50 col.6)", "best_iters": int(xm.argmax()) + 1, "mean_window_auc": xm.max(),
                 "at_130": xm[129]})
    print(rows[-1], flush=True)
    for name, cfg in CFGS:
        curves = []
        for k, (trm, vam) in enumerate(F):
            t0 = time.time()
            w = weights(tv[trm], o2.HL, tv.loc[trm, "age_years"].min())
            m = CatBoostClassifier(iterations=MAXIT, eval_metric="AUC", thread_count=2, random_seed=0, verbose=0,
                                   use_best_model=False, **cfg)
            m.fit(Pool(X[trm], y[trm], cat_features=CAT, weight=w), eval_set=Pool(X[vam], y[vam], cat_features=CAT))
            cur = np.array(m.get_evals_result()["validation"]["AUC"])
            curves.append(cur)
            print(f"  {name} window {k+1}: best {cur.max():.4f} @ {cur.argmax()+1}  ({time.time()-t0:.0f}s)", flush=True)
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], 0)
        rows.append({"model": name, "best_iters": int(mean.argmax()) + 1, "mean_window_auc": mean.max(),
                     **{f"w{k+1}": c[mean.argmax()] for k, c in enumerate(curves)}})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("reports/p2_catboost_rolling.csv", index=False)
    pd.DataFrame(rows).to_csv("reports/p2_catboost_rolling.csv", index=False)
