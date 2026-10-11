"""
Task M1: re-tune the live v5b model with ROLLING-ORIGIN validation (train+val only; test untouched).

Why: v5b's 43 trees were picked by early stopping on one validation window that is unusually
hard (val AUC < test AUC for every model since Phase 8), so the model may be under-fitted.
Folds (issues ordered by creation date within train+val; each window = 1/8 of train+val):
  fold 1: fit on issues before 62.5%, score [62.5%, 75%)
  fold 2: fit on issues before 75%,   score [75%, 87.5%)
  fold 3: fit on issues before 87.5%, score [87.5%, 100%]  (= the official validation period)
Each config is fitted once per fold up to MAX_TREES with per-iteration validation AUC logged; the
tree count maximising the mean fold AUC is chosen. Selection never looks at test.
"""
import json, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

HL = 4.0


def folds(tv):
    cd = pd.read_csv("data/features/phase8_split.csv", usecols=["ID", "Creation_Date"]).set_index("ID")["Creation_Date"]
    t = pd.to_datetime(tv["Issue_ID"].map(cd)).astype("int64").to_numpy()
    q = np.quantile(t, [0.625, 0.75, 0.875])
    edges = list(q) + [t.max() + 1]
    return [(t < edges[k], (t >= edges[k]) & (t < edges[k + 1])) for k in range(3)]


def run(cfg, tv, cols, F, max_trees, tag=""):
    curves = []
    for k, (trm, vam) in enumerate(F):
        tr, va = tv[trm], tv[vam]
        m = xgb.XGBClassifier(**{**seeded(0), "eval_metric": "auc", "n_estimators": max_trees, **cfg})
        t0 = time.time()
        m.fit(tr[cols], tr[LABEL], sample_weight=weights(tr, HL, tr["age_years"].min()),
              eval_set=[(va[cols], va[LABEL])], verbose=False)
        curves.append(np.array(m.evals_result()["validation_0"]["auc"]))
        print(f"    {tag} fold {k+1}: best {curves[-1].max():.4f} @ {curves[-1].argmax()+1} trees, "
              f"at 43 trees {curves[-1][min(42, len(curves[-1])-1)]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    mean = np.mean(curves, axis=0)
    return mean, curves


if __name__ == "__main__":
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    cols = meta["features"]
    ck = load_all()
    tv = ck[ck.split.isin(["train", "val"])].reset_index(drop=True)
    del ck
    F = folds(tv)
    print("fold sizes:", [(int(a.sum()), int(b.sum())) for a, b in F], flush=True)
    if sys.argv[1:] == ["timing"]:
        run({"learning_rate": 0.05, "max_depth": 8}, tv, cols, F[2:], 200, "timing")
        sys.exit()
    GRID = [
        ("A v5b (lr.05 d8)",            {"learning_rate": 0.05, "max_depth": 8}, 4.0),
        ("B d6",                        {"learning_rate": 0.05, "max_depth": 6}, 4.0),
        ("C d10",                       {"learning_rate": 0.05, "max_depth": 10}, 4.0),
        ("D d8 mcw50",                  {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50}, 4.0),
        ("E d8 colsample.5",            {"learning_rate": 0.05, "max_depth": 8, "colsample_bytree": 0.5}, 4.0),
        ("F d8 lambda10 mcw20",         {"learning_rate": 0.05, "max_depth": 8, "reg_lambda": 10, "min_child_weight": 20}, 4.0),
        ("G lr.03 d8 mcw20 col.6",      {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20, "colsample_bytree": 0.6}, 4.0),
        ("H v5b, half-life 2y",         {"learning_rate": 0.05, "max_depth": 8}, 2.0),
        ("I d6 mcw50 col.6, hl 2y",     {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 50, "colsample_bytree": 0.6}, 2.0),
    ]
    MAX = 400
    import os
    done = pd.read_csv(f"{REPORTS_DIR}/m1_rolling_retune.csv") if os.path.exists(f"{REPORTS_DIR}/m1_rolling_retune.csv") else None
    rows = done.to_dict("records") if done is not None else []
    skip = set(done["config"]) if done is not None else set()
    skip |= {"F d8 lambda10 mcw20", "G lr.03 d8 mcw20 col.6"}   # near-duplicates of D (flat landscape A-D); skipped for time
    for name, cfg, hl in GRID:
        if name in skip:
            continue
        HL = hl
        globals()["HL"] = hl
        mean, curves = run(cfg, tv, cols, F, MAX, name)
        b = int(mean.argmax())
        rows.append({"config": name, **cfg, "half_life": hl, "best_trees": b + 1, "mean_fold_auc": mean[b],
                     **{f"fold{k+1}_auc_at_best": c[min(b, len(c)-1)] for k, c in enumerate(curves)},
                     "mean_auc_at_43": mean[min(42, len(mean)-1)], "mean_auc_at_100": mean[min(99, len(mean)-1)],
                     "mean_auc_at_200": mean[min(199, len(mean)-1)]})
        print(f"  ==> {name}: mean fold AUC {mean[b]:.4f} @ {b+1} trees (v5b-style 43 trees: {rows[-1]['mean_auc_at_43']:.4f})", flush=True)
        pd.DataFrame(rows).to_csv(f"{REPORTS_DIR}/m1_rolling_retune.csv", index=False)
    print(pd.DataFrame(rows).sort_values("mean_fold_auc", ascending=False).round(4).to_string(index=False))
