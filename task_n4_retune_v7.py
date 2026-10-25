"""
Task N4: rolling-origin re-tune for the v7 feature set (same protocol as Task M1; train+val only, test untouched).
Writes reports/n4_rolling_retune_v7.csv and models/n4_v7_selected.json.
"""
import json, os
import numpy as np, pandas as pd
import task_m1_retune as m1
import task_n2_v7_model as n2

GRID = [
    ("A v6 config (d8 mcw50)", {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50}, 4.0),
    ("B d10 mcw50", {"learning_rate": 0.05, "max_depth": 10, "min_child_weight": 50}, 4.0),
    ("C d8 mcw50 col.6", {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50, "colsample_bytree": 0.6}, 4.0),
]
MAX = 300

if __name__ == "__main__":
    cols = n2.VARIANTS["v7"]
    ck = n2.data()
    tv = ck[ck.split.isin(["train", "val"])].reset_index(drop=True)
    del ck
    F = m1.folds(tv)
    out = "reports/n4_rolling_retune_v7.csv"
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    done = {r["config"] for r in rows}
    for name, cfg, hl in GRID:
        if name in done:
            continue
        m1.HL = hl
        mean, curves = m1.run(cfg, tv, cols, F, MAX, name)
        b = int(mean.argmax())
        rows.append({"config": name, **cfg, "half_life": hl, "best_trees": b + 1, "mean_fold_auc": mean[b],
                     "auc_at_109": mean[min(108, len(mean) - 1)],
                     **{f"fold{k+1}_auc": c[b] for k, c in enumerate(curves)}})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{name}: best {b+1} trees, mean fold AUC {mean[b]:.4f} (at 109 trees {mean[min(108, len(mean)-1)]:.4f})", flush=True)
    g = pd.DataFrame(rows).sort_values("mean_fold_auc", ascending=False).iloc[0]
    cfg = {"learning_rate": float(g["learning_rate"]), "max_depth": int(g["max_depth"]), "n_estimators": int(g["best_trees"])}
    for k in ("min_child_weight", "colsample_bytree"):
        if k in g and not pd.isna(g[k]):
            cfg[k] = float(g[k])
    json.dump({"config": cfg, "half_life": float(g["half_life"]), "selected": g["config"]},
              open("models/n4_v7_selected.json", "w"), indent=2)
    print("selected:", g["config"], cfg)
