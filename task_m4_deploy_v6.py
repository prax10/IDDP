"""
Task M4: deployable live model v6 = v5b live features + the M1 rolling-origin-selected config
(lr 0.05, depth 8, min_child_weight 50, 109 trees), 5 seeds on train+val. Same output layout as
task_j6_live_model.py (joblib per seed + meta with prior-match thresholds and categories), so
app_v5_backend can load it with Backend(live_variant="v6").
"""
import json
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from h_common import LABEL, MODELS_DIR, seeded
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

SEEDS = [0, 1, 2, 3, 4]


def main():
    old = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    v6 = json.load(open(f"{MODELS_DIR}/xgb_v6_retuned_meta.json"))
    cols, cfg, hl = v6["features"], v6["config"], v6["half_life"]
    assert cols == old["features"]
    ck = load_all()
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"]
    w = weights(tv, hl, tv["age_years"].min())
    models, P = [], []
    for s in SEEDS:
        m = xgb.XGBClassifier(**{**seeded(s), **cfg}).fit(tv[cols], tv[LABEL], sample_weight=w)
        joblib.dump(m, f"{MODELS_DIR}/xgb_dynamic_live_v6_seed{s}.joblib")
        models.append(m)
        P.append(m.predict_proba(te[cols])[:, 1])
    auc = float(np.mean([roc_auc_score(te[LABEL], p) for p in P]))
    p_tv = np.mean([m.predict_proba(tv[cols])[:, 1] for m in models], axis=0)
    tvc = tv["checkpoint_index"].to_numpy()
    rate = tv.groupby("checkpoint_index")[LABEL].mean()
    thr = {int(t): float(np.quantile(p_tv[tvc == t], 1 - rate[t])) for t in range(1, 11)}
    glob = float(np.quantile(p_tv, 1 - tv[LABEL].mean()))
    json.dump({**old, "config": cfg, "recency_half_life_years": hl, "seeds": SEEDS, "source_variant": "v6 (M1 retune of v5b)",
               "prior_match_thresholds_by_checkpoint": thr, "prior_match_threshold_global": glob,
               "test_auc_live": auc, "selection": "rolling-origin validation over 3 windows of train+val (reports/m1_rolling_retune.csv)"},
              open(f"{MODELS_DIR}/xgb_dynamic_live_v6_meta.json", "w"), indent=2)
    print(f"v6 live: test AUC {auc:.4f} (M3 stage: 0.8719 expected); global threshold {glob:.3f}")


if __name__ == "__main__":
    main()
