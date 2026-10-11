"""
Task J6 (model part): the "live" dynamic model used by app_v5.

The app must score ANY open issue at ANY date. Every feature can be computed
at an arbitrary moment by feature_engine.FeatureEngine EXCEPT the two Phase 8
trajectory features (log1p_stall needs the dwell-baseline tables, pattern_label
needs the DTW medoid assignment), which only exist at the 10 fixed checkpoints.
So the live model = chosen v5 variant minus those two features. This script
trains it (same config / recency weights / 5 seeds) and measures what it costs.
Usage: python task_j6_live_model.py <variant>
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, SEEDS, seeded, paired_bootstrap_seeds
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

HL = 4.0


def main(variant):
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_pooled_{variant}_meta.json"))
    full_cols = meta["features"]
    cfg = {k: meta["config"][k] for k in ("learning_rate", "max_depth", "n_estimators")}
    live_cols = [c for c in full_cols if c not in ("log1p_stall", "pattern_label")]
    ck = load_all()
    tv = ck[ck.split.isin(["train", "val"])]
    te = ck[ck.split == "test"].reset_index(drop=True)
    y, c = te[LABEL].to_numpy(), te["checkpoint_index"].to_numpy()
    w = weights(tv, HL, tv["age_years"].min())
    P = {"full": [], "live": []}
    models = []
    for s in SEEDS:
        for name, cols in [("full", full_cols), ("live", live_cols)]:
            m = xgb.XGBClassifier(**cfg, **seeded(s))
            m.fit(tv[cols], tv[LABEL], sample_weight=w)
            P[name].append(m.predict_proba(te[cols])[:, 1])
            if name == "live":
                models.append(m)
                joblib.dump(m, f"{MODELS_DIR}/xgb_dynamic_live_{variant}_seed{s}.joblib")
    rows = []
    for sc in ["all"] + [f"M{t}" for t in range(1, 11)]:
        msk = np.ones(len(y), bool) if sc == "all" else c == int(sc[1:])
        bs = paired_bootstrap_seeds(y[msk], [p[msk] for p in P["live"]], [p[msk] for p in P["full"]], n_boot=500)
        rows.append({"scope": sc, "full_auc": np.mean([roc_auc_score(y[msk], p[msk]) for p in P["full"]]),
                     "live_auc": np.mean([roc_auc_score(y[msk], p[msk]) for p in P["live"]]), **bs})
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/j6_live_vs_full_{variant}.csv", index=False)
    print(res[["scope", "full_auc", "live_auc", "d_auc", "d_auc_lo", "d_auc_hi"]].round(4).to_string(index=False))
    # thresholds: prior-matched on train+val in-sample scores, by elapsed/expected bucket (= checkpoint index)
    p_tv = np.mean([m.predict_proba(tv[live_cols])[:, 1] for m in models], axis=0)
    tvc = tv["checkpoint_index"].to_numpy()
    rate = tv.groupby("checkpoint_index")[LABEL].mean()
    thr = {int(t): float(np.quantile(p_tv[tvc == t], 1 - rate[t])) for t in range(1, 11)}
    glob = float(np.quantile(p_tv, 1 - tv[LABEL].mean()))
    cats = {col: list(tv[col].cat.categories) for col in live_cols if str(tv[col].dtype) == "category"}
    json.dump({"features": live_cols, "config": cfg, "recency_half_life_years": HL, "seeds": SEEDS,
               "source_variant": variant, "prior_match_thresholds_by_checkpoint": thr,
               "prior_match_threshold_global": glob, "categories": {k: [str(x) for x in v] for k, v in cats.items()},
               "test_auc_live": float(res.loc[res.scope == "all", "live_auc"].iloc[0]),
               "test_auc_full": float(res.loc[res.scope == "all", "full_auc"].iloc[0])},
              open(f"{MODELS_DIR}/xgb_dynamic_live_{variant}_meta.json", "w"), indent=2)
    print("thresholds", {k: round(v, 3) for k, v in thr.items()}, "global", round(glob, 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v5a")
