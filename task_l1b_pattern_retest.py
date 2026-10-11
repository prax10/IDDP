"""
Task L1b: re-test the pattern contribution with FULL-SCALE medoids.

Dynamic pattern labels from the full-scale CLARA medoids (task_l1_full_dtw.py):
  at checkpoint t >= 3 the issue's partial trajectory M1..Mt (log1p stall, NaN
  checkpoints skipped) is matched by DTW to the nearest medoid; t < 3 ->
  insufficient_history. (Phase 8 used M3..Mt by accident; both are reported.)
Variants: old published medoids (as used in the paper), old medoids with the
M1..Mt fix, full-scale 'level' medoids (K=4), full-scale 'shape' medoids (K=3,
z-normalised prefixes).
Models: XGBoost, 3 seeds, test set; paired bootstrap of AUC(with) - AUC(without)
  - on the Phase 8 / Task H baseline feature set (where the pattern was originally judged)
  - on the v5b live feature set (current model)
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from task_l1_full_dtw import dtw_to_medoids, pad
from h_common import LABEL, REPORTS_DIR, seeded, paired_bootstrap_seeds, verdict
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

SEEDS3 = [0, 1, 2]


def assign(ck, medoids, start_at=1, znorm=False):
    """prefix from checkpoint `start_at` .. t, for t >= 3."""
    ck = ck.sort_values(["Issue_ID", "checkpoint_index"])
    v = np.log1p(ck["stall_ratio"].to_numpy())
    iid = ck["Issue_ID"].to_numpy()
    t = ck["checkpoint_index"].to_numpy()
    prefixes, rows = [], []
    cur_id, buf = None, []
    for i in range(len(ck)):
        if iid[i] != cur_id:
            cur_id, buf = iid[i], []
        if t[i] >= start_at and not np.isnan(v[i]):
            buf.append(v[i])
        if t[i] >= 3 and len(buf) > 0:
            p = np.array(buf)
            if znorm:
                sd = p.std()
                p = (p - p.mean()) / sd if sd > 1e-9 else np.zeros_like(p)
            prefixes.append(p[-10:])
            rows.append(i)
    X, L = pad(prefixes)
    MX, ML = pad([np.asarray(m) for m in medoids])
    lab = dtw_to_medoids(X, L, MX, ML).argmin(axis=1)
    out = np.array(["insufficient_history"] * len(ck), dtype=object)
    out[np.array(rows)] = [f"cluster_{c+1}" for c in lab]
    s = pd.Series(out, index=ck.index)
    return s


def main():
    ck = load_all()
    raw = pd.read_csv("data/features/phase8_checkpoints.csv", usecols=["Issue_ID", "checkpoint_index", "stall_ratio"])
    raw = raw.set_index(["Issue_ID", "checkpoint_index"])
    base = ck[["Issue_ID", "checkpoint_index"]].join(raw, on=["Issue_ID", "checkpoint_index"])
    med_old = pd.read_csv("data/features/phase8_medoids.csv")
    old = [g.sort_values("checkpoint_index")["log1p_stall_ratio"].to_numpy() for _, g in med_old.groupby("cluster")]
    new = json.load(open("data/features/l1_full_scale_medoids.json"))
    variants = {
        "old medoids, M3..Mt (as published)": None,     # = existing pattern_label column
        "old medoids, M1..Mt (fixed)": assign(base, old, start_at=1),
        "full-scale level medoids (K=4)": assign(base, new["level (log1p stall)"]["medoids"], start_at=1),
        "full-scale shape medoids (K=3)": assign(base, new["shape (z-normalised)"]["medoids"], start_at=1, znorm=True),
    }
    dist = {}
    for name, s in variants.items():
        col = "pattern_label" if s is None else "pat_" + str(abs(hash(name)) % 10000)
        if s is not None:
            ck[col] = pd.Categorical(s.reindex(ck.index).to_numpy())
        variants[name] = col
        real = ck[ck[col].astype(str) != "insufficient_history"][col].astype(str)
        dist[name] = real.value_counts(normalize=True).round(3).to_dict()
        print(f"{name}: {dist[name]}", flush=True)

    meta = json.load(open("models/xgb_dynamic_live_v5b_meta.json"))
    feature_sets = {
        "baseline (Phase 8 / Task H features)": [c for c in __import__("h_common").dyn_cols(30) if c != "pattern_label"],
        "v5b live features": meta["features"],
    }
    cfgs = {"baseline (Phase 8 / Task H features)": dict(learning_rate=0.05, max_depth=8, n_estimators=29),
            "v5b live features": {k: meta["config"][k] for k in ("learning_rate", "max_depth", "n_estimators")}}
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    w = weights(tv, 4.0, tv["age_years"].min())
    y, c = te[LABEL].to_numpy(), te["checkpoint_index"].to_numpy()
    rows = []
    for fs_name, cols in feature_sets.items():
        def fit(cc):
            P = []
            for s in SEEDS3:
                m = xgb.XGBClassifier(**cfgs[fs_name], **seeded(s))
                m.fit(tv[cc], tv[LABEL], sample_weight=w if fs_name.startswith("v5b") else None)
                P.append(m.predict_proba(te[cc])[:, 1])
            return P
        P0 = fit(cols)
        for vname, col in variants.items():
            P1 = fit(cols + [col])
            for sc in ["all", "M3", "M5", "M10"]:
                msk = np.ones(len(y), bool) if sc == "all" else c == int(sc[1:])
                bs = paired_bootstrap_seeds(y[msk], [p[msk] for p in P1], [p[msk] for p in P0], n_boot=500)
                rows.append({"features": fs_name, "pattern_variant": vname, "scope": sc,
                             "auc_without": np.mean([roc_auc_score(y[msk], p[msk]) for p in P0]),
                             "auc_with": np.mean([roc_auc_score(y[msk], p[msk]) for p in P1]), **bs,
                             "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"], margin=0.005)})
            print(f"  done {fs_name} / {vname}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/l1b_pattern_contribution.csv", index=False)
    json.dump(dist, open(f"{REPORTS_DIR}/l1b_pattern_distributions.json", "w"), indent=2)
    print(res[["features", "pattern_variant", "scope", "auc_without", "auc_with", "d_auc", "d_auc_lo", "d_auc_hi"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
