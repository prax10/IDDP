"""
Task I4: freeze the improved dynamic model (variant C) and re-test the
static-vs-dynamic crossover with it.
  - 5-seed XGBoost ensemble, lr=0.05, depth=8, 32 trees, train+val, recency
    half-life 4 years, features = baseline 43 + 18 activity features
  - per-checkpoint 'prior_match' thresholds (train+val only) saved for the app
  - static (B4 tuned config, 5 seeds) vs new dynamic, paired bootstrap per checkpoint
"""
import json, joblib, numpy as np, pandas as pd, xgboost as xgb
from h_common import LABEL, REPORTS_DIR, MODELS_DIR, TUNED_STATIC, SEEDS, static_cols, seeded, load_static, paired_bootstrap_seeds
from task_i2_model_upgrades import load, BASE, ACT, weights

CFG = dict(learning_rate=0.05, max_depth=8, n_estimators=32); HL = 4.0; COLS = BASE + ACT
ck = load(); tv = ck[ck.split.isin(["train", "val"])]; test = ck[ck.split == "test"].reset_index(drop=True)
w = weights(tv, HL, tv["age_years"].min())
P, PTV = [], []
for s in SEEDS:
    m = xgb.XGBClassifier(**CFG, **seeded(s)); m.fit(tv[COLS], tv[LABEL], sample_weight=w)
    joblib.dump(m, f"{MODELS_DIR}/xgb_dynamic_pooled_v4_activity_seed{s}.joblib")
    P.append(m.predict_proba(test[COLS])[:, 1]); PTV.append(m.predict_proba(tv[COLS])[:, 1])
ptv = np.mean(PTV, axis=0); tv_c = tv["checkpoint_index"].to_numpy()
rate = tv.groupby("checkpoint_index")[LABEL].mean()
thr = {int(t): float(np.quantile(ptv[tv_c == t], 1 - rate[t])) for t in range(1, 11)}
json.dump({"features": COLS, "config": CFG, "recency_half_life_years": HL, "seeds": SEEDS,
           "ensemble": "mean predict_proba over the 5 seed models",
           "prior_match_thresholds": thr}, open(f"{MODELS_DIR}/xgb_dynamic_pooled_v4_activity_meta.json", "w"), indent=2)
print("thresholds:", {k: round(v, 3) for k, v in thr.items()})

st = load_static(); st_tv = st[st.split.isin(["train", "val"])]
PS = []
for s in SEEDS:
    m = xgb.XGBClassifier(**TUNED_STATIC, **seeded(s)); m.fit(st_tv[static_cols(30)], st_tv[LABEL])
    PS.append(m.predict_proba(test[static_cols(30)])[:, 1])
y, c = test[LABEL].to_numpy(), test["checkpoint_index"].to_numpy()
rows = []
for t in range(1, 11):
    mm = c == t
    bs = paired_bootstrap_seeds(y[mm], [p[mm] for p in P], [p[mm] for p in PS], n_boot=1000, seed=t)
    rows.append({"checkpoint": f"M{t}", "dynamic_v4_minus_static": bs["d_auc"], "lo": bs["d_auc_lo"], "hi": bs["d_auc_hi"]})
    print(f"  M{t}: dynamic_v4 - static = {bs['d_auc']:+.4f} [{bs['d_auc_lo']:+.4f}, {bs['d_auc_hi']:+.4f}]")
pd.DataFrame(rows).to_csv(f"{REPORTS_DIR}/i4_dynamic_v4_vs_static.csv", index=False)
