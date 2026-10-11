"""
Task J9: deployment-realistic ROLLING RETRAINING.

Task H showed recent data matters a lot; a model in production would be
retrained regularly rather than frozen at the end of the validation period.
Protocol over the test period:
  for every calendar month m in the test period:
     training rows = checkpoint rows of ANY issue whose outcome is known at
                     the start of m (Resolution_Date < start of m), from all splits
     predict       = test checkpoint rows whose checkpoint moment T falls in m
Compared, on exactly the same rows, with the frozen model (trained once on
train+val). Same features / config / recency weighting as the frozen model,
3 seeds. Reported on all rows and on censoring-safe rows (J1).

Usage: python task_j9_rolling_retrain.py <variant>   (v4 | v5a | v5b)
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded, paired_bootstrap_seeds, verdict
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all
from feature_engine import secs

SEEDS3 = [0, 1, 2]
HL = 4.0


def main(variant):
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_pooled_{variant}_meta.json")) if variant != "v4" else \
        json.load(open(f"{MODELS_DIR}/xgb_dynamic_pooled_v4_activity_meta.json"))
    cols = meta["features"]
    cfg = meta["config"]
    cfg = {k: cfg[k] for k in ("learning_rate", "max_depth", "n_estimators")}
    ck = load_all()
    iss = pd.read_csv("data/raw/issue_slim.csv.gz", usecols=["ID", "Creation_Date", "Resolution_Date"])
    iss["c_s"], iss["r_s"] = secs(iss["Creation_Date"]), secs(iss["Resolution_Date"])
    ck = ck.merge(iss[["ID", "c_s", "r_s"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    ck["T"] = ck["c_s"] + np.floor(ck["elapsed_minutes"].astype(float) * 60).astype(np.int64)
    ck["T_month"] = pd.to_datetime(ck["T"], unit="s").dt.to_period("M")

    tv = ck[ck.split.isin(["train", "val"])]
    test = ck[ck.split == "test"].reset_index(drop=True)
    y = test[LABEL].to_numpy()

    # frozen model (as deployed in Task I / J4)
    P_frozen = []
    for s in SEEDS3:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(tv[cols], tv[LABEL], sample_weight=weights(tv, HL, tv["age_years"].min()))
        P_frozen.append(m.predict_proba(test[cols])[:, 1])

    P_roll = [np.full(len(test), np.nan) for _ in SEEDS3]
    months = sorted(test["T_month"].unique())
    log = []
    for mth in months:
        start = int(pd.Timestamp(mth.start_time).timestamp())
        tr = ck[ck["r_s"].between(0, start - 1)]
        idx = np.where(test["T_month"].to_numpy() == mth)[0]
        if len(idx) == 0:
            continue
        w = weights(tr, HL, tr["age_years"].min())
        for k, s in enumerate(SEEDS3):
            m = xgb.XGBClassifier(**cfg, **seeded(s))
            m.fit(tr[cols], tr[LABEL], sample_weight=w)
            P_roll[k][idx] = m.predict_proba(test.iloc[idx][cols])[:, 1]
        n_test_known = (tr.split == "test").sum()
        log.append({"month": str(mth), "n_pred_rows": len(idx), "n_train_rows": len(tr),
                    "n_train_rows_from_test_period": int(n_test_known)})
        print(f"  {mth}: predict {len(idx)} rows; train {len(tr)} rows ({n_test_known} from test-period issues)", flush=True)
    pd.DataFrame(log).to_csv(f"{REPORTS_DIR}/j9_rolling_months_{variant}.csv", index=False)

    ok = ~np.isnan(P_roll[0])
    rows = []
    c = test["checkpoint_index"].to_numpy()
    safe = test["safe"].to_numpy(bool)
    for sub, mask in [("all", ok), ("safe", ok & safe), ("at_risk", ok & ~safe)]:
        for sc in ["all", "M1", "M3", "M5", "M10"]:
            m = mask if sc == "all" else mask & (c == int(sc[1:]))
            bs = paired_bootstrap_seeds(y[m], [p[m] for p in P_roll], [p[m] for p in P_frozen], n_boot=1000, seed=5)
            rows.append({"subset": sub, "scope": sc, "N": int(m.sum()),
                         "frozen_auc": np.mean([roc_auc_score(y[m], p[m]) for p in P_frozen]),
                         "rolling_auc": np.mean([roc_auc_score(y[m], p[m]) for p in P_roll]),
                         "frozen_acc": np.mean([((p[m] >= .5) == y[m]).mean() for p in P_frozen]),
                         "rolling_acc": np.mean([((p[m] >= .5) == y[m]).mean() for p in P_roll]),
                         **bs, "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
    res = pd.DataFrame(rows)
    res.to_csv(f"{REPORTS_DIR}/j9_rolling_vs_frozen_{variant}.csv", index=False)
    print(res[["subset", "scope", "N", "frozen_auc", "rolling_auc", "d_auc", "d_auc_lo", "d_auc_hi",
               "frozen_acc", "rolling_acc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v5a")
