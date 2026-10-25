"""Task O3: pick v8 among the top validation candidates with 3 seeds (TRAIN -> VAL only)."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from h_common import LABEL
import task_n2_v7_model as n2, task_o2_v8_model as o2
C = ["pt+epic", "pt+epic+reporter", "pt+epic+reporter+knn", "v8_all"]
ck = o2.data(); tr, va = ck[ck.split == "train"], ck[ck.split == "val"]; yv = va[LABEL].to_numpy()
rows = []
for n in C:
    cols = o2.VARIANTS[n]
    a = [roc_auc_score(yv, n2.fit(tr, cols, s, o2.CFG, o2.HL).predict_proba(va[cols])[:, 1]) for s in (0, 1, 2)]
    rows.append({"variant": n, "val_auc_3seed_mean": np.mean(a), "sd": np.std(a)}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("reports/o3_v8_selection.csv", index=False)
