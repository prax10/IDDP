"""Task N5: test-set report for the N2 variants (scored once). -> reports/n5_results.csv, n5_by_checkpoint.csv, n5_vs_v6.csv"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss
from h_common import paired_bootstrap_seeds

NAMES = sys.argv[1:] or ["v6", "v6_clean", "v7"]
Z = {n: np.load(f"reports/n2_preds_{n}.npz") for n in NAMES}
ref = Z[NAMES[0]]
y, c, past, safe = ref["yt"], ref["ck_t"], ref["past"].astype(bool), ref["safe"].astype(bool)
for n, z in Z.items():
    assert (z["issue"] == ref["issue"]).all() and (z["ck_t"] == c).all()


def prec_top(p, m, frac=0.10):
    idx = np.where(m)[0]
    k = max(1, int(len(idx) * frac))
    top = idx[np.argsort(-p[idx])[:k]]
    return y[top].mean()


rows, ckrows = [], []
for n, z in Z.items():
    P = list(z["test"])
    p = np.mean(P, 0)
    sm = lambda m: float(np.mean([roc_auc_score(y[m], q[m]) for q in P]))
    allm = np.ones(len(y), bool)
    rows.append({
        "variant": n, "AUC overall (seed mean)": sm(allm), "AUC overall (5-seed ensemble)": roc_auc_score(y, p),
        "M1-M3": sm(c <= 3), "M5-M10": sm(c >= 5), "M8-M10": sm(c >= 8),
        "undetermined rows": sm(~past), "censoring-safe + undetermined": sm(safe & ~past),
        "accuracy@0.5": accuracy_score(y, p >= 0.5), "balanced acc@0.5": balanced_accuracy_score(y, p >= 0.5),
        "brier": brier_score_loss(y, p),
        "top10% precision (all open)": prec_top(p, allm), "top10% precision (undetermined)": prec_top(p, ~past),
        "top10% precision at M1": prec_top(p, c == 1), "base rate": y.mean()})
    for k in range(1, 11):
        ckrows.append({"variant": n, "checkpoint": f"M{k}", "auc": sm(c == k)})
res = pd.DataFrame(rows)
res.to_csv("reports/n5_results.csv", index=False)
byck = pd.DataFrame(ckrows).pivot(index="checkpoint", columns="variant", values="auc")
byck = byck.loc[[f"M{k}" for k in range(1, 11)]]
byck.to_csv("reports/n5_by_checkpoint.csv")
print(res.round(4).T.to_string())
print(byck.round(4).to_string())

base = NAMES[0]
cmp = []
for n in NAMES[1:]:
    for scope, m in [("all", np.ones(len(y), bool)), ("M1", c == 1), ("undetermined", ~past)]:
        b = paired_bootstrap_seeds(y[m], [q[m] for q in Z[n]["test"]], [q[m] for q in Z[base]["test"]], n_boot=500)
        cmp.append({"variant": n, "vs": base, "scope": scope, **{k: b[k] for k in ("d_auc", "d_auc_lo", "d_auc_hi")}})
    if n != "v6_clean" and "v6_clean" in Z:
        b = paired_bootstrap_seeds(y, list(Z[n]["test"]), list(Z["v6_clean"]["test"]), n_boot=500)
        cmp.append({"variant": n, "vs": "v6_clean", "scope": "all", **{k: b[k] for k in ("d_auc", "d_auc_lo", "d_auc_hi")}})
cmp = pd.DataFrame(cmp)
cmp.to_csv("reports/n5_vs_v6.csv", index=False)
print(cmp.round(4).to_string(index=False))
