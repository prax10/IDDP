"""
Task O4: does an equal-weight blend of v8 XGBoost with an MLP (same v8 features) help?
Equal weights, nothing fitted on validation or test. Stage "val": TRAIN-only models scored on VAL.
Stage "test": TRAIN+VAL models, MLP 3 seeds, averaged with the saved v8 XGBoost test predictions.
"""
import sys, gc
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from h_common import LABEL
import task_l3_baselines as l3
import task_n2_v7_model as n2, task_o2_v8_model as o2

l3.CATS = l3.CATS + ["type_T"]
COLS = o2.VARIANTS["v8"]


def mlp_preds(Xtr, ytr, Xte, seeds):
    return [l3.predict(l3.train_torch(lambda: l3.MLP(Xtr.shape[1]), Xtr, ytr, fixed_epochs=1, seed=s)[0], Xte) for s in seeds]


def rank(p):
    return pd.Series(p).rank(pct=True).to_numpy()


if __name__ == "__main__":
    ck = o2.data()
    if sys.argv[1] == "val":
        tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
        yv = va[LABEL].to_numpy()
        px = np.mean([n2.fit(tr, COLS, s, o2.CFG, o2.HL).predict_proba(va[COLS])[:, 1] for s in (0, 1)], 0)
        Xtr, st = l3.matrix(tr, COLS)
        Xva, _ = l3.matrix(va, COLS, st)
        ytr = tr[LABEL].to_numpy()
        del ck, tr; gc.collect()
        pm = np.mean(mlp_preds(Xtr, ytr, Xva, (0, 1)), 0)
        rows = [{"model": "xgb v8", "val_auc": roc_auc_score(yv, px)},
                {"model": "mlp v8", "val_auc": roc_auc_score(yv, pm)},
                {"model": "avg prob xgb+mlp", "val_auc": roc_auc_score(yv, (px + pm) / 2)},
                {"model": "avg rank xgb+mlp", "val_auc": roc_auc_score(yv, (rank(px) + rank(pm)) / 2)}]
        d = pd.DataFrame(rows); d.to_csv("reports/o4_blend_val.csv", index=False); print(d.round(4).to_string(index=False))
    else:
        tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
        Xtv, st = l3.matrix(tv, COLS)
        Xte, _ = l3.matrix(te, COLS, st)
        ytv = tv[LABEL].to_numpy()
        del ck, tv; gc.collect()
        PM = mlp_preds(Xtv, ytv, Xte, (0, 1, 2))
        z = np.load("reports/n2_preds_v8.npz")
        assert (z["issue"] == te["Issue_ID"].to_numpy()).all()
        PX = list(z["test"])
        y = z["yt"]
        blend = [(px + pm) / 2 for px, pm in zip(PX, PM + PM[:2])]   # pair each XGB seed with an MLP seed
        np.savez_compressed("reports/n2_preds_v8_blend.npz", test=np.array(blend), yt=y, ck_t=z["ck_t"], past=z["past"],
                            safe=z["safe"], issue=z["issue"])
        np.savez_compressed("reports/o4_mlp_test.npz", test=np.array(PM))
        print("MLP test AUC", np.mean([roc_auc_score(y, p) for p in PM]).round(4),
              " blend test AUC", np.mean([roc_auc_score(y, p) for p in blend]).round(4))
