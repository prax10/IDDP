"""
Task M3: build and test the v6 candidates. TEST IS SCORED ONCE, AT THE END, FOR PRE-DECLARED VARIANTS.

Stage "xgb"  (python task_m3_v6.py xgb)
  For each XGB variant: fit on TRAIN -> predict VAL (for stacking weights), fit on TRAIN+VAL -> predict TEST.
    v5b_live        reference (v5b config, live features)
    retuned         best config from M1 rolling-origin validation, live features
    retuned_ctext   same + comment-text features (M2), if data/features/m2_comment_text_features.parquet exists
Stage "nn"   (python task_m3_v6.py nn <feature_variant>)
  MLP and GRU on the chosen feature set, same TRAIN->VAL / TRAIN+VAL->TEST protocol, 2 seeds.
Stage "stack" (python task_m3_v6.py stack)
  Stacking weights = logistic regression on the members' VAL logits (fitted on VAL only), applied to
  the TEST logits of the train+val refits. Reported with paired bootstrap against v5b live.
All prediction arrays go to reports/m3_preds_<name>.npz; final table reports/m3_v6_results.csv.
"""
import gc
import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss

from h_common import LABEL, REPORTS_DIR, MODELS_DIR, seeded, paired_bootstrap_seeds, verdict
from task_i2_model_upgrades import weights
from task_j4_v5_model import load_all

HL = 4.0
SEEDS = [0, 1, 2, 3, 4]
CTEXT = "data/features/m2_comment_text_features.parquet"


def data(with_ctext):
    ck = load_all()
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    cols = list(meta["features"])
    if with_ctext:
        ct = pd.read_parquet(CTEXT)
        ck = ck.merge(ct, on=["Issue_ID", "checkpoint_index"], how="left")
        cols += [c for c in ct.columns if c not in ("Issue_ID", "checkpoint_index")]
    return ck, cols, meta


def best_retuned():
    g = pd.read_csv(f"{REPORTS_DIR}/m1_rolling_retune.csv").sort_values("mean_fold_auc", ascending=False).iloc[0]
    cfg = {"learning_rate": float(g["learning_rate"]), "max_depth": int(g["max_depth"]),
           "n_estimators": int(g["best_trees"])}
    for k in ("min_child_weight", "colsample_bytree", "reg_lambda"):
        if k in g and not pd.isna(g[k]):
            cfg[k] = float(g[k])
    return cfg, float(g["half_life"]), str(g["config"])


def xgb_stage():
    meta = json.load(open(f"{MODELS_DIR}/xgb_dynamic_live_v5b_meta.json"))
    v5 = {k: meta["config"][k] for k in ("learning_rate", "max_depth", "n_estimators")}
    rc, rhl, rname = best_retuned()
    print(f"retuned config from M1: {rname} -> {rc}, half-life {rhl}", flush=True)
    variants = [("v5b_live", v5, HL, False), ("retuned", rc, rhl, False)]
    if os.path.exists(CTEXT):
        variants.append(("retuned_ctext", rc, rhl, True))
    for name, cfg, hl, ct in variants:
        out = f"{REPORTS_DIR}/m3_preds_{name}.npz"
        if os.path.exists(out):
            print(f"{name}: cached", flush=True)
            continue
        ck, cols, _ = data(ct)
        tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
        tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"]
        pv = [xgb.XGBClassifier(**{**seeded(s), **cfg}).fit(tr[cols], tr[LABEL], sample_weight=weights(tr, hl, tr["age_years"].min()))
              .predict_proba(va[cols])[:, 1] for s in SEEDS[:3]]
        pt = []
        for s in SEEDS:
            m = xgb.XGBClassifier(**{**seeded(s), **cfg}).fit(tv[cols], tv[LABEL], sample_weight=weights(tv, hl, tv["age_years"].min()))
            pt.append(m.predict_proba(te[cols])[:, 1])
            if name != "v5b_live":
                m.save_model(f"{MODELS_DIR}/xgb_v6_{name}_seed{s}.json")
        np.savez_compressed(out, val=np.mean(pv, 0), yv=va[LABEL].to_numpy(), test=np.array(pt), yt=te[LABEL].to_numpy(),
                            ck_t=te["checkpoint_index"].to_numpy(), past=te["past_median"].to_numpy(),
                            safe=te["safe"].to_numpy() if "safe" in te else np.ones(len(te)))
        json.dump({"features": cols, "config": cfg, "half_life": hl}, open(f"{MODELS_DIR}/xgb_v6_{name}_meta.json", "w"), indent=2)
        print(f"{name}: val AUC {roc_auc_score(va[LABEL], np.mean(pv, 0)):.4f} (test scored only in stack stage)", flush=True)
        del ck, tr, va, tv, te
        gc.collect()


def nn_stage(variant):
    import torch
    from task_l3_baselines import matrix, MLP, GRUNet, train_torch, predict, to_sequences
    torch.set_num_threads(2)
    ck, cols, _ = data(variant == "ctext")
    ck = ck.assign(_o=ck.split.map({"train": 0, "val": 1, "test": 2})).sort_values(["_o", "Issue_ID", "checkpoint_index"])
    tr_m, va_m, te_m = (ck._o == 0).to_numpy(), (ck._o == 1).to_numpy(), (ck._o == 2).to_numpy()
    y = ck[LABEL].to_numpy()
    X_tv, stats = matrix(ck[tr_m | va_m], cols)            # standardisation from train+val (unsupervised)
    X_te, _ = matrix(ck[te_m], cols, stats)
    seq_tv = ck.loc[tr_m | va_m, ["Issue_ID", "checkpoint_index", LABEL]].reset_index(drop=True)
    seq_te = ck.loc[te_m, ["Issue_ID", "checkpoint_index", LABEL]].reset_index(drop=True)
    ntr = int(tr_m.sum())
    y_tv, y_te = y[tr_m | va_m], y[te_m]
    del ck
    gc.collect()
    res = {}
    # MLP: 1 epoch (L3 showed val AUC peaks at epoch 1), train->val and train+val->test
    pv = [predict(train_torch(lambda: MLP(X_tv.shape[1]), X_tv[:ntr], y_tv[:ntr], fixed_epochs=1, seed=s)[0], X_tv[ntr:]) for s in (0, 1)]
    pt = [predict(train_torch(lambda: MLP(X_tv.shape[1]), X_tv, y_tv, fixed_epochs=1, seed=s)[0], X_te) for s in (0, 1)]
    res["mlp"] = (np.mean(pv, 0), np.array(pt))
    print(f"MLP val AUC {roc_auc_score(y_tv[ntr:], res['mlp'][0]):.4f}", flush=True)
    # GRU
    S, Yq, Mq, inv_tv, t_tv = to_sequences(seq_tv, X_tv)
    St, _, _, inv_te, t_te = to_sequences(seq_te, X_te)
    n_tr_seq = int(np.unique(inv_tv[:ntr]).size)
    import torch.nn as nn

    def fit_gru(Sx, Yx, Mx, seed):
        torch.manual_seed(seed)
        m = GRUNet(Sx.shape[2]); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.BCEWithLogitsLoss(reduction="none"); St_, Yt_, Mt_ = map(torch.from_numpy, (Sx, Yx, Mx))
        m.train(); perm = torch.randperm(len(St_))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; l = (lf(m(St_[b]), Yt_[b]) * Mt_[b]).sum() / Mt_[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        return m

    def pred_seq(m, Sx):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Sx), 2048):
                out.append(torch.sigmoid(m(torch.from_numpy(Sx[i:i + 2048]))).numpy())
        return np.concatenate(out)
    pv, pt = [], []
    for s in (0, 1):
        m = fit_gru(S[:n_tr_seq], Yq[:n_tr_seq], Mq[:n_tr_seq], s)
        pv.append(pred_seq(m, S[n_tr_seq:])[inv_tv[ntr:] - n_tr_seq, t_tv[ntr:]])
        m = fit_gru(S, Yq, Mq, s)
        pt.append(pred_seq(m, St)[inv_te, t_te])
    res["gru"] = (np.mean(pv, 0), np.array(pt))
    print(f"GRU val AUC {roc_auc_score(y_tv[ntr:], res['gru'][0]):.4f}", flush=True)
    for k, (v, t) in res.items():
        np.savez_compressed(f"{REPORTS_DIR}/m3_preds_{k}_{variant}.npz", val=v, yv=y_tv[ntr:], test=t, yt=y_te)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5),
            "balanced_acc": balanced_accuracy_score(y, p >= 0.5), "brier": brier_score_loss(y, p)}


def stack_stage():
    ref = np.load(f"{REPORTS_DIR}/m3_preds_v5b_live.npz")
    yt, ckt, past, safe = ref["yt"], ref["ck_t"], ref["past"], ref["safe"]
    members = {}
    for f in sorted(os.listdir(REPORTS_DIR)):
        if f.startswith("m3_preds_") and f.endswith(".npz"):
            z = np.load(f"{REPORTS_DIR}/{f}")
            members[f[9:-4]] = z
    order = lambda z: z  # all arrays share the load_all test order except NN ones (sorted by Issue_ID, checkpoint)
    # align NN members (sorted) to load_all test order
    te_key = None
    rows, P = [], {}
    base = load_all()
    te = base[base.split == "test"][["Issue_ID", "checkpoint_index"]].reset_index(drop=True)
    va = base[base.split == "val"][["Issue_ID", "checkpoint_index"]].reset_index(drop=True)
    del base
    srt_te = te.sort_values(["Issue_ID", "checkpoint_index"]).index.to_numpy()
    srt_va = va.sort_values(["Issue_ID", "checkpoint_index"]).index.to_numpy()
    for name, z in members.items():
        v, t = z["val"], z["test"]
        if name.startswith(("mlp", "gru")):          # undo the sort
            vv, tt = np.empty_like(v), np.empty_like(t)
            vv[srt_va] = v
            tt[:, srt_te] = t
            v, t = vv, tt
        P[name] = (v, list(t))
    yv = members["v5b_live"]["yv"]
    singles = [k for k in P]
    cands = {k: [k] for k in singles}
    nn = [k for k in ("mlp_ctext", "gru_ctext") if k in P] or [k for k in ("mlp_live", "gru_live") if k in P]
    for bx in ("retuned", "retuned_ctext"):
        if bx in P and len(nn) == 2:
            cands[f"stack: {bx} + MLP + GRU (val-fitted weights)"] = [bx] + nn
            cands[f"average: {bx} + MLP + GRU (equal weights)"] = ["AVG"] + [bx] + nn
    out_rows = []
    for label, mem in cands.items():
        if len(mem) == 1:
            Pt = P[mem[0]][1]
        elif mem[0] == "AVG":
            Pt = [np.mean([np.mean(P[m][1], 0) for m in mem[1:]], axis=0)]
        else:
            Zv = np.column_stack([logit(P[m][0]) for m in mem])
            lr = LogisticRegression(C=1.0).fit(Zv, yv)                 # weights from VAL only
            Zt = np.column_stack([logit(np.mean(P[m][1], 0)) for m in mem])
            Pt = [lr.predict_proba(Zt)[:, 1]]
            print(f"{label}: stacking weights {dict(zip(mem, lr.coef_[0].round(3)))}", flush=True)
        r = {"model": label, **{k: np.mean([metrics(yt, p)[k] for p in Pt]) for k in ["auc", "accuracy", "balanced_acc", "brier"]}}
        for k in (1, 3, 5, 8, 10):
            msk = ckt == k
            r[f"auc_M{k}"] = np.mean([roc_auc_score(yt[msk], p[msk]) for p in Pt])
        und = past == 0
        r["auc_undetermined"] = np.mean([roc_auc_score(yt[und], p[und]) for p in Pt])
        r["auc_safe_undetermined"] = np.mean([roc_auc_score(yt[und & (safe == 1)], p[und & (safe == 1)]) for p in Pt])
        early, late = np.isin(ckt, [1, 2, 3]), ckt >= 5
        r["auc_M1_3"] = np.mean([roc_auc_score(yt[early], p[early]) for p in Pt])
        r["auc_M5_10"] = np.mean([roc_auc_score(yt[late], p[late]) for p in Pt])
        if label != "v5b_live":
            bs = paired_bootstrap_seeds(yt, Pt, P["v5b_live"][1], n_boot=500)
            r.update({"d_auc_vs_v5b": bs["d_auc"], "lo": bs["d_auc_lo"], "hi": bs["d_auc_hi"],
                      "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"], margin=0.002)})
        out_rows.append(r)
    res = pd.DataFrame(out_rows).sort_values("auc", ascending=False)
    res.to_csv(f"{REPORTS_DIR}/m3_v6_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "xgb":
        xgb_stage()
    elif stage == "nn":
        nn_stage(sys.argv[2])
    elif stage == "stack":
        stack_stage()
