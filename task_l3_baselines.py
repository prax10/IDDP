"""
Task L3: stronger baselines -- is XGBoost v5b actually the right model?

All models see the SAME information (the v5b live feature set, computed strictly
before each checkpoint) unless stated, are trained on train+val, and are scored
on the identical 262,433 test checkpoint rows.

  1. Logistic regression          (standardised numerics + missing flags + one-hot categoricals)
  2. Random forest                (sklearn, 300 trees, min_samples_leaf=50, max_samples=0.25)
  3. Literature feature set       Choetkiertikul et al. (2015/2017) style delay-risk factors
                                  -- discussion, priority/fix-version churn, links, workload, assignee
                                  and reporter track record, elapsed time -- with their best learner
                                  (random forest). A re-implementation of the feature *idea* on TAWOS,
                                  not their exact pipeline (their code/data are project-specific).
  4. MLP                          (PyTorch, 2 x 256, dropout, early-stopped on val, refit on train+val)
  5. GRU sequence model           (PyTorch) over each issue's sequence of checkpoint snapshots
                                  M1..Mt -> prediction at every t (causal: step t only sees 1..t)
  ref. XGBoost v5b                (Task J4, 5-seed ensemble)
Metrics: AUC, accuracy@0.5, balanced accuracy, Brier; paired bootstrap of AUC vs v5b.
"""

import gc
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, brier_score_loss

from h_common import LABEL, REPORTS_DIR, paired_bootstrap_seeds, verdict
from task_j4_v5_model import load_all

torch.set_num_threads(2)
CATS = ["Project_ID", "Priority_Normalized", "Type_Normalized", "status_bucket"]
LIT = ["n_comments", "n_comment_authors", "comments_last_7d", "n_priority_chg", "n_fixver_chg", "n_links",
       "n_blocking_links", "n_link_chg", "assignee_open_at_T", "assignee_concurrent_open",
       "assignee_prior_delay_rate", "assignee_prior_resolved_count", "reporter_rate_known", "n_reopens",
       "Priority_Normalized", "Type_Normalized", "log_desc_len", "elapsed_minutes", "n_transitions",
       "Story_Point", "is_unassigned"]


def matrix(df, cols, stats=None):
    num = [c for c in cols if c not in CATS]
    X = df[num].astype("float32")
    if stats is None:
        med = X.median()
        miss_cols = [c for c in num if X[c].isna().any()]
        Xf = X.fillna(med)
        mu, sd = Xf.mean(), Xf.std().replace(0, 1)
        cat_levels = {c: sorted(df[c].astype(str).unique()) for c in cols if c in CATS}
        stats = (med, miss_cols, mu, sd, cat_levels)
    med, miss_cols, mu, sd, cat_levels = stats
    parts = [((X.fillna(med) - mu) / sd).to_numpy("float32"),
             X[miss_cols].isna().to_numpy("float32")]
    for c, lv in cat_levels.items():
        v = df[c].astype(str).to_numpy()
        parts.append((v[:, None] == np.array(lv)[None, :]).astype("float32"))
    return np.hstack(parts), stats


def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5),
            "balanced_accuracy": balanced_accuracy_score(y, p >= 0.5), "brier": brier_score_loss(y, p)}


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.gru = nn.GRU(h, h, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        z = self.inp(x)
        o, _ = self.gru(z)
        return self.out(torch.cat([o, z], -1)).squeeze(-1)


def train_torch(make, Xtr, ytr, Xva=None, yva=None, epochs=12, bs=1024, lr=1e-3, seed=0, mask_tr=None, mask_va=None,
                fixed_epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype("float32"))
    mtr = torch.from_numpy(mask_tr) if mask_tr is not None else None
    best, best_ep, hist = -1, 0, []
    n_ep = fixed_epochs or epochs
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logit = model(Xtr_t[b])
            l = lossf(logit, ytr_t[b])
            if mtr is not None:
                l = (l * mtr[b]).sum() / mtr[b].sum()
            else:
                l = l.mean()
            opt.zero_grad()
            l.backward()
            opt.step()
        if Xva is not None:
            p = predict(model, Xva)
            if mask_va is not None:
                auc = roc_auc_score(yva[mask_va > 0], p[mask_va > 0])
            else:
                auc = roc_auc_score(yva, p)
            hist.append(auc)
            if auc > best:
                best, best_ep = auc, ep + 1
            print(f"      epoch {ep+1}: val AUC {auc:.4f}", flush=True)
    return model, best_ep, hist


def predict(model, X, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy())
    return np.concatenate(out)


def to_sequences(df, X):
    """issue -> [10, d] tensor of its checkpoint rows (zeros where the checkpoint does not exist)."""
    iid = df["Issue_ID"].to_numpy()
    t = df["checkpoint_index"].to_numpy() - 1
    inv, u = pd.factorize(iid)            # order of first appearance (train issues first)
    S = np.zeros((len(u), 10, X.shape[1]), dtype="float32")
    Y = np.zeros((len(u), 10), dtype="float32")
    M = np.zeros((len(u), 10), dtype="float32")
    S[inv, t] = X
    Y[inv, t] = df[LABEL].to_numpy()
    M[inv, t] = 1
    return S, Y, M, inv, t


CACHE = f"{REPORTS_DIR}/l3_cache"


def cached(name, fn):
    """run fn() -> (list of test-prediction arrays, seconds) once; reuse on rerun (OOM-safe)."""
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{name.split(' (')[0].replace(' ', '_').replace('+', 'plus')}.npz"
    if os.path.exists(path):
        z = np.load(path)
        print(f"{name}: cached", flush=True)
        return [z[k] for k in z.files if k.startswith("p")], float(z["secs"])
    t0 = time.time()
    PP = fn()
    secs = time.time() - t0
    np.savez_compressed(path, secs=secs, **{f"p{i}": p for i, p in enumerate(PP)})
    return PP, secs


def main():
    meta = json.load(open("models/xgb_dynamic_live_v5b_meta.json"))
    cols = meta["features"]
    ck = load_all()
    tr, va = ck[ck.split == "train"], ck[ck.split == "val"]
    tv, te = ck[ck.split.isin(["train", "val"])], ck[ck.split == "test"].reset_index(drop=True)
    y = te[LABEL].to_numpy()
    z = np.load(f"{REPORTS_DIR}/j4_test_predictions.npz")
    assert (z["y"] == y).all()
    P = {"XGBoost v5b (reference)": [z["v5b"]]}

    # order train+val so train issues come first -> train / val sequence blocks are views, not copies
    tv = tv.assign(_o=(tv.split == "val").astype(int)).sort_values(["_o", "Issue_ID", "checkpoint_index"]).drop(columns="_o")
    Xtv, stats = matrix(tv, cols)
    Xte, _ = matrix(te, cols, stats)
    ytv = tv[LABEL].to_numpy()
    is_tr = (tv.split == "train").to_numpy()
    ntr = int(is_tr.sum())
    print(f"design matrix: {Xtv.shape}", flush=True)
    times = {}

    def run_lr():
        lr = LogisticRegression(max_iter=300, C=1.0)
        lr.fit(Xtv, ytv)
        return [lr.predict_proba(Xte)[:, 1]]

    def run_rf():
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_samples=0.25, max_features="sqrt",
                                    n_jobs=2, random_state=0)
        rf.fit(Xtv, ytv)
        return [rf.predict_proba(Xte)[:, 1]]

    def run_lit():
        lit = [c for c in LIT if c in tv.columns]
        Ltv, lst = matrix(tv, lit)
        Lte, _ = matrix(te, lit, lst)
        rf2 = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_samples=0.25, max_features="sqrt",
                                     n_jobs=2, random_state=0)
        rf2.fit(Ltv, ytv)
        return [rf2.predict_proba(Lte)[:, 1]]

    for name, fn in [("Logistic regression", run_lr), ("Random forest", run_rf),
                     ("Literature feature set + RF (Choetkiertikul-style)", run_lit)]:
        P[name], times[name] = cached(name, fn)
        gc.collect()
        print(f"{name}: AUC {roc_auc_score(y, P[name][0]):.4f}", flush=True)

    # keep only what the neural models need; free the big dataframes
    seq_tv = tv[["Issue_ID", "checkpoint_index", LABEL]].reset_index(drop=True)
    seq_te = te[["Issue_ID", "checkpoint_index", LABEL]].copy()
    c = te["checkpoint_index"].to_numpy()
    del ck, tr, va, tv, te
    gc.collect()

    # ---- MLP: pick epochs on val (trained on train rows), refit on train+val
    def run_mlp():
        print("MLP: selecting epochs on val", flush=True)
        _, ep, _ = train_torch(lambda: MLP(Xtv.shape[1]), Xtv[:ntr], ytv[:ntr], Xtv[ntr:], ytv[ntr:], epochs=8)
        out = []
        for s in [0, 1]:
            m, _, _ = train_torch(lambda: MLP(Xtv.shape[1]), Xtv, ytv, fixed_epochs=max(ep, 1), seed=s)
            out.append(predict(m, Xte))
        print(f"MLP epochs={ep}", flush=True)
        return out

    P["MLP (PyTorch)"], times["MLP (PyTorch)"] = cached("MLP (PyTorch)", run_mlp)
    print(f"MLP: AUC {np.mean([roc_auc_score(y, p) for p in P['MLP (PyTorch)']]):.4f}", flush=True)

    # ---- GRU over checkpoint sequences
    def predict_seq(model, S, bs=2048):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(S), bs):
                out.append(torch.sigmoid(model(torch.from_numpy(S[i:i + bs]))).numpy())
        return np.concatenate(out)

    def train_gru(S, Y, M, epochs, seed, Sv=None, Yv=None, Mv=None):
        torch.manual_seed(seed)
        model = GRUNet(S.shape[2])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        St, Yt, Mt = torch.from_numpy(S), torch.from_numpy(Y), torch.from_numpy(M)
        best, best_ep = -1, 1
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(St))
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                l = (lossf(model(St[b]), Yt[b]) * Mt[b]).sum() / Mt[b].sum()
                opt.zero_grad()
                l.backward()
                opt.step()
            if Sv is not None:
                p = predict_seq(model, Sv).reshape(-1)
                yv, mv = Yv.reshape(-1), Mv.reshape(-1)
                a = roc_auc_score(yv[mv > 0], p[mv > 0])
                print(f"      epoch {ep+1}: val AUC {a:.4f}", flush=True)
                if a > best:
                    best, best_ep = a, ep + 1
        return model, best_ep

    def run_gru():
        Stv, Ytv, Mtv, inv_tv, _ = to_sequences(seq_tv, Xtv)
        Ste, _, _, inv, tt = to_sequences(seq_te, Xte)
        # issues are wholly in one split; train issues were ordered first
        n_tr_seq = int(np.unique(inv_tv[:ntr]).size)
        assert inv_tv[:ntr].max() < n_tr_seq <= inv_tv[ntr:].min()
        print(f"GRU: {n_tr_seq} train / {len(Stv) - n_tr_seq} val sequences; selecting epochs on val", flush=True)
        _, ep = train_gru(Stv[:n_tr_seq], Ytv[:n_tr_seq], Mtv[:n_tr_seq], 8, 0,
                          Stv[n_tr_seq:], Ytv[n_tr_seq:], Mtv[n_tr_seq:])
        out = []
        for s in [0, 1]:
            m, _ = train_gru(Stv, Ytv, Mtv, max(ep, 1), s)
            out.append(predict_seq(m, Ste)[inv, tt])
        print(f"GRU epochs={ep}", flush=True)
        return out

    P["GRU sequence model (PyTorch)"], times["GRU sequence model (PyTorch)"] = cached("GRU sequence model (PyTorch)", run_gru)
    print(f"GRU: AUC {np.mean([roc_auc_score(y, p) for p in P['GRU sequence model (PyTorch)']]):.4f}", flush=True)

    rows = []
    for name, PP in P.items():
        r = {"model": name, **{k: np.mean([metrics(y, p)[k] for p in PP]) for k in ["auc", "accuracy", "balanced_accuracy", "brier"]},
             "auc_M1": np.mean([roc_auc_score(y[c == 1], p[c == 1]) for p in PP]),
             "auc_M10": np.mean([roc_auc_score(y[c == 10], p[c == 10]) for p in PP]),
             "train_seconds": times.get(name, np.nan)}
        if not name.startswith("XGBoost"):
            bs = paired_bootstrap_seeds(y, PP, P["XGBoost v5b (reference)"], n_boot=500)
            r.update({"auc_minus_xgb": bs["d_auc"], "lo": bs["d_auc_lo"], "hi": bs["d_auc_hi"]})
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    res.to_csv(f"{REPORTS_DIR}/l3_baselines.csv", index=False)
    np.savez_compressed(f"{REPORTS_DIR}/l3_baseline_predictions.npz", y=y, **{k.split(" (")[0].replace(" ", "_"): np.mean(v, 0) for k, v in P.items()})
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
