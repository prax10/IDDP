"""
Task H2 (future-work #2): does more text signal help?

30 PCA components captured only 42.5% of SBERT variance, and text was the
largest SHAP group. Sweep k in {30, 50, 75, 100}:
  - PCA refit on TRAIN-split embeddings only (leakage-safe, same as task_a_step2)
  - for each k, static and pooled-dynamic models are re-selected on
    validation (9-config grid, mean val AUC over 3 seeds), refit on
    train+val with 5 seeds, scored on test
  - paired bootstrap (1,000 resamples) of
    seed-mean AUC(k) - AUC(30)

PCA components are nested (the first 30 of a 100-component PCA are the
30-component PCA), so one fit serves every k. A reproducibility check
confirms our first 30 components match the ones already in the v3 table.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from h_common import (FEATURES_DIR, REPORTS_DIR, LABEL, GRID, SEEDS, dyn_cols, static_cols, text_cols,
                      seeded, load_dynamic, load_static, paired_bootstrap_seeds, seed_metric_panel, verdict)

KS = [30, 50, 75, 100]
SEL_SEEDS = [0, 1, 2]
DYN_SCOPES = ["all", "M1", "M3", "M5", "M8", "M10"]


def build_pca():
    emb = np.load(f"{FEATURES_DIR}/sbert_raw.npy")
    ids = np.load(f"{FEATURES_DIR}/sbert_raw_ids.npy")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv").set_index("ID")["split"]
    train_mask = pd.Series(ids).map(split).to_numpy() == "train"
    print(f"embeddings {emb.shape}, PCA fit rows (train only): {train_mask.sum()}")

    full = PCA(n_components=emb.shape[1], random_state=0).fit(emb[train_mask])
    cum = np.cumsum(full.explained_variance_ratio_)
    var = pd.DataFrame({"k": np.arange(1, len(cum) + 1), "cumulative_variance": cum})
    var.to_csv(f"{REPORTS_DIR}/h2_pca_cumulative_variance.csv", index=False)
    for k in [30, 50, 75, 100, 150, 200]:
        print(f"  k={k:3d}: cumulative variance {cum[k-1]:.4f}")

    pca = PCA(n_components=max(KS), random_state=0).fit(emb[train_mask])
    Z = pca.transform(emb).astype("float32")
    df = pd.DataFrame(Z, columns=text_cols(max(KS)))
    df.insert(0, "ID", ids)

    # reproducibility vs. the 30 components already in phase6_feature_table_v3
    old = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID"] + text_cols(30))
    mm = old.merge(df[["ID"] + text_cols(30)], on="ID", suffixes=("_old", "_new"))
    corr = [abs(np.corrcoef(mm[f"text_pc_{i}_old"], mm[f"text_pc_{i}_new"])[0, 1]) for i in range(30)]
    print(f"  reproducibility: |corr(old PC_i, new PC_i)| min={min(corr):.6f} mean={np.mean(corr):.6f}")
    return df, cum, corr


def select(train, val, cols, tag):
    rows = []
    for cfg in GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(train[cols], train[LABEL], eval_set=[(val[cols], val[LABEL])], verbose=False)
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[cols])[:, 1]))
            iters.append(m.best_iteration + 1)
        rows.append({**cfg, "val_auc_mean": np.mean(aucs), "trees_median": int(np.median(iters))})
    g = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False).reset_index(drop=True)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b["learning_rate"]), "max_depth": int(b["max_depth"]),
           "n_estimators": max(int(b["trees_median"]), 10)}
    print(f"  [{tag}] selected {cfg} val_AUC={b['val_auc_mean']:.4f}", flush=True)
    g.insert(0, "model", tag)
    return cfg, b["val_auc_mean"], g


def fit_seeds(cfg, data, cols, X_test):
    out = []
    for s in SEEDS:
        m = xgb.XGBClassifier(**cfg, **seeded(s))
        m.fit(data[cols], data[LABEL])
        out.append(m.predict_proba(X_test)[:, 1])
    return out


def main():
    text_all, cum, corr = build_pca()
    static_P, dyn_P, grids, sel = {}, {}, [], []
    y_st = y_dyn = ck_idx = None

    for k in KS:
        print(f"\n=== k={k} ===", flush=True)
        override = text_all[["ID"] + text_cols(k)]

        st = load_static(override)
        tr, va = st[st["split"] == "train"], st[st["split"] == "val"]
        tv, te = st[st["split"].isin(["train", "val"])], st[st["split"] == "test"]
        cfg, vauc, g = select(tr, va, static_cols(k), f"static_k{k}")
        grids.append(g)
        static_P[k] = fit_seeds(cfg, tv, static_cols(k), te[static_cols(k)])
        y_st = te[LABEL].to_numpy()
        sel.append({"model": "static", "k": k, **cfg, "val_auc": vauc})
        del st, tr, va, tv

        ck = load_dynamic(override)
        tr, va = ck[ck["split"] == "train"], ck[ck["split"] == "val"]
        tv = ck[ck["split"].isin(["train", "val"])]
        te = ck[ck["split"] == "test"].reset_index(drop=True)
        cfg, vauc, g = select(tr, va, dyn_cols(k), f"pooled_k{k}")
        grids.append(g)
        dyn_P[k] = fit_seeds(cfg, tv, dyn_cols(k), te[dyn_cols(k)])
        y_dyn, ck_idx = te[LABEL].to_numpy(), te["checkpoint_index"].to_numpy()
        sel.append({"model": "pooled_dynamic", "k": k, **cfg, "val_auc": vauc})
        del ck, tr, va, tv, te

    pd.concat(grids).to_csv(f"{REPORTS_DIR}/h2_tuning_grids.csv", index=False)
    pd.DataFrame(sel).to_csv(f"{REPORTS_DIR}/h2_selected_configs.csv", index=False)

    # ---- metrics panel
    panel = []
    for k in KS:
        panel.append(seed_metric_panel(y_st, static_P[k], model="static", k=k, scope="all",
                                       pca_variance=cum[k - 1]))
        for scope in DYN_SCOPES + [f"M{t}" for t in (2, 4, 6, 7, 9)]:
            m = np.ones(len(y_dyn), bool) if scope == "all" else ck_idx == int(scope[1:])
            panel.append(seed_metric_panel(y_dyn[m], [p[m] for p in dyn_P[k]], model="pooled_dynamic",
                                           k=k, scope=scope, pca_variance=cum[k - 1]))
    panel = pd.DataFrame(panel)
    panel.to_csv(f"{REPORTS_DIR}/h2_metrics_panel.csv", index=False)

    # ---- significance vs k=30
    eff = []
    for k in KS[1:]:
        bs = paired_bootstrap_seeds(y_st, static_P[k], static_P[30], n_boot=1000, seed=k)
        eff.append({"model": "static", "k": k, "scope": "all", **bs, "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
        for scope in DYN_SCOPES:
            m = np.ones(len(y_dyn), bool) if scope == "all" else ck_idx == int(scope[1:])
            nb = 1000
            bs = paired_bootstrap_seeds(y_dyn[m], [p[m] for p in dyn_P[k]], [p[m] for p in dyn_P[30]],
                                        n_boot=nb, seed=k)
            eff.append({"model": "pooled_dynamic", "k": k, "scope": scope, **bs,
                        "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
        print(f"  bootstrap done k={k}", flush=True)
    eff = pd.DataFrame(eff)
    eff.to_csv(f"{REPORTS_DIR}/h2_vs_k30_significance.csv", index=False)

    print("\nAUC by k (seed mean):")
    show = panel[panel["scope"].isin(DYN_SCOPES)].pivot_table(index=["model", "scope"], columns="k", values="auc")
    print(show.round(4).to_string())
    print("\nAccuracy by k:")
    print(panel[panel["scope"].isin(DYN_SCOPES)].pivot_table(index=["model", "scope"], columns="k",
                                                              values="accuracy").round(4).to_string())
    print("\nAUC(k) - AUC(30), 95% CI:")
    for _, r in eff.iterrows():
        print(f"  {r['model']:15s} k={r['k']:3d} {r['scope']:4s} {r['d_auc']:+.4f} "
              f"[{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]  {r['verdict']}")
    with open(f"{REPORTS_DIR}/h2_pca_summary.json", "w") as f:
        json.dump({"cumulative_variance": {k: float(cum[k - 1]) for k in KS},
                   "reproducibility_min_abs_corr_first30": float(min(corr))}, f, indent=2)
    plot(panel, cum)


def plot(panel, cum):
    SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#2a78d6", linewidth=2)
    for k in KS:
        ax.axvline(k, color=GRIDC, linewidth=1)
        ax.annotate(f"k={k}\n{cum[k-1]:.0%}", (k, cum[k - 1]), textcoords="offset points", xytext=(4, -24),
                    fontsize=8, color=INK2)
    ax.set_xlabel("PCA components", color=INK2)
    ax.set_ylabel("Cumulative explained variance (train)", color=INK2)
    ax.set_title("SBERT variance captured", color=INK)
    ax = axes[1]
    for scope, c in [("M1", "#9ec5ee"), ("M5", "#2a78d6"), ("M10", "#0b3b75")]:
        s = panel[(panel["model"] == "pooled_dynamic") & (panel["scope"] == scope)].sort_values("k")
        ax.errorbar(s["k"], s["auc"], yerr=s["auc_sd"], color=c, marker="o", capsize=3, label=f"pooled dynamic {scope}")
    s = panel[panel["model"] == "static"].sort_values("k")
    ax.errorbar(s["k"], s["auc"], yerr=s["auc_sd"], color="#54A24B", marker="o", capsize=3, label="static (all test)")
    ax.set_xticks(KS)
    ax.set_xlabel("Text PCA components (k)", color=INK2)
    ax.set_ylabel("Test AUC (mean +/- sd, 5 seeds)", color=INK2)
    ax.set_title("Does more text help?", color=INK)
    ax.legend(fontsize=8, facecolor=SURFACE, edgecolor=GRIDC)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=GRIDC, linewidth=0.8)
        for sp in a.spines.values():
            sp.set_color(GRIDC)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/h2_text_pca_sweep.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
