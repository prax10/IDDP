"""
Task I2: make the dynamic model better, honestly.

Variants (pooled dynamic model, same chronological split, label delayed_v2):
  A  baseline        current 43 features (Phase 8 / Task H)
  B  +activity       A + 18 checkpoint-time activity features (task_i1)
  C  +activity+recency  B with time-decay sample weights; half-life picked on val
  D  LightGBM        B features, LightGBM instead of XGBoost
Every variant is re-selected on validation the same way as Task H
(9-config grid, mean val AUC over 3 seeds), refit on train+val with 5 seeds,
and scored on test. Paired bootstrap (1,000 test resamples) vs A.

Decision thresholds (accuracy / balanced accuracy / F1), per checkpoint:
  t_default = 0.5
  t_prior   = cut-off on the final model's train+val scores at that checkpoint
              so the predicted-delayed share equals the historical delay share
              (uses train+val only, no test information)
  t_val     = threshold maximising balanced accuracy on VAL, from the same
              config fit on TRAIN only (honest val predictions)
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from h_common import (FEATURES_DIR, REPORTS_DIR, LABEL, GRID, SEEDS, dyn_cols, seeded, load_dynamic,
                      paired_bootstrap_seeds, seed_metric_panel, verdict)

SEL_SEEDS = [0, 1, 2]
BASE = dyn_cols(30)
ACT_NUM = ["n_comments", "n_comment_authors", "comments_last_7d", "log_min_since_last_comment",
           "n_transitions", "n_reopens", "n_back_moves", "ever_in_progress", "ever_in_review",
           "log_min_since_last_change", "log_min_since_last_activity",
           "project_open_at_T", "project_resolved_30d", "assignee_open_at_T",
           "log_title_len", "log_desc_len", "desc_has_code"]
ACT = ACT_NUM + ["status_bucket"]
HALF_LIVES_YEARS = [None, 1.0, 2.0, 4.0]


def load():
    ck = load_dynamic()
    act = pd.read_csv(f"{FEATURES_DIR}/phase8_activity_features.csv")
    ck = ck.merge(act, on=["Issue_ID", "checkpoint_index"], how="left")
    assert ck["n_comments"].notna().all()
    ck["status_bucket"] = ck["status_bucket"].astype("category")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv", parse_dates=["Creation_Date"])
    ck = ck.merge(split[["ID", "Creation_Date"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    ck["age_years"] = (ck["Creation_Date"].max() - ck["Creation_Date"]).dt.days / 365.25
    return ck


def weights(df, half_life, ref_age=0.0):
    if half_life is None:
        return None
    return np.power(0.5, (df["age_years"].to_numpy() - ref_age) / half_life)


def xgb_select(train, val, cols, tag, w_train=None):
    rows = []
    for cfg in GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(train[cols], train[LABEL], sample_weight=w_train,
                  eval_set=[(val[cols], val[LABEL])], verbose=False)
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[cols])[:, 1]))
            iters.append(m.best_iteration + 1)
        rows.append({**cfg, "val_auc_mean": np.mean(aucs), "trees_median": int(np.median(iters))})
    g = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False).reset_index(drop=True)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b["learning_rate"]), "max_depth": int(b["max_depth"]),
           "n_estimators": max(int(b["trees_median"]), 10)}
    print(f"  [{tag}] selected {cfg} val_AUC={b['val_auc_mean']:.4f}", flush=True)
    g.insert(0, "variant", tag)
    return cfg, float(b["val_auc_mean"]), g


LGB_GRID = [{"learning_rate": lr, "num_leaves": nl} for lr in [0.03, 0.1] for nl in [31, 127, 255]]
LGB_FIXED = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
                 verbose=-1, n_jobs=2)


def lgb_select(train, val, cols, tag):
    rows = []
    for cfg in LGB_GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = lgb.LGBMClassifier(n_estimators=2000, random_state=s, **cfg, **LGB_FIXED)
            m.fit(train[cols], train[LABEL], eval_set=[(val[cols], val[LABEL])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[cols])[:, 1]))
            iters.append(m.best_iteration_)
        rows.append({**cfg, "val_auc_mean": np.mean(aucs), "trees_median": int(np.median(iters))})
    g = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False).reset_index(drop=True)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b["learning_rate"]), "num_leaves": int(b["num_leaves"]),
           "n_estimators": max(int(b["trees_median"]), 10)}
    print(f"  [{tag}] selected {cfg} val_AUC={b['val_auc_mean']:.4f}", flush=True)
    g.insert(0, "variant", tag)
    return cfg, float(b["val_auc_mean"]), g


def fit_predict(kind, cfg, fit_df, cols, pred_dfs, w=None, seeds=SEEDS):
    outs = [[] for _ in pred_dfs]
    for s in seeds:
        if kind == "xgb":
            m = xgb.XGBClassifier(**cfg, **seeded(s))
        else:
            m = lgb.LGBMClassifier(random_state=s, **cfg, **LGB_FIXED)
        m.fit(fit_df[cols], fit_df[LABEL], sample_weight=w)
        for o, d in zip(outs, pred_dfs):
            o.append(m.predict_proba(d[cols])[:, 1])
    return outs


def main():
    ck = load()
    train = ck[ck["split"] == "train"]
    val = ck[ck["split"] == "val"].reset_index(drop=True)
    trainval = ck[ck["split"].isin(["train", "val"])]
    test = ck[ck["split"] == "test"].reset_index(drop=True)
    y = test[LABEL].to_numpy()
    cki = test["checkpoint_index"].to_numpy()
    val_cki = val["checkpoint_index"].to_numpy()
    print(f"train={len(train)} val={len(val)} test={len(test)}; features base={len(BASE)} +act={len(ACT)}")

    grids, sel, P, Pval, PTV = [], [], {}, {}, {}
    tv_cki = trainval["checkpoint_index"].to_numpy()

    print("\n=== A baseline ===")
    cfgA, vA, g = xgb_select(train, val, BASE, "A_baseline"); grids.append(g)
    P["A_baseline"], PTV["A_baseline"] = fit_predict("xgb", cfgA, trainval, BASE, [test, trainval])
    Pval["A_baseline"], = fit_predict("xgb", cfgA, train, BASE, [val])
    sel.append({"variant": "A_baseline", **cfgA, "val_auc": vA})

    COLS_B = BASE + ACT
    print("\n=== B +activity ===")
    cfgB, vB, g = xgb_select(train, val, COLS_B, "B_activity"); grids.append(g)
    P["B_activity"], PTV["B_activity"] = fit_predict("xgb", cfgB, trainval, COLS_B, [test, trainval])
    Pval["B_activity"], = fit_predict("xgb", cfgB, train, COLS_B, [val])
    sel.append({"variant": "B_activity", **cfgB, "val_auc": vB})

    print("\n=== C +activity +recency weighting (half-life chosen on val, config from B) ===")
    hl_rows = []
    val_ref = train["age_years"].min()  # newest train issue gets weight 1
    for hl in HALF_LIVES_YEARS:
        w = weights(train, hl, val_ref)
        aucs = []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(**cfgB, **seeded(s))
            m.fit(train[COLS_B], train[LABEL], sample_weight=w)
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[COLS_B])[:, 1]))
        hl_rows.append({"half_life_years": hl if hl else "none", "val_auc_mean": np.mean(aucs)})
        print(f"  half-life={hl}: val_AUC={np.mean(aucs):.4f}", flush=True)
    hl_df = pd.DataFrame(hl_rows)
    hl_df.to_csv(f"{REPORTS_DIR}/i2_recency_halflife_val.csv", index=False)
    best_hl = HALF_LIVES_YEARS[int(np.argmax(hl_df["val_auc_mean"].to_numpy()))]
    print(f"  selected half-life: {best_hl}")
    P["C_activity_recency"], PTV["C_activity_recency"] = fit_predict("xgb", cfgB, trainval, COLS_B, [test, trainval],
                                           w=weights(trainval, best_hl, trainval["age_years"].min()))
    Pval["C_activity_recency"], = fit_predict("xgb", cfgB, train, COLS_B, [val], w=weights(train, best_hl, val_ref))
    sel.append({"variant": "C_activity_recency", **cfgB, "val_auc": float(hl_df["val_auc_mean"].max()),
                "half_life_years": best_hl})

    print("\n=== D LightGBM (+activity) ===")
    cfgD, vD, g = lgb_select(train, val, COLS_B, "D_lightgbm"); grids.append(g)
    P["D_lightgbm"], PTV["D_lightgbm"] = fit_predict("lgb", cfgD, trainval, COLS_B, [test, trainval])
    Pval["D_lightgbm"], = fit_predict("lgb", cfgD, train, COLS_B, [val])
    sel.append({"variant": "D_lightgbm", **cfgD, "val_auc": vD})

    pd.concat(grids).to_csv(f"{REPORTS_DIR}/i2_tuning_grids.csv", index=False)
    pd.DataFrame(sel).to_csv(f"{REPORTS_DIR}/i2_selected_configs.csv", index=False)
    np.savez_compressed(f"{REPORTS_DIR}/i2_test_predictions.npz",
                        y=y, checkpoint=cki, **{k: np.mean(v, axis=0) for k, v in P.items()})

    # ---------------- AUC + metric panel ----------------
    scopes = ["all"] + [f"M{t}" for t in range(1, 11)]
    panel = []
    for name, PP in P.items():
        for sc in scopes:
            m = np.ones(len(y), bool) if sc == "all" else cki == int(sc[1:])
            panel.append(seed_metric_panel(y[m], [p[m] for p in PP], variant=name, scope=sc))
    panel = pd.DataFrame(panel)
    panel.to_csv(f"{REPORTS_DIR}/i2_metrics_panel.csv", index=False)
    auc_w = panel.pivot(index="scope", columns="variant", values="auc").reindex(scopes)
    print("\nTest AUC (5-seed mean):")
    print(auc_w.round(4).to_string())

    eff = []
    for name in [k for k in P if k != "A_baseline"]:
        for sc in scopes:
            m = np.ones(len(y), bool) if sc == "all" else cki == int(sc[1:])
            bs = paired_bootstrap_seeds(y[m], [p[m] for p in P[name]], [p[m] for p in P["A_baseline"]],
                                        n_boot=1000, seed=7)
            eff.append({"variant": name, "scope": sc, **bs, "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
    eff = pd.DataFrame(eff)
    eff.to_csv(f"{REPORTS_DIR}/i2_vs_baseline_significance.csv", index=False)
    print("\nAUC gain vs A_baseline, 95% CI:")
    for _, r in eff[eff["scope"].isin(["all", "M1", "M3", "M5", "M10"])].iterrows():
        print(f"  {r['variant']:20s} {r['scope']:4s} {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]")

    # ---------------- thresholds ----------------
    trv_rate = trainval.groupby("checkpoint_index")[LABEL].mean()
    th_rows = []
    for name in P:
        p_test = np.mean(P[name], axis=0)
        p_val = np.mean(Pval[name], axis=0)
        p_tv = np.mean(PTV[name], axis=0)
        for t in range(1, 11):
            m, mv = cki == t, val_cki == t
            yv = val.loc[mv, LABEL].to_numpy()
            grid = np.linspace(0.05, 0.95, 91)
            ba = [balanced_accuracy_score(yv, (p_val[mv] >= g).astype(int)) for g in grid]
            t_val = float(grid[int(np.argmax(ba))])
            # prior threshold: the final model's in-sample train+val scores at this checkpoint,
            # cut so the predicted-delayed share equals the historical (train+val) delay share.
            t_prior = float(np.quantile(p_tv[tv_cki == t], 1 - trv_rate[t]))
            for tname, thr in [("default_0.5", 0.5), ("prior_match", t_prior), ("val_balanced", t_val)]:
                yh = (p_test[m] >= thr).astype(int)
                th_rows.append({"variant": name, "checkpoint": f"M{t}", "threshold_rule": tname, "threshold": thr,
                                "accuracy": accuracy_score(y[m], yh),
                                "balanced_accuracy": balanced_accuracy_score(y[m], yh),
                                "f1": f1_score(y[m], yh), "majority_accuracy": max(y[m].mean(), 1 - y[m].mean())})
    th = pd.DataFrame(th_rows)
    th.to_csv(f"{REPORTS_DIR}/i2_thresholds.csv", index=False)
    print("\nBalanced accuracy by threshold rule (mean over M1-M10):")
    print(th.groupby(["variant", "threshold_rule"])[["accuracy", "balanced_accuracy", "f1"]].mean().round(4).to_string())

    with open(f"{REPORTS_DIR}/i2_summary.json", "w") as f:
        json.dump({"selected": sel, "auc": auc_w.round(5).to_dict()}, f, indent=2, default=str)
    plot(auc_w)


def plot(auc_w):
    SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    colors = {"A_baseline": "#898781", "B_activity": "#2a78d6", "C_activity_recency": "#eb6834", "D_lightgbm": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(1, 11))
    for v, c in colors.items():
        if v in auc_w.columns:
            ax.plot(x, auc_w.loc[[f"M{t}" for t in x], v], color=c, linewidth=2, marker="o", markersize=4, label=v)
    ax.set_xticks(x)
    ax.set_xlabel("Checkpoint", color=INK2)
    ax.set_ylabel("Test AUC (mean of 5 seeds)", color=INK2)
    ax.set_title("Pooled dynamic model: upgrades vs baseline", color=INK)
    ax.grid(color=GRIDC, linewidth=0.8)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDC)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/i2_model_upgrades.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
