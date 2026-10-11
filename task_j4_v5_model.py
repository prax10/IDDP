"""
Task J4 (model): v5 = v4 + change-history / known-at-T features.

Variants (all XGBoost, recency weights half-life 4y, multi-seed val selection,
grid extended with depth 10/12 since the feature set is larger):
  v4        Task I model (baseline 43 + 18 activity)
  v5a       v4 + reassignment, sprint, story-point history, field churn, known-at-T rates
  v5b       v5a + position relative to the label threshold (log_elapsed_over_median, past_median)

Scored on: all test rows, censoring-SAFE test rows (Task J1), and UNDETERMINED
rows (past_median == 0, i.e. the label is not yet fixed at T).
Paired bootstrap (1,000) of seed-mean AUC vs v4, 5 seeds.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from h_common import (FEATURES_DIR, REPORTS_DIR, MODELS_DIR, LABEL, SEEDS, seeded, paired_bootstrap_seeds,
                      seed_metric_panel, verdict)
from task_i2_model_upgrades import load, BASE, ACT, weights
from task_j1_censoring_check import safe_flags

SEL_SEEDS = [0, 1]
HL = 4.0
GRID = [{"learning_rate": 0.1, "max_depth": d} for d in [6, 8]] + [{"learning_rate": 0.05, "max_depth": 8}]  # depth 10/12 lost on val, see j4_run_log_partial_depthgrid.txt
CH = ["n_assignee_changes", "ever_reassigned", "assigned_at_T", "log_min_since_assign_change",
      "log_min_to_first_assign", "n_sprint_changes", "n_sprint_carryovers", "in_sprint_at_T",
      "days_to_sprint_end", "sprint_overdue", "sprint_len_days", "sprint_started", "n_sp_changes", "sp_at_T",
      "n_field_chg_total", "n_field_chg_last_7d", "n_priority_chg", "n_type_chg", "n_rank_chg", "n_link_chg",
      "n_fixver_chg", "n_epic_chg", "n_estimate_chg", "n_label_chg", "n_component_chg",
      "proj_rate_known", "proj_rate_known_90d", "proj_n_resolved_90d", "assignee_rate_known",
      "assignee_n_known", "reporter_rate_known"]
THR = ["log_elapsed_over_median", "past_median"]


def load_all():
    ck = load()
    ch = pd.read_csv(f"{FEATURES_DIR}/phase8_change_history_features.csv")
    ck = ck.merge(ch, on=["Issue_ID", "checkpoint_index"], how="left")
    sf = safe_flags().rename(columns={"ID": "Issue_ID"})
    ck = ck.merge(sf[["Issue_ID", "safe"]], on="Issue_ID", how="left")
    for col in ck.columns:
        if ck[col].dtype == "float64":
            ck[col] = ck[col].astype("float32")
        elif ck[col].dtype == "int64" and col not in ("Issue_ID", "ID"):
            ck[col] = pd.to_numeric(ck[col], downcast="integer")
    return ck


def select(train, val, cols, tag, w):
    rows = []
    for cfg in GRID:
        aucs, iters = [], []
        for s in SEL_SEEDS:
            m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **cfg, **seeded(s))
            m.fit(train[cols], train[LABEL], sample_weight=w, eval_set=[(val[cols], val[LABEL])], verbose=False)
            aucs.append(roc_auc_score(val[LABEL], m.predict_proba(val[cols])[:, 1]))
            iters.append(m.best_iteration + 1)
        rows.append({**cfg, "val_auc_mean": np.mean(aucs), "trees_median": int(np.median(iters))})
        print(f"    {tag} {cfg}: val {np.mean(aucs):.4f} trees {int(np.median(iters))}", flush=True)
    g = pd.DataFrame(rows).sort_values("val_auc_mean", ascending=False).reset_index(drop=True)
    b = g.iloc[0]
    cfg = {"learning_rate": float(b["learning_rate"]), "max_depth": int(b["max_depth"]),
           "n_estimators": max(int(b["trees_median"]), 10)}
    print(f"  [{tag}] selected {cfg} val_AUC={b['val_auc_mean']:.4f}", flush=True)
    g.insert(0, "variant", tag)
    return cfg, float(b["val_auc_mean"]), g


def main():
    ck = load_all()
    train = ck[ck.split == "train"]
    val = ck[ck.split == "val"]
    tv = ck[ck.split.isin(["train", "val"])]
    test = ck[ck.split == "test"].reset_index(drop=True)
    y, c = test[LABEL].to_numpy(), test["checkpoint_index"].to_numpy()
    safe = test["safe"].to_numpy(bool)
    undet = test["past_median"].to_numpy() == 0
    w_tr = weights(train, HL, train["age_years"].min())
    w_tv = weights(tv, HL, tv["age_years"].min())

    variants = {"v4": BASE + ACT, "v5a": BASE + ACT + CH, "v5b": BASE + ACT + CH + THR}
    P, sel, grids, models = {}, [], [], {}
    for name, cols in variants.items():
        print(f"\n=== {name} ({len(cols)} features) ===", flush=True)
        cfg, v, g = select(train, val, cols, name, w_tr)
        grids.append(g)
        sel.append({"variant": name, **cfg, "val_auc": v, "n_features": len(cols)})
        P[name], models[name] = [], []
        for s in SEEDS:
            m = xgb.XGBClassifier(**cfg, **seeded(s))
            m.fit(tv[cols], tv[LABEL], sample_weight=w_tv)
            P[name].append(m.predict_proba(test[cols])[:, 1])
            models[name].append(m)
    pd.concat(grids).to_csv(f"{REPORTS_DIR}/j4_tuning_grids.csv", index=False)
    pd.DataFrame(sel).to_csv(f"{REPORTS_DIR}/j4_selected_configs.csv", index=False)

    subsets = {"all": np.ones(len(y), bool), "safe": safe, "undetermined": undet, "safe_undetermined": safe & undet}
    scopes = ["all"] + [f"M{t}" for t in range(1, 11)]
    panel, eff = [], []
    for name in P:
        for sub, mask in subsets.items():
            for sc in scopes:
                m = mask if sc == "all" else mask & (c == int(sc[1:]))
                if m.sum() < 200 or len(np.unique(y[m])) < 2:
                    continue
                panel.append(seed_metric_panel(y[m], [p[m] for p in P[name]], variant=name, subset=sub, scope=sc))
                if name != "v4" and sc in ("all", "M1", "M5", "M10"):
                    bs = paired_bootstrap_seeds(y[m], [p[m] for p in P[name]], [p[m] for p in P["v4"]],
                                                n_boot=1000, seed=3)
                    eff.append({"variant": name, "subset": sub, "scope": sc, **bs,
                                "verdict": verdict(bs["d_auc_lo"], bs["d_auc_hi"])})
    panel = pd.DataFrame(panel)
    panel.to_csv(f"{REPORTS_DIR}/j4_metrics_panel.csv", index=False)
    eff = pd.DataFrame(eff)
    eff.to_csv(f"{REPORTS_DIR}/j4_vs_v4_significance.csv", index=False)
    print("\nTest AUC (5-seed mean):")
    print(panel[panel.scope.isin(["all", "M1", "M5", "M10"])].pivot_table(
        index=["subset", "scope"], columns="variant", values="auc").round(4).to_string())
    print("\nAccuracy@0.5:")
    print(panel[panel.scope.isin(["all", "M1", "M5", "M10"])].pivot_table(
        index=["subset", "scope"], columns="variant", values="accuracy").round(4).to_string())
    print("\nGain vs v4 (95% CI):")
    for _, r in eff.iterrows():
        print(f"  {r['variant']:4s} {r['subset']:18s} {r['scope']:3s} {r['d_auc']:+.4f} [{r['d_auc_lo']:+.4f}, {r['d_auc_hi']:+.4f}]")

    # importance of best variant
    best = max(["v5a", "v5b"], key=lambda k: panel[(panel.variant == k) & (panel.subset == "safe_undetermined") & (panel.scope == "all")]["auc"].iloc[0])
    imp = pd.Series(models[best][0].get_booster().get_score(importance_type="gain"))
    (imp / imp.sum()).sort_values(ascending=False).to_csv(f"{REPORTS_DIR}/j4_gain_importance_{best}.csv")
    print(f"\nTop features ({best}):\n" + (imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())
    np.savez_compressed(f"{REPORTS_DIR}/j4_test_predictions.npz", y=y, checkpoint=c, safe=safe, undet=undet,
                        **{k: np.mean(v, axis=0) for k, v in P.items()})
    import joblib
    for k in ("v5a", "v5b"):
        cols = variants[k]
        for s, m in zip(SEEDS, models[k]):
            joblib.dump(m, f"{MODELS_DIR}/xgb_dynamic_pooled_{k}_seed{s}.joblib")
        json.dump({"features": cols, "config": [x for x in sel if x["variant"] == k][0],
                   "recency_half_life_years": HL, "seeds": SEEDS},
                  open(f"{MODELS_DIR}/xgb_dynamic_pooled_{k}_meta.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
