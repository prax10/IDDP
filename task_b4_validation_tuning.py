"""
Task B4: validation-based hyperparameter tuning for static, pooled
dynamic, and per-checkpoint dynamic models (v3 labels, v2/with-text
features), then the final tuned three-line comparison + robustness table
extension.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]
DYN_FEATURE_COLS = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]

GRID = [
    {"learning_rate": lr, "max_depth": d}
    for lr in [0.05, 0.1, 0.2]
    for d in [4, 6, 8]
]
N_ESTIMATORS_CAP = 1000
EARLY_STOP = 30
FIXED = dict(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
             enable_categorical=True, tree_method="hist", random_state=42)

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)
COLOR_STATIC, COLOR_POOLED, COLOR_PERCK = "#54A24B", "#2a78d6", "#eb6834"


def load_static():
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")
    return feat


def load_dynamic():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck


def tune_static(feat):
    print("=== Step 1a: tuning static model on validation ===")
    train = feat[feat["split"] == "train"]
    val = feat[feat["split"] == "val"]
    X_train, y_train = train[STATIC_FEATURE_COLS_V2], train["delayed_v2"]
    X_val, y_val = val[STATIC_FEATURE_COLS_V2], val["delayed_v2"]

    results = []
    for cfg in GRID:
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS_CAP, early_stopping_rounds=EARLY_STOP, **cfg, **FIXED
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        results.append({**cfg, "best_iteration": model.best_iteration, "val_auc": auc})
        print(f"  lr={cfg['learning_rate']} depth={cfg['max_depth']}: "
              f"best_iter={model.best_iteration} val_AUC={auc:.4f}")

    results_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    best = results_df.iloc[0]
    print(f"\n  selected: lr={best['learning_rate']} depth={best['max_depth']} "
          f"n_estimators={int(best['best_iteration'])} (val_AUC={best['val_auc']:.4f})")
    return {
        "learning_rate": best["learning_rate"], "max_depth": int(best["max_depth"]),
        "n_estimators": max(int(best["best_iteration"]), 10),
    }, best["val_auc"], results_df


def tune_dynamic(ck):
    print("\n=== Step 1b: tuning pooled dynamic model on validation (shared with per-checkpoint) ===")
    train = ck[ck["split"] == "train"]
    val = ck[ck["split"] == "val"]
    X_train, y_train = train[DYN_FEATURE_COLS], train["delayed_v2"]
    X_val, y_val = val[DYN_FEATURE_COLS], val["delayed_v2"]

    results = []
    for cfg in GRID:
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS_CAP, early_stopping_rounds=EARLY_STOP, **cfg, **FIXED
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        results.append({**cfg, "best_iteration": model.best_iteration, "val_auc": auc})
        print(f"  lr={cfg['learning_rate']} depth={cfg['max_depth']}: "
              f"best_iter={model.best_iteration} val_AUC={auc:.4f}")

    results_df = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    best = results_df.iloc[0]
    print(f"\n  selected (shared by pooled + per-checkpoint): lr={best['learning_rate']} "
          f"depth={int(best['max_depth'])} n_estimators={int(best['best_iteration'])} "
          f"(val_AUC={best['val_auc']:.4f})")
    return {
        "learning_rate": best["learning_rate"], "max_depth": int(best["max_depth"]),
        "n_estimators": max(int(best["best_iteration"]), 10),
    }, best["val_auc"], results_df


def main():
    feat = load_static()
    ck = load_dynamic()

    static_cfg, static_val_auc, static_grid = tune_static(feat)
    dyn_cfg, dyn_val_auc, dyn_grid = tune_dynamic(ck)
    static_grid.to_csv(f"{REPORTS_DIR}/phase8_tuning_grid_static.csv", index=False)
    dyn_grid.to_csv(f"{REPORTS_DIR}/phase8_tuning_grid_dynamic.csv", index=False)

    print("\n=== Refit static on train+val with selected config, evaluate on test ===")
    trainval = feat[feat["split"].isin(["train", "val"])]
    test = feat[feat["split"] == "test"]
    static_final = xgb.XGBClassifier(**static_cfg, **FIXED)
    static_final.fit(trainval[STATIC_FEATURE_COLS_V2], trainval["delayed_v2"])
    static_test_proba = static_final.predict_proba(test[STATIC_FEATURE_COLS_V2])[:, 1]
    static_test_auc = roc_auc_score(test["delayed_v2"], static_test_proba)
    print(f"  static: config={static_cfg} val_AUC={static_val_auc:.4f} test_AUC={static_test_auc:.4f}")
    joblib.dump(static_final, f"{MODELS_DIR}/xgb_static_v3_tuned.joblib")

    print("\n=== Refit pooled dynamic on train+val with selected config ===")
    trainval_ck = ck[ck["split"].isin(["train", "val"])]
    test_ck = ck[ck["split"] == "test"]
    pooled_final = xgb.XGBClassifier(**dyn_cfg, **FIXED)
    pooled_final.fit(trainval_ck[DYN_FEATURE_COLS], trainval_ck["delayed_v2"])
    joblib.dump(pooled_final, f"{MODELS_DIR}/xgb_dynamic_pooled_v3_tuned.joblib")
    print(f"  pooled: config={dyn_cfg} val_AUC={dyn_val_auc:.4f}")

    print("\n=== Refit 10 per-checkpoint models with shared selected config ===")
    per_ck_models = {}
    for t in range(1, 11):
        train_t = trainval_ck[trainval_ck["checkpoint_index"] == t]
        clf = xgb.XGBClassifier(**dyn_cfg, **FIXED)
        clf.fit(train_t[DYN_FEATURE_COLS], train_t["delayed_v2"])
        per_ck_models[t] = clf
        joblib.dump(clf, f"{MODELS_DIR}/xgb_dynamic_percheckpoint_M{t}_v3_tuned.joblib")
        print(f"  M{t}: n_train={len(train_t)}")

    print("\n=== Step 2: final tuned three-line per-checkpoint comparison ===")
    rows = []
    for t in range(1, 11):
        sub = test_ck[test_ck["checkpoint_index"] == t]
        n = len(sub)
        if n == 0:
            continue
        y_true = sub["delayed_v2"].to_numpy()
        pos_rate = y_true.mean()
        majority_class = 1 if pos_rate >= 0.5 else 0
        maj_acc = accuracy_score(y_true, np.full(n, majority_class))

        proba_static = static_final.predict_proba(sub[STATIC_FEATURE_COLS_V2])[:, 1]
        proba_pooled = pooled_final.predict_proba(sub[DYN_FEATURE_COLS])[:, 1]
        proba_perck = per_ck_models[t].predict_proba(sub[DYN_FEATURE_COLS])[:, 1]

        rows.append({
            "checkpoint_index": t, "N": n, "pct_delayed_v2": pos_rate,
            "majority_baseline_accuracy": maj_acc,
            "static_tuned_auc": roc_auc_score(y_true, proba_static),
            "pooled_tuned_auc": roc_auc_score(y_true, proba_pooled),
            "per_checkpoint_tuned_auc": roc_auc_score(y_true, proba_perck),
        })
        print(f"  M{t}: N={n} static={rows[-1]['static_tuned_auc']:.4f} "
              f"pooled={rows[-1]['pooled_tuned_auc']:.4f} "
              f"per_checkpoint={rows[-1]['per_checkpoint_tuned_auc']:.4f}")

    final_df = pd.DataFrame(rows)
    final_df.to_csv(f"{REPORTS_DIR}/phase8_final_tuned_comparison.csv", index=False)
    print(f"\n  saved -> {REPORTS_DIR}/phase8_final_tuned_comparison.csv")

    plot_final(final_df)

    print("\n=== Step 3: settling the headline questions ===")
    pooled_beats_static = final_df[final_df["pooled_tuned_auc"] > final_df["static_tuned_auc"]]
    print(f"  Q1: pooled beats static at: "
          f"{pooled_beats_static['checkpoint_index'].tolist() if len(pooled_beats_static) else 'NEVER'}")
    if len(pooled_beats_static):
        for _, r in pooled_beats_static.iterrows():
            print(f"     M{int(r['checkpoint_index'])}: pooled={r['pooled_tuned_auc']:.4f} "
                  f"vs static={r['static_tuned_auc']:.4f} (+{r['pooled_tuned_auc']-r['static_tuned_auc']:.4f})")

    per_ck_beats_pooled = final_df["per_checkpoint_tuned_auc"] > final_df["pooled_tuned_auc"]
    mean_gap = (final_df["per_checkpoint_tuned_auc"] - final_df["pooled_tuned_auc"]).mean()
    print(f"  Q2: per-checkpoint beats pooled at {per_ck_beats_pooled.sum()}/10 checkpoints, "
          f"mean gap={mean_gap:+.4f}")

    crossover = None
    for _, r in final_df.iterrows():
        if r["per_checkpoint_tuned_auc"] > r["static_tuned_auc"]:
            crossover = int(r["checkpoint_index"])
            break
    print(f"  Q3: per-checkpoint vs static crossover: M{crossover}")

    print("\n=== Step 4: extending robustness table with tuned column ===")
    robustness = pd.read_csv(f"{REPORTS_DIR}/label_definition_robustness.csv")
    tuned_values = {
        "Phase 7 static AUC": f"{static_test_auc:.4f}",
        "Pooled dynamic (with-pattern) crossover vs static": (
            "/".join(f"M{c}" for c in pooled_beats_static["checkpoint_index"])
            if len(pooled_beats_static) else "NEVER (static wins all 10 checkpoints)"
        ),
        "Per-checkpoint dynamic crossover vs static": f"M{crossover}",
        "Per-checkpoint M10 AUC": f"{final_df.iloc[-1]['per_checkpoint_tuned_auc']:.4f}",
        "Pooled (with-pattern) M10 AUC": f"{final_df.iloc[-1]['pooled_tuned_auc']:.4f}",
    }
    robustness["Tuned (validation-selected hyperparameters)"] = robustness["Result"].map(tuned_values)
    robustness.to_csv(f"{REPORTS_DIR}/label_definition_robustness.csv", index=False)
    print(robustness.to_string(index=False))
    print(f"\n  saved -> {REPORTS_DIR}/label_definition_robustness.csv")

    print(f"\n  selected static config: {static_cfg}")
    print(f"  selected dynamic config (pooled + per-checkpoint): {dyn_cfg}")


def plot_final(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(df["checkpoint_index"], df["static_tuned_auc"], color=COLOR_STATIC, linewidth=2,
            marker="o", markersize=5, label="Static (tuned)")
    ax.plot(df["checkpoint_index"], df["pooled_tuned_auc"], color=COLOR_POOLED, linewidth=2,
            marker="o", markersize=5, label="Pooled dynamic (tuned)")
    ax.plot(df["checkpoint_index"], df["per_checkpoint_tuned_auc"], color=COLOR_PERCK, linewidth=2,
            marker="o", markersize=5, label="Per-checkpoint dynamic (tuned)")
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Random (AUC=0.5)")

    ax.set_xlabel("Checkpoint index (1-10)", color=INK_SECONDARY)
    ax.set_ylabel("AUC", color=INK_SECONDARY)
    ax.set_title("Final tuned comparison (v3 labels, validation-selected hyperparameters)", color=INK_PRIMARY)
    ax.set_xticks(range(1, 11))
    ax.tick_params(colors=INK_MUTED)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_PRIMARY, loc="lower right")

    ax2 = ax.twinx()
    ax2.bar(df["checkpoint_index"], df["N"], color=INK_MUTED, alpha=0.12, width=0.5)
    ax2.set_ylabel("N (test issues)", color=INK_MUTED, fontsize=8)
    ax2.tick_params(colors=INK_MUTED, labelsize=7)

    fig.tight_layout()
    out_path = f"{REPORTS_DIR}/phase8_final_tuned_comparison.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
