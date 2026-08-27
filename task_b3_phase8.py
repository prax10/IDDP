"""
Task B3, Step 2.2: re-run Phase 8 dynamic models (pooled no-pattern,
pooled with-pattern, per-checkpoint with-pattern) using delayed_v2 labels.
Regenerate the three-line per-checkpoint comparison (static v3 / pooled /
per-checkpoint) on identical test populations, with N, %delayed, and
majority-F1 per checkpoint.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

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

MATCHED_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
    early_stopping_rounds=20, random_state=42,
)
PHASE7_PARAMS_NO_EARLYSTOP = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
)

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)
COLOR_STATIC, COLOR_POOLED, COLOR_PERCK = "#54A24B", "#2a78d6", "#eb6834"


def build_merged():
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


def main():
    print("=== Task B3: Phase 8 v3 (delayed_v2 labels) ===")
    merged = build_merged()
    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "val"
    trainval_mask = merged["split"].isin(["train", "val"])
    test_mask = merged["split"] == "test"

    feature_sets = {
        "no_pattern": STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"],
        "with_pattern": STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"],
    }

    y_train = merged.loc[train_mask, "delayed_v2"]
    y_val = merged.loc[val_mask, "delayed_v2"]

    print("\n-- Pooled models --")
    pooled_models = {}
    for name, cols in feature_sets.items():
        model = xgb.XGBClassifier(**MATCHED_PARAMS)
        model.fit(
            merged.loc[train_mask, cols], y_train,
            eval_set=[(merged.loc[val_mask, cols], y_val)], verbose=False,
        )
        pooled_models[name] = model
        joblib.dump(model, f"{MODELS_DIR}/xgb_dynamic_{name}_v3.joblib")
        print(f"  pooled '{name}': best_iteration={model.best_iteration}")

    static_model_v3 = joblib.load(f"{MODELS_DIR}/xgb_static_v3.joblib")

    print("\n-- Per-checkpoint models (with pattern) --")
    per_ck_feature_cols = feature_sets["with_pattern"]
    per_ck_models = {}
    for t in range(1, 11):
        train_t = merged[trainval_mask & (merged["checkpoint_index"] == t)]
        clf = xgb.XGBClassifier(**PHASE7_PARAMS_NO_EARLYSTOP)
        clf.fit(train_t[per_ck_feature_cols], train_t["delayed_v2"])
        per_ck_models[t] = clf
        joblib.dump(clf, f"{MODELS_DIR}/xgb_dynamic_percheckpoint_M{t}_v3.joblib")
        print(f"  M{t}: n_train={len(train_t)}")

    print("\n-- Per-checkpoint evaluation (test split) --")
    rows = []
    for t in range(1, 11):
        sub = merged[test_mask & (merged["checkpoint_index"] == t)]
        n = len(sub)
        if n == 0:
            continue
        y_true = sub["delayed_v2"].to_numpy()
        pos_rate = y_true.mean()
        majority_class = 1 if pos_rate >= 0.5 else 0
        maj_pred = np.full(n, majority_class)

        proba_static = static_model_v3.predict_proba(sub[STATIC_FEATURE_COLS_V2])[:, 1]
        static_auc = roc_auc_score(y_true, proba_static)

        proba_pooled_np = pooled_models["no_pattern"].predict_proba(sub[feature_sets["no_pattern"]])[:, 1]
        pooled_no_pattern_auc = roc_auc_score(y_true, proba_pooled_np)
        proba_pooled_wp = pooled_models["with_pattern"].predict_proba(sub[feature_sets["with_pattern"]])[:, 1]
        pooled_with_pattern_auc = roc_auc_score(y_true, proba_pooled_wp)

        proba_perck = per_ck_models[t].predict_proba(sub[per_ck_feature_cols])[:, 1]
        per_checkpoint_auc = roc_auc_score(y_true, proba_perck)

        rows.append({
            "checkpoint_index": t, "N": n, "pct_delayed_v2": pos_rate,
            "majority_baseline_f1": f1_score(y_true, maj_pred, zero_division=0),
            "static_v3_auc": static_auc,
            "pooled_no_pattern_auc": pooled_no_pattern_auc,
            "pooled_with_pattern_auc": pooled_with_pattern_auc,
            "per_checkpoint_auc": per_checkpoint_auc,
        })
        print(f"  M{t}: N={n} static={static_auc:.4f} pooled_np={pooled_no_pattern_auc:.4f} "
              f"pooled_wp={pooled_with_pattern_auc:.4f} per_checkpoint={per_checkpoint_auc:.4f}")

    results_df = pd.DataFrame(rows)
    out_path = f"{REPORTS_DIR}/phase8_dynamic_metrics_v3.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")

    crossover_pooled = None
    crossover_perck = None
    for _, r in results_df.iterrows():
        if crossover_pooled is None and r["pooled_with_pattern_auc"] > r["static_v3_auc"]:
            crossover_pooled = int(r["checkpoint_index"])
        if crossover_perck is None and r["per_checkpoint_auc"] > r["static_v3_auc"]:
            crossover_perck = int(r["checkpoint_index"])
    print(f"\n  crossover (pooled with-pattern vs static): M{crossover_pooled}")
    print(f"  crossover (per-checkpoint vs static): M{crossover_perck}")

    plot_three_line(results_df)
    return results_df


def plot_three_line(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(df["checkpoint_index"], df["static_v3_auc"], color=COLOR_STATIC, linewidth=2,
            marker="o", markersize=5, label="Static v3 (train+val-median labels)")
    ax.plot(df["checkpoint_index"], df["pooled_with_pattern_auc"], color=COLOR_POOLED, linewidth=2,
            marker="o", markersize=5, label="Pooled dynamic, with pattern")
    ax.plot(df["checkpoint_index"], df["per_checkpoint_auc"], color=COLOR_PERCK, linewidth=2,
            marker="o", markersize=5, label="Per-checkpoint dynamic, with pattern")
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Random (AUC=0.5)")

    ax.set_xlabel("Checkpoint index (1-10)", color=INK_SECONDARY)
    ax.set_ylabel("AUC", color=INK_SECONDARY)
    ax.set_title("v3 (train+val-median labels): static vs. pooled vs. per-checkpoint", color=INK_PRIMARY)
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
    out_path = f"{REPORTS_DIR}/phase8_checkpoint_accuracy_v3.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
