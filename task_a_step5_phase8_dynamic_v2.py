"""
Task A, Step 5: retrain Phase 8 dynamic models (no-pattern, with-pattern,
with-pattern-no-elapsed_minutes ablation) on the v2 (with-text) feature
table, using Phase 7's exact hyperparameters this time (previously the
dynamic models used lr=0.1/300 rounds vs. Phase 7's lr=0.05/400 -- this
was a stated mismatch, now fixed). Re-run the per-checkpoint three-way
evaluation (static v2 vs. dynamic no-pattern vs. dynamic with-pattern) on
identical test populations.
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

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

# Phase 7's exact hyperparameters -- now matched, previously the dynamic
# models used a different lr/n_estimators than Phase 7.
MATCHED_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
    early_stopping_rounds=20, random_state=42,
)

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)
COLOR_STATIC, COLOR_NO_PATTERN, COLOR_WITH_PATTERN = "#54A24B", "#2a78d6", "#eb6834"


def build_merged():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])

    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck


def main():
    print("=== Task A, Step 5: Phase 8 dynamic v2 (matched hyperparameters) ===")
    merged = build_merged()

    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "val"
    test_mask = merged["split"] == "test"
    print(f"  train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")

    feature_sets = {
        "no_pattern": STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"],
        "with_pattern": STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"],
        "with_pattern_no_elapsed": STATIC_FEATURE_COLS_V2 + ["log1p_stall", "pattern_label"],
    }

    y_train = merged.loc[train_mask, "delayed"]
    y_val = merged.loc[val_mask, "delayed"]

    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}
    for name, cols in feature_sets.items():
        print(f"  training '{name}' ({len(cols)} features) ...")
        model = xgb.XGBClassifier(**MATCHED_PARAMS)
        model.fit(
            merged.loc[train_mask, cols], y_train,
            eval_set=[(merged.loc[val_mask, cols], y_val)], verbose=False,
        )
        models[name] = model
        print(f"    best iteration: {model.best_iteration}")
        joblib.dump(model, f"{MODELS_DIR}/xgb_dynamic_{name}_v2.joblib")

    static_model_v2 = joblib.load(f"{MODELS_DIR}/xgb_static_v2.joblib")

    print("\n  per-checkpoint evaluation (test split) ...")
    rows = []
    for t in range(1, 11):
        sub = merged[test_mask & (merged["checkpoint_index"] == t)]
        n = len(sub)
        if n == 0:
            continue
        y_true = sub["delayed"].to_numpy()
        pos_rate = y_true.mean()
        majority_class = 1 if pos_rate >= 0.5 else 0
        maj_pred = np.full(n, majority_class)

        row = {
            "checkpoint_index": t, "N": n, "pct_delayed": pos_rate,
            "majority_baseline_f1": f1_score(y_true, maj_pred, zero_division=0),
            "majority_baseline_accuracy": accuracy_score(y_true, maj_pred),
        }

        proba_static = static_model_v2.predict_proba(sub[STATIC_FEATURE_COLS_V2])[:, 1]
        pred_static = (proba_static >= 0.5).astype(int)
        row["static_v2_auc"] = roc_auc_score(y_true, proba_static)
        row["static_v2_f1"] = f1_score(y_true, pred_static, zero_division=0)

        for name, cols in feature_sets.items():
            proba = models[name].predict_proba(sub[cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            row[f"{name}_auc"] = roc_auc_score(y_true, proba)
            row[f"{name}_f1"] = f1_score(y_true, pred, zero_division=0)
            row[f"{name}_precision"] = precision_score(y_true, pred, zero_division=0)
            row[f"{name}_recall"] = recall_score(y_true, pred, zero_division=0)

        rows.append(row)
        print(
            f"    M{t}: N={n} static={row['static_v2_auc']:.4f} "
            f"no_pattern={row['no_pattern_auc']:.4f} with_pattern={row['with_pattern_auc']:.4f} "
            f"no_elapsed={row['with_pattern_no_elapsed_auc']:.4f}"
        )

    results_df = pd.DataFrame(rows)
    out_path = f"{REPORTS_DIR}/phase8_dynamic_metrics_v2.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")

    crossover = None
    for _, r in results_df.iterrows():
        if r["no_pattern_auc"] > r["static_v2_auc"] or r["with_pattern_auc"] > r["static_v2_auc"]:
            crossover = int(r["checkpoint_index"])
            break
    print(f"  crossover checkpoint (dynamic first beats static v2): M{crossover}")

    plot_three_way(results_df)
    return results_df


def plot_three_way(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(df["checkpoint_index"], df["static_v2_auc"], color=COLOR_STATIC, linewidth=2,
            marker="o", markersize=5, label="Static baseline v2 (with text)")
    ax.plot(df["checkpoint_index"], df["no_pattern_auc"], color=COLOR_NO_PATTERN, linewidth=2,
            marker="o", markersize=5, label="Dynamic v2, no pattern")
    ax.plot(df["checkpoint_index"], df["with_pattern_auc"], color=COLOR_WITH_PATTERN, linewidth=2,
            marker="o", markersize=5, label="Dynamic v2, with pattern")
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Random (AUC=0.5)")

    ax.set_xlabel("Checkpoint index (1-10)", color=INK_SECONDARY)
    ax.set_ylabel("AUC", color=INK_SECONDARY)
    ax.set_title("v2 (with text): static vs. dynamic AUC, matched hyperparameters", color=INK_PRIMARY)
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
    out_path = f"{REPORTS_DIR}/phase8_checkpoint_accuracy_v2.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
