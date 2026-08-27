"""
Task B, Part 2: train ten separate per-checkpoint models (v2 features +
pattern_label, Phase 7 hyperparameters) instead of one pooled model, and
compare per-checkpoint AUC against the pooled with-pattern model from
Task A.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]
FEATURE_COLS = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]

PHASE7_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
)


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
    print("=== Task B, Part 2: per-checkpoint models vs. pooled model ===")
    merged = build_merged()
    # train+val combined as the training pool for these per-checkpoint
    # models (Phase 7-style: no early stopping / no held-out val needed)
    train_mask = merged["split"].isin(["train", "val"])
    test_mask = merged["split"] == "test"

    rows = []
    for t in range(1, 11):
        train_t = merged[train_mask & (merged["checkpoint_index"] == t)]
        test_t = merged[test_mask & (merged["checkpoint_index"] == t)]
        n_train, n_test = len(train_t), len(test_t)
        if n_test == 0 or train_t["delayed"].nunique() < 2:
            print(f"  M{t}: skipped (n_train={n_train}, n_test={n_test})")
            continue

        clf = xgb.XGBClassifier(**PHASE7_PARAMS)
        clf.fit(train_t[FEATURE_COLS], train_t["delayed"])
        proba = clf.predict_proba(test_t[FEATURE_COLS])[:, 1]
        auc = roc_auc_score(test_t["delayed"], proba)

        rows.append({
            "checkpoint_index": t, "n_train": n_train, "N_test": n_test,
            "per_checkpoint_auc": auc,
        })
        print(f"  M{t}: n_train={n_train} N_test={n_test} per_checkpoint_AUC={auc:.4f}")

    per_ck_df = pd.DataFrame(rows)

    pooled = pd.read_csv(f"{REPORTS_DIR}/phase8_dynamic_metrics_v2.csv")
    pooled = pooled[["checkpoint_index", "N", "with_pattern_auc"]].rename(
        columns={"with_pattern_auc": "pooled_auc", "N": "N_test_pooled"}
    )

    combined = per_ck_df.merge(pooled, on="checkpoint_index", how="left")
    combined["pooled_minus_per_checkpoint"] = combined["pooled_auc"] - combined["per_checkpoint_auc"]

    out_path = f"{REPORTS_DIR}/phase8_pooled_vs_percheckpoint.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")
    print(combined.to_string(index=False))

    n_pooled_wins = (combined["pooled_minus_per_checkpoint"] > 0).sum()
    n_per_ck_wins = (combined["pooled_minus_per_checkpoint"] < 0).sum()
    print(f"\n  pooled wins at {n_pooled_wins}/{len(combined)} checkpoints, "
          f"per-checkpoint wins at {n_per_ck_wins}/{len(combined)}")
    print(f"  mean(pooled - per_checkpoint) AUC delta: {combined['pooled_minus_per_checkpoint'].mean():+.4f}")

    plot_comparison(combined)
    return combined


def plot_comparison(df):
    surface, ink_primary, ink_secondary, ink_muted, gridline = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
    )
    color_pooled, color_per_ck = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ax.plot(df["checkpoint_index"], df["pooled_auc"], color=color_pooled, linewidth=2,
            marker="o", markersize=5, label="Pooled model (one classifier, all checkpoints)")
    ax.plot(df["checkpoint_index"], df["per_checkpoint_auc"], color=color_per_ck, linewidth=2,
            marker="o", markersize=5, label="Per-checkpoint model (10 separate classifiers)")

    ax.set_xlabel("Checkpoint index (1-10)", color=ink_secondary)
    ax.set_ylabel("AUC", color=ink_secondary)
    ax.set_title("Pooled vs. per-checkpoint dynamic models (test split)", color=ink_primary)
    ax.set_xticks(range(1, 11))
    ax.tick_params(colors=ink_muted)
    ax.grid(color=gridline, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(gridline)
    ax.legend(fontsize=9, facecolor=surface, edgecolor=gridline, labelcolor=ink_primary, loc="lower right")

    ax2 = ax.twinx()
    ax2.bar(df["checkpoint_index"] - 0.15, df["n_train"], width=0.3, color=ink_muted, alpha=0.25, label="n_train")
    ax2.bar(df["checkpoint_index"] + 0.15, df["N_test"], width=0.3, color=ink_muted, alpha=0.5, label="N_test")
    ax2.set_ylabel("N (train / test rows)", color=ink_muted, fontsize=8)
    ax2.tick_params(colors=ink_muted, labelsize=7)
    ax2.legend(fontsize=7, loc="upper right", facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)

    fig.tight_layout()
    out_path = f"{REPORTS_DIR}/phase8_pooled_vs_percheckpoint.png"
    fig.savefig(out_path, dpi=150, facecolor=surface)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
