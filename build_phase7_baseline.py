"""
Phase 7: static XGBoost baseline (chronological 80/20 split, test set
aligned to Phase 8's test split) + Part B: class-balance / majority-
baseline context added to Phase 8's per-checkpoint milestone table.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
)
from xgboost import XGBClassifier

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]


def part_a_train_static_baseline():
    model_path = os.path.join(MODELS_DIR, "xgb_static_baseline.joblib")
    metrics_path = os.path.join(REPORTS_DIR, "phase7_metrics.json")
    if os.path.exists(model_path) or os.path.exists(metrics_path):
        print("Phase 7 artifacts already exist -- skipping training, loading existing.")
        return joblib.load(model_path), json.load(open(metrics_path))

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== Part A: Phase 7 static baseline ===")
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    issue = pd.read_csv(
        os.path.join("data/raw", "issue.csv"),
        usecols=["ID", "Creation_Date"],
        parse_dates=["Creation_Date"],
    )
    feat = feat.merge(issue, on="ID", how="left")
    assert feat["Creation_Date"].notna().all()

    feat = feat.sort_values("Creation_Date", kind="mergesort").reset_index(drop=True)
    n = len(feat)
    n_train = int(round(n * 0.80))
    own_split = np.array(["train"] * n_train + ["test"] * (n - n_train))
    feat["own_split"] = own_split

    phase8_split = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_split.csv"))
    phase8_test_ids = set(phase8_split.loc[phase8_split["split"] == "test", "ID"])
    own_test_ids = set(feat.loc[feat["own_split"] == "test", "ID"])

    if own_test_ids == phase8_test_ids:
        print(
            f"  Verified: recomputed 80/20 test split ({len(own_test_ids)} issues) "
            "matches Phase 8's test split exactly."
        )
        feat["split"] = feat["own_split"]
    else:
        overlap = len(own_test_ids & phase8_test_ids)
        print(
            f"  MISMATCH: recomputed test split has {len(own_test_ids)} issues, "
            f"Phase 8's test split has {len(phase8_test_ids)}, overlap={overlap}. "
            "Using Phase 8's test split directly instead, per instructions."
        )
        feat["split"] = np.where(
            feat["ID"].isin(phase8_test_ids), "test", "train"
        )

    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    train = feat[feat["split"] == "train"]
    test = feat[feat["split"] == "test"]
    print(f"  train={len(train)} test={len(test)}")

    X_train, y_train = train[STATIC_FEATURE_COLS], train["delayed"]
    X_test, y_test = test[STATIC_FEATURE_COLS], test["delayed"]

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        enable_categorical=True, tree_method="hist",
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()

    majority_class = y_train.mode()[0]
    maj_pred = np.full(len(y_test), majority_class)
    maj_f1 = f1_score(y_test, maj_pred, zero_division=0)
    maj_acc = accuracy_score(y_test, maj_pred)
    maj_precision = precision_score(y_test, maj_pred, zero_division=0)
    maj_recall = recall_score(y_test, maj_pred, zero_division=0)

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "test_split_matches_phase8": own_test_ids == phase8_test_ids,
        "model": {
            "auc": auc, "f1": f1, "precision": precision, "recall": recall,
            "accuracy": acc, "confusion_matrix": cm,
        },
        "majority_baseline": {
            "majority_class": int(majority_class),
            "f1": maj_f1, "precision": maj_precision, "recall": maj_recall,
            "accuracy": maj_acc,
        },
        "hyperparameters": {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss",
        },
    }

    joblib.dump(clf, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  AUC={auc:.4f} F1={f1:.4f} precision={precision:.4f} recall={recall:.4f} acc={acc:.4f}")
    print(f"  majority baseline: F1={maj_f1:.4f} acc={maj_acc:.4f}")
    print(f"  confusion matrix: {cm}")
    print(f"  saved model -> {model_path}")
    print(f"  saved metrics -> {metrics_path}")

    plot_phase7(y_test, proba, metrics)
    return clf, metrics


def plot_phase7(y_test, proba, metrics):
    from sklearn.metrics import roc_curve

    surface, ink_primary, ink_secondary, gridline = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    fpr, tpr, _ = roc_curve(y_test, proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot(fpr, tpr, color="#2a78d6", linewidth=2, label=f"XGBoost (AUC={metrics['model']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=gridline, linewidth=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", color=ink_secondary)
    ax.set_ylabel("True Positive Rate", color=ink_secondary)
    ax.set_title("Phase 7 static baseline ROC (test split)", color=ink_primary)
    ax.legend(fontsize=9, facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)
    ax.grid(color=gridline, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(gridline)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "phase7_roc.png"), dpi=150, facecolor=surface)
    print(f"  saved plot -> {os.path.join(REPORTS_DIR, 'phase7_roc.png')}")


def part_b_class_balance_context():
    print("\n=== Part B: class-balance context for Phase 8 milestone table ===")
    ck_path = os.path.join(FEATURES_DIR, "phase8_checkpoints.csv")
    metrics_path = os.path.join(REPORTS_DIR, "phase8_dynamic_metrics.csv")

    ck = pd.read_csv(ck_path, usecols=["Issue_ID", "checkpoint_index", "split"])
    feat = pd.read_csv(
        os.path.join(FEATURES_DIR, "phase6_feature_table.csv"), usecols=["ID", "delayed"]
    )
    ck = ck.merge(feat, left_on="Issue_ID", right_on="ID", how="left")
    test_ck = ck[ck["split"] == "test"]

    rows = []
    for t, g in test_ck.groupby("checkpoint_index"):
        n = len(g)
        pos_rate = g["delayed"].mean()
        majority_class = 1 if pos_rate >= 0.5 else 0
        maj_pred = np.full(n, majority_class)
        y_true = g["delayed"].to_numpy()
        maj_f1 = f1_score(y_true, maj_pred, zero_division=0)
        maj_acc = accuracy_score(y_true, maj_pred)
        rows.append(
            {
                "checkpoint_index": t,
                "pct_delayed": pos_rate,
                "majority_class": majority_class,
                "majority_baseline_f1": maj_f1,
                "majority_baseline_accuracy": maj_acc,
            }
        )
    context_df = pd.DataFrame(rows).sort_values("checkpoint_index")

    dyn_metrics = pd.read_csv(metrics_path)
    dyn_metrics = dyn_metrics.drop(
        columns=[c for c in ["pct_delayed", "majority_class", "majority_baseline_f1",
                              "majority_baseline_accuracy"] if c in dyn_metrics.columns]
    )
    merged = dyn_metrics.merge(context_df, on="checkpoint_index", how="left")
    merged.to_csv(metrics_path, index=False)

    print(merged[["checkpoint_index", "N", "pct_delayed", "majority_baseline_f1",
                   "no_pattern_f1", "with_pattern_f1"]].to_string(index=False))
    print(f"\n  updated -> {metrics_path}")
    return merged


if __name__ == "__main__":
    part_a_train_static_baseline()
    part_b_class_balance_context()
