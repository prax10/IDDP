"""
Task B3, Step 2.1: re-run Phase 7 static baseline with delayed_v2 labels
(train+val-only medians), v2 (with-text) features, same hyperparameters
and split.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
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
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]


def main():
    print("=== Task B3: Phase 7 v3 (delayed_v2 labels) ===")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    train = feat[feat["split"].isin(["train", "val"])]
    test = feat[feat["split"] == "test"]
    print(f"  train={len(train)} test={len(test)}")

    X_train, y_train = train[STATIC_FEATURE_COLS_V2], train["delayed_v2"]
    X_test, y_test = test[STATIC_FEATURE_COLS_V2], test["delayed_v2"]

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        enable_categorical=True, tree_method="hist",
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1_default = f1_score(y_test, pred_default)
    majority_class = y_train.mode()[0]
    maj_f1 = f1_score(y_test, np.full(len(y_test), majority_class), zero_division=0)
    maj_acc = accuracy_score(y_test, np.full(len(y_test), majority_class))

    # train-tuned threshold (leakage-safe, tuned on train, applied to test)
    train_proba = clf.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 197)
    best_thresh, best_f1 = 0.5, -1
    for thresh in thresholds:
        f1 = f1_score(y_train, (train_proba >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    pred_tuned = (proba >= best_thresh).astype(int)

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "auc": auc,
        "f1_at_0.5": f1_default,
        "precision_at_0.5": precision_score(y_test, pred_default),
        "recall_at_0.5": recall_score(y_test, pred_default),
        "tuned_threshold": float(best_thresh),
        "f1_at_tuned_threshold": f1_score(y_test, pred_tuned),
        "precision_at_tuned_threshold": precision_score(y_test, pred_tuned),
        "recall_at_tuned_threshold": recall_score(y_test, pred_tuned),
        "majority_baseline_f1": maj_f1,
        "majority_baseline_accuracy": maj_acc,
    }

    joblib.dump(clf, f"{MODELS_DIR}/xgb_static_v3.joblib")
    with open(f"{REPORTS_DIR}/phase7_metrics_v3.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  AUC={auc:.4f} (v2/old-label: 0.7343)")
    print(f"  F1@0.5={f1_default:.4f}  F1@tuned({best_thresh:.3f})={metrics['f1_at_tuned_threshold']:.4f}")
    print(f"  majority baseline: F1={maj_f1:.4f} acc={maj_acc:.4f}")
    print(f"  saved -> {MODELS_DIR}/xgb_static_v3.joblib, {REPORTS_DIR}/phase7_metrics_v3.json")


if __name__ == "__main__":
    main()
