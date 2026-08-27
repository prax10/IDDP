"""
Task A, Step 4: re-run Phase 7 static baseline with text features added,
same chronological split (aligned to Phase 8's test split) and same
hyperparameters as the original Phase 7.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
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
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]


def main():
    print("=== Phase 7 v2: static baseline WITH text features ===")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    issue = pd.read_csv("data/raw/issue.csv", usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left")
    feat = feat.sort_values("Creation_Date", kind="mergesort").reset_index(drop=True)

    phase8_split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    test_ids = set(phase8_split.loc[phase8_split["split"] == "test", "ID"])
    own_test_ids = set(feat.iloc[int(round(len(feat) * 0.8)):]["ID"])
    if own_test_ids == test_ids:
        print(f"  Verified: 80/20 test split matches Phase 8's test split ({len(test_ids)} issues).")
        feat["split"] = np.where(feat["ID"].isin(test_ids), "test", "train")
    else:
        print("  MISMATCH: falling back to Phase 8's test split directly.")
        feat["split"] = np.where(feat["ID"].isin(test_ids), "test", "train")

    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    train = feat[feat["split"] == "train"]
    test = feat[feat["split"] == "test"]
    print(f"  train={len(train)} test={len(test)}")

    X_train, y_train = train[STATIC_FEATURE_COLS_V2], train["delayed"]
    X_test, y_test = test[STATIC_FEATURE_COLS_V2], test["delayed"]

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

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "features": STATIC_FEATURE_COLS_V2,
        "model": {"auc": auc, "f1": f1, "precision": precision, "recall": recall,
                  "accuracy": acc, "confusion_matrix": cm},
        "majority_baseline": {"majority_class": int(majority_class), "f1": maj_f1, "accuracy": maj_acc},
        "hyperparameters": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                             "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss"},
        "baseline_v1_auc_no_text": 0.7242,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, f"{MODELS_DIR}/xgb_static_v2.joblib")
    with open(f"{REPORTS_DIR}/phase7_metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  AUC={auc:.4f} (v1 baseline without text: 0.7242, delta={auc-0.7242:+.4f})")
    print(f"  F1={f1:.4f} precision={precision:.4f} recall={recall:.4f} acc={acc:.4f}")
    print(f"  majority baseline: F1={maj_f1:.4f} acc={maj_acc:.4f}")
    print(f"  saved -> {MODELS_DIR}/xgb_static_v2.joblib, {REPORTS_DIR}/phase7_metrics_v2.json")


if __name__ == "__main__":
    main()
