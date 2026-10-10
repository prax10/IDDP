"""
Task G -- Model comparison benchmark. Empirical justification for choosing
XGBoost, scoped to the STATIC (Phase 7) problem only: one row per issue,
data/features/phase6_feature_table_v3.csv, delayed_v2 label, the existing
chronological split (data/features/phase8_split.csv). No dynamic/checkpoint
work here -- that pipeline is not rebuilt.

Fairness protocol (chosen over the "defaults only" alternative): logistic
regression, random forest, and LightGBM each get a small grid, selected by
validation AUC only (never test) -- mirroring the exact protocol Task B4
used to tune XGBoost itself, not sklearn's cross-validated GridSearchCV,
since k-fold CV would blur the chronological train/val boundary this whole
project is built to respect.

XGBoost's own hyperparameters are NOT re-selected here (already tuned via
that same validation protocol in Task B4/B5) -- it is retrained once with
its locked config so its measured training time is comparable to the other
five models' fresh-run times in this same script/session.
"""

import time
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

import demo_data as dd

OUT_DIR = "model_comparison"
FEATURES_DIR = "data/features"
LABEL = "delayed_v2"
FEATURE_COLS = dd.STATIC_FEATURE_COLS_V2
CATEGORICAL_COLS = dd.CATEGORICAL_COLS
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

XGB_PARAMS = dict(learning_rate=0.2, max_depth=4, n_estimators=42,
                   subsample=0.8, colsample_bytree=0.8,
                   enable_categorical=True, tree_method="hist",
                   eval_metric="logloss")


def load_data():
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv", usecols=["ID", "split"])
    df = feat.merge(split, on="ID", how="inner")
    assert LABEL in df.columns, f"{LABEL} missing from phase6_feature_table_v3.csv"
    return df


def xgb_ready(df):
    """XGBoost/LightGBM-native frame: categorical dtype, NaN untouched."""
    out = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        out[c] = out[c].astype("category")
    for c in NUMERIC_COLS:
        out[c] = out[c].astype(float)
    return out


def sklearn_preprocessor():
    """Median-impute + scale numeric, one-hot the 3 categoricals. Fit only
    on whatever split is passed to .fit_transform at each stage -- never
    fit on test, and never fit on val+test when selecting on val."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def metrics_row(name, y_true, proba, train_seconds, imputation_required,
                 tuning_protocol, best_config):
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "auc": roc_auc_score(y_true, proba),
        "accuracy_at_0.5": accuracy_score(y_true, pred),
        "f1_at_0.5": f1_score(y_true, pred),
        "precision_at_0.5": precision_score(y_true, pred),
        "recall_at_0.5": recall_score(y_true, pred),
        "train_seconds": train_seconds,
        "imputation_required": imputation_required,
        "tuning_protocol": tuning_protocol,
        "best_config": best_config,
    }


def grid_select_sklearn(model_fn, grid, X_train, y_train, X_val, y_val):
    """Fit each grid config on train only, score on val only, return the
    config with the best validation AUC (never touches test)."""
    best_auc, best_cfg = -1.0, None
    for cfg in grid:
        model = model_fn(**cfg)
        model.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if val_auc > best_auc:
            best_auc, best_cfg = val_auc, cfg
    return best_cfg, best_auc


def main():
    print("Loading data...")
    df = load_data()
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    trainval_mask = df["split"].isin(["train", "val"])
    test_mask = df["split"] == "test"
    print(f"train={train_mask.sum()}  val={val_mask.sum()}  "
          f"train+val={trainval_mask.sum()}  test={test_mask.sum()}")

    y = df[LABEL].astype(int)
    n_missing_assignee_features = df.loc[
        trainval_mask | test_mask, ["assignee_prior_resolved_count", "assignee_prior_delay_rate",
                                     "assignee_concurrent_open"]
    ].isna().any(axis=1).sum()
    print(f"Rows with >=1 NaN assignee feature (train+val+test): {n_missing_assignee_features}")

    results = []

    # ---------------- 1. Majority-class baseline ----------------
    majority_class = int(y[trainval_mask].mean() >= 0.5)
    proba_majority = np.full(test_mask.sum(), float(majority_class))
    # A constant-score classifier has a degenerate ROC curve (a single
    # point) -- AUC is undefined; report 0.5 by convention (the diagonal
    # a truly uninformative ranking traces) and let accuracy carry the
    # real "floor" signal for this model.
    results.append(metrics_row("Majority-class baseline", y[test_mask], proba_majority,
                                train_seconds=0.0, imputation_required=False,
                                tuning_protocol="none (no parameters)", best_config="predict majority class"))

    # ---------------- sklearn-preprocessed matrices ----------------
    X_sk_train = df.loc[train_mask, FEATURE_COLS]
    X_sk_val = df.loc[val_mask, FEATURE_COLS]
    X_sk_trainval = df.loc[trainval_mask, FEATURE_COLS]
    X_sk_test = df.loc[test_mask, FEATURE_COLS]
    y_train, y_val, y_trainval, y_test = y[train_mask], y[val_mask], y[trainval_mask], y[test_mask]

    pre_grid = sklearn_preprocessor()
    Xg_train = pre_grid.fit_transform(X_sk_train)
    Xg_val = pre_grid.transform(X_sk_val)

    pre_final = sklearn_preprocessor()
    Xf_trainval = pre_final.fit_transform(X_sk_trainval)
    Xf_test = pre_final.transform(X_sk_test)

    # ---------------- 2. Logistic regression ----------------
    print("Logistic regression: validation grid over C...")
    lr_grid = [{"C": c, "max_iter": 1000} for c in [0.01, 0.1, 1.0, 10.0]]
    best_cfg, val_auc = grid_select_sklearn(LogisticRegression, lr_grid, Xg_train, y_train, Xg_val, y_val)
    print(f"  selected {best_cfg}, val AUC={val_auc:.4f}")
    t0 = time.time()
    lr_final = LogisticRegression(**best_cfg)
    lr_final.fit(Xf_trainval, y_trainval)
    lr_seconds = time.time() - t0
    lr_proba = lr_final.predict_proba(Xf_test)[:, 1]
    results.append(metrics_row("Logistic regression", y_test, lr_proba, lr_seconds,
                                imputation_required=True, tuning_protocol="validation grid (C)",
                                best_config=json.dumps(best_cfg)))

    # ---------------- 3. Random forest ----------------
    print("Random forest: validation grid over max_depth...")
    rf_grid = [{"max_depth": d, "n_estimators": 300, "n_jobs": -1, "random_state": 0}
               for d in [4, 8, 12, None]]
    best_cfg, val_auc = grid_select_sklearn(RandomForestClassifier, rf_grid, Xg_train, y_train, Xg_val, y_val)
    print(f"  selected {best_cfg}, val AUC={val_auc:.4f}")
    t0 = time.time()
    rf_final = RandomForestClassifier(**best_cfg)
    rf_final.fit(Xf_trainval, y_trainval)
    rf_seconds = time.time() - t0
    rf_proba = rf_final.predict_proba(Xf_test)[:, 1]
    results.append(metrics_row("Random forest", y_test, rf_proba, rf_seconds,
                                imputation_required=True, tuning_protocol="validation grid (max_depth)",
                                best_config=json.dumps({k: v for k, v in best_cfg.items() if k == "max_depth"})))

    # ---------------- 4. XGBoost (already-tuned config, retrained fresh for timing) ----------------
    print("XGBoost: retraining locked config for a comparable timing measurement...")
    X_xgb_trainval = xgb_ready(df.loc[trainval_mask])
    X_xgb_test = xgb_ready(df.loc[test_mask])
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_xgb_trainval, y_trainval)
    xgb_seconds = time.time() - t0
    xgb_proba = xgb_model.predict_proba(X_xgb_test)[:, 1]
    results.append(metrics_row("XGBoost (tuned, Task B4/B5)", y_test, xgb_proba, xgb_seconds,
                                imputation_required=False,
                                tuning_protocol="validation grid, selected previously (Task B4/B5) -- not re-tuned here",
                                best_config=json.dumps(XGB_PARAMS)))

    # ---------------- 5. LightGBM ----------------
    print("LightGBM: validation grid over learning_rate/max_depth...")
    lgb_grid = [{"learning_rate": lr, "max_depth": d, "n_estimators": 100, "verbosity": -1}
                for lr in [0.05, 0.1, 0.2] for d in [4, 6, 8]]
    X_lgb_train = xgb_ready(df.loc[train_mask])
    X_lgb_val = xgb_ready(df.loc[val_mask])
    best_auc, best_cfg = -1.0, None
    for cfg in lgb_grid:
        m = lgb.LGBMClassifier(**cfg)
        m.fit(X_lgb_train, y_train)
        val_auc = roc_auc_score(y_val, m.predict_proba(X_lgb_val)[:, 1])
        if val_auc > best_auc:
            best_auc, best_cfg = val_auc, cfg
    print(f"  selected {best_cfg}, val AUC={best_auc:.4f}")
    t0 = time.time()
    lgb_final = lgb.LGBMClassifier(**best_cfg)
    lgb_final.fit(X_xgb_trainval, y_trainval)
    lgb_seconds = time.time() - t0
    lgb_proba = lgb_final.predict_proba(X_xgb_test)[:, 1]
    results.append(metrics_row("LightGBM (tuned)", y_test, lgb_proba, lgb_seconds,
                                imputation_required=False, tuning_protocol="validation grid (learning_rate, max_depth)",
                                best_config=json.dumps(best_cfg)))

    # Secondary, non-grid comparison point: LightGBM given XGBoost's exact
    # hyperparameters, to isolate "library" from "tuning" specifically --
    # kept separate from the primary (tuned) LightGBM row above.
    print("LightGBM (secondary): XGBoost's exact hyperparameters, library-isolation check...")
    lgb_matched_params = dict(learning_rate=XGB_PARAMS["learning_rate"], max_depth=XGB_PARAMS["max_depth"],
                              n_estimators=XGB_PARAMS["n_estimators"], subsample=XGB_PARAMS["subsample"],
                              colsample_bytree=XGB_PARAMS["colsample_bytree"], verbosity=-1)
    t0 = time.time()
    lgb_matched = lgb.LGBMClassifier(**lgb_matched_params)
    lgb_matched.fit(X_xgb_trainval, y_trainval)
    lgb_matched_seconds = time.time() - t0
    lgb_matched_proba = lgb_matched.predict_proba(X_xgb_test)[:, 1]
    results.append(metrics_row("LightGBM (XGBoost's exact hyperparams -- library isolation only)",
                                y_test, lgb_matched_proba, lgb_matched_seconds,
                                imputation_required=False,
                                tuning_protocol="none -- deliberately matched to XGBoost's config, not independently tuned",
                                best_config=json.dumps(lgb_matched_params)))

    # ---------------- 6. Small MLP ----------------
    print("MLP: fitting (64,32) with early stopping...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                         max_iter=200, random_state=0)
    mlp.fit(Xf_trainval, y_trainval)
    mlp_seconds = time.time() - t0
    capped = mlp_seconds > 600
    mlp_proba = mlp.predict_proba(Xf_test)[:, 1]
    results.append(metrics_row("MLP (64,32)", y_test, mlp_proba, mlp_seconds,
                                imputation_required=True,
                                tuning_protocol="defaults (not in the small-grid scope; capped at 10 min per brief)"
                                                + (" -- CAPPED, see note" if capped else ""),
                                best_config=json.dumps({"hidden_layer_sizes": [64, 32], "early_stopping": True})))

    # ---------------- Save ----------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print(results_df[["model", "auc", "accuracy_at_0.5", "f1_at_0.5", "train_seconds"]]
          .to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df = results_df[results_df["model"] != "LightGBM (XGBoost's exact hyperparams -- library isolation only)"]
    order = plot_df.sort_values("auc", ascending=True)
    colors = ["#8a8f98" if "Majority" in m else ("#0d7d8a" if "XGBoost" in m else "#4C78A8")
              for m in order["model"]]
    ax.barh(order["model"], order["auc"], color=colors)
    ax.axvline(0.5, color="#8a8f98", linestyle="--", linewidth=1.2, label="Majority-class floor (AUC = 0.5)")
    ax.set_xlabel("Test AUC")
    ax.set_xlim(0.45, max(0.85, order["auc"].max() + 0.03))
    ax.legend(loc="lower right", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    print(f"\nSaved {OUT_DIR}/model_comparison.csv and {OUT_DIR}/model_comparison.png")

    return results_df, n_missing_assignee_features


if __name__ == "__main__":
    main()
