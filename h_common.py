"""
Shared helpers for Task H (future-work items #1-#3).

Loading mirrors task_b4_validation_tuning.py / task_b5_final_verification.py
exactly (same files, same merges, same feature lists, same XGBoost FIXED
params) so numbers are directly comparable with the existing reports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, brier_score_loss, log_loss,
)

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

STATIC_BASE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]
DYN_EXTRA_COLS = ["log1p_stall", "elapsed_minutes", "pattern_label"]
LABEL = "delayed_v2"


def text_cols(k=30):
    return [f"text_pc_{i}" for i in range(k)]


def static_cols(k=30):
    return STATIC_BASE_COLS + text_cols(k)


def dyn_cols(k=30):
    return static_cols(k) + DYN_EXTRA_COLS


GRID = [
    {"learning_rate": lr, "max_depth": d}
    for lr in [0.05, 0.1, 0.2]
    for d in [4, 6, 8]
]
FIXED = dict(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
             enable_categorical=True, tree_method="hist", random_state=42)

# Selected configs already in the paper (task_b4)
UNTUNED_DYN = dict(learning_rate=0.05, max_depth=6)          # task_b3_phase8, 400 cap, ES=20
TUNED_DYN = dict(learning_rate=0.1, max_depth=8, n_estimators=11)
TUNED_STATIC = dict(learning_rate=0.2, max_depth=4, n_estimators=42)


def _feature_table(text_override=None):
    """phase6 v3 table; optionally swap its 30 text PCs for a different set."""
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    if text_override is not None:
        feat = feat.drop(columns=[c for c in feat.columns if c.startswith("text_pc_")])
        feat = feat.merge(text_override, on="ID", how="left")
        tc = [c for c in text_override.columns if c.startswith("text_pc_")]
        assert feat[tc].isna().sum().sum() == 0, "text merge produced NaNs"
    tc = [c for c in feat.columns if c.startswith("text_pc_")]
    feat[tc] = feat[tc].astype("float32")
    return feat


def load_static(text_override=None):
    feat = _feature_table(text_override)
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")
    return feat


def load_dynamic(text_override=None):
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    ck = ck.drop(columns=["current_status", "last_status_change_time"], errors="ignore")
    feat = _feature_table(text_override)
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck


def metrics(y, p, prefix=""):
    """Full metric panel. Threshold-based metrics at 0.5."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    maj = 1 if y.mean() >= 0.5 else 0
    out = {
        "N": len(y),
        "pos_rate": y.mean(),
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, yhat),
        "majority_accuracy": accuracy_score(y, np.full(len(y), maj)),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


def fast_auc(y, p):
    """Rank-based AUC (Mann-Whitney), ~10x faster than sklearn for bootstrap loops."""
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.empty(len(p), dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(p_sorted, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    ranks[order] = avg[inv]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def paired_bootstrap(y, p_a, p_b, n_boot=1000, seed=0):
    """CI on AUC(a) - AUC(b) and accuracy(a) - accuracy(b), resampling test rows."""
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    acc_a = ((p_a >= 0.5).astype(int) == y).astype(float)
    acc_b = ((p_b >= 0.5).astype(int) == y).astype(float)
    d_auc, d_acc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        d_auc[i] = fast_auc(yy, p_a[idx]) - fast_auc(yy, p_b[idx])
        d_acc[i] = acc_a[idx].mean() - acc_b[idx].mean()
    point_auc = fast_auc(y, p_a) - fast_auc(y, p_b)
    return {
        "d_auc": point_auc,
        "d_auc_lo": np.percentile(d_auc, 2.5),
        "d_auc_hi": np.percentile(d_auc, 97.5),
        "d_acc": acc_a.mean() - acc_b.mean(),
        "d_acc_lo": np.percentile(d_acc, 2.5),
        "d_acc_hi": np.percentile(d_acc, 97.5),
    }


SEEDS = [0, 1, 2, 3, 4]


def seeded(seed):
    f = dict(FIXED)
    f["random_state"] = seed
    return f


def paired_bootstrap_seeds(y, PA, PB, n_boot=1000, seed=0):
    """
    Paired bootstrap of  mean_s AUC(A_s) - mean_s AUC(B_s)  over test rows.
    PA / PB: lists of probability vectors, one per training seed. Averaging
    over seeds removes training-RNG noise from the point estimate (XGBoost
    row subsampling alone moves single-run M10 AUC by ~0.02); resampling
    test rows gives the CI. Accuracy difference is reported the same way.
    """
    y = np.asarray(y).astype(float)
    PA = [np.asarray(p, float) for p in PA]
    PB = [np.asarray(p, float) for p in PB]
    accA = np.mean([((p >= 0.5).astype(int) == y) for p in PA], axis=0)
    accB = np.mean([((p >= 0.5).astype(int) == y) for p in PB], axis=0)
    # Pre-rank each prediction vector once; a bootstrap resample is then just
    # a vector of row counts, and AUC becomes a weighted O(n) bincount.
    invs = [np.unique(p, return_inverse=True)[1] for p in PA + PB]
    rng = np.random.default_rng(seed)
    n = len(y)

    def wauc(inv, w):
        U = inv.max() + 1
        pos = np.bincount(inv, weights=w * y, minlength=U)
        neg = np.bincount(inv, weights=w * (1 - y), minlength=U)
        below = np.cumsum(neg) - neg
        return (pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())

    nA = len(PA)
    d_auc, d_acc = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a = [wauc(inv, w) for inv in invs]
        d_auc[i] = np.mean(a[:nA]) - np.mean(a[nA:])
        d_acc[i] = (w @ (accA - accB)) / n
    ones = np.ones(n)
    a = [wauc(inv, ones) for inv in invs]
    point = np.mean(a[:nA]) - np.mean(a[nA:])
    return {
        "d_auc": point, "d_auc_lo": np.percentile(d_auc, 2.5), "d_auc_hi": np.percentile(d_auc, 97.5),
        "d_acc": accA.mean() - accB.mean(),
        "d_acc_lo": np.percentile(d_acc, 2.5), "d_acc_hi": np.percentile(d_acc, 97.5),
    }


def seed_metric_panel(y, P, **tags):
    """Mean and SD across seeds of every metric in `metrics`."""
    per = pd.DataFrame([metrics(y, p) for p in P])
    row = dict(tags)
    for c in per.columns:
        row[c] = per[c].mean()
        if c not in ("N", "pos_rate", "majority_accuracy"):
            row[f"{c}_sd"] = per[c].std(ddof=1) if len(per) > 1 else 0.0
    return row


def verdict(lo, hi, margin=0.01):
    if lo > 0:
        return "A better (CI > 0)"
    if hi < 0:
        return "B better (CI < 0)"
    if lo >= -margin and hi <= margin:
        return f"equivalent (CI within +/-{margin})"
    return "inconclusive (CI spans 0)"


def per_checkpoint_auc(test_ck, proba):
    rows = []
    for t in range(1, 11):
        m = (test_ck["checkpoint_index"] == t).to_numpy()
        rows.append(roc_auc_score(test_ck[LABEL].to_numpy()[m], proba[m]))
    return rows
