"""
Part A: fair static-vs-dynamic per-checkpoint comparison (same populations).
Part B: threshold tuning for Phase 7 (tuned on train, applied to test).
Part C: Phase 10 SHAP analysis (static model, dynamic with-pattern model,
importance shift across checkpoints, one local waterfall example).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)
COLOR_STATIC = "#54A24B"
COLOR_NO_PATTERN = "#2a78d6"
COLOR_WITH_PATTERN = "#eb6834"
COLOR_MAJORITY = "#898781"


def load_merged_test_checkpoints():
    ck = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_checkpoints.csv"))
    pattern = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_checkpoints_with_pattern.csv"))
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])

    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck[ck["split"] == "test"].copy()


def part_a_fair_comparison(test_ck):
    print("=== Part A: fair static-vs-dynamic per-checkpoint comparison ===")
    static_model = joblib.load(os.path.join(MODELS_DIR, "xgb_static_baseline.joblib"))

    dyn_metrics = pd.read_csv(os.path.join(REPORTS_DIR, "phase8_dynamic_metrics.csv"))
    keep_cols = [
        "checkpoint_index", "N", "no_pattern_auc", "no_pattern_f1",
        "no_pattern_precision", "no_pattern_recall",
        "with_pattern_auc", "with_pattern_f1", "with_pattern_precision", "with_pattern_recall",
    ]
    dyn_metrics = dyn_metrics[[c for c in keep_cols if c in dyn_metrics.columns]]

    rows = []
    for t, g in test_ck.groupby("checkpoint_index"):
        n = len(g)
        y_true = g["delayed"].to_numpy()

        X_static = g[STATIC_FEATURE_COLS]
        proba = static_model.predict_proba(X_static)[:, 1]
        pred = (proba >= 0.5).astype(int)
        static_auc = roc_auc_score(y_true, proba)
        static_f1 = f1_score(y_true, pred, zero_division=0)
        static_precision = precision_score(y_true, pred, zero_division=0)
        static_recall = recall_score(y_true, pred, zero_division=0)

        pos_rate = y_true.mean()
        majority_class = 1 if pos_rate >= 0.5 else 0
        maj_pred = np.full(n, majority_class)
        maj_f1 = f1_score(y_true, maj_pred, zero_division=0)
        maj_acc = accuracy_score(y_true, maj_pred)

        rows.append({
            "checkpoint_index": t,
            "pct_delayed": pos_rate,
            "majority_class": majority_class,
            "majority_baseline_f1": maj_f1,
            "majority_baseline_accuracy": maj_acc,
            "static_auc": static_auc,
            "static_f1": static_f1,
            "static_precision": static_precision,
            "static_recall": static_recall,
        })

    context_df = pd.DataFrame(rows).sort_values("checkpoint_index")
    merged = dyn_metrics.merge(context_df, on="checkpoint_index", how="left")
    out_path = os.path.join(REPORTS_DIR, "phase8_dynamic_metrics.csv")
    merged.to_csv(out_path, index=False)

    print(merged[[
        "checkpoint_index", "N", "pct_delayed", "static_auc", "no_pattern_auc",
        "with_pattern_auc", "majority_baseline_f1",
    ]].to_string(index=False))
    print(f"\n  updated -> {out_path}")

    winner_rows = []
    for _, r in merged.iterrows():
        aucs = {"static": r["static_auc"], "no_pattern": r["no_pattern_auc"], "with_pattern": r["with_pattern_auc"]}
        winner = max(aucs, key=aucs.get)
        winner_rows.append((int(r["checkpoint_index"]), winner, aucs))
    print("\n  AUC winner per checkpoint (same test population now):")
    for cidx, winner, aucs in winner_rows:
        print(f"    M{cidx}: {winner}  (static={aucs['static']:.4f}, no_pattern={aucs['no_pattern']:.4f}, "
              f"with_pattern={aucs['with_pattern']:.4f})")

    plot_three_way(merged)
    return merged


def plot_three_way(merged):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(merged["checkpoint_index"], merged["static_auc"], color=COLOR_STATIC,
            linewidth=2, marker="o", markersize=5, label="Static baseline (Phase 7)")
    ax.plot(merged["checkpoint_index"], merged["no_pattern_auc"], color=COLOR_NO_PATTERN,
            linewidth=2, marker="o", markersize=5, label="Dynamic, no pattern")
    ax.plot(merged["checkpoint_index"], merged["with_pattern_auc"], color=COLOR_WITH_PATTERN,
            linewidth=2, marker="o", markersize=5, label="Dynamic, with pattern")
    ax.axhline(0.5, color=COLOR_MAJORITY, linewidth=1.5, linestyle="--",
               label="Random / majority-class reference (AUC=0.5)")

    ax.set_xlabel("Checkpoint index (1-10)", color=INK_SECONDARY)
    ax.set_ylabel("AUC", color=INK_SECONDARY)
    ax.set_title("Static vs. dynamic AUC on identical per-checkpoint test populations", color=INK_PRIMARY)
    ax.set_xticks(range(1, 11))
    ax.tick_params(colors=INK_MUTED)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_PRIMARY, loc="lower right")

    ax2 = ax.twinx()
    ax2.bar(merged["checkpoint_index"], merged["N"], color=INK_MUTED, alpha=0.12, width=0.5)
    ax2.set_ylabel("N (test issues)", color=INK_MUTED, fontsize=8)
    ax2.tick_params(colors=INK_MUTED, labelsize=7)

    fig.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "phase8_checkpoint_accuracy.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"  saved -> {out_path}")


def part_b_threshold_tuning():
    print("\n=== Part B: threshold tuning for Phase 7 ===")
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    issue = pd.read_csv("data/raw/issue.csv", usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left")
    feat = feat.sort_values("Creation_Date", kind="mergesort").reset_index(drop=True)

    phase8_split = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_split.csv"))
    test_ids = set(phase8_split.loc[phase8_split["split"] == "test", "ID"])
    feat["split"] = np.where(feat["ID"].isin(test_ids), "test", "train")

    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    train = feat[feat["split"] == "train"]
    test = feat[feat["split"] == "test"]

    clf = joblib.load(os.path.join(MODELS_DIR, "xgb_static_baseline.joblib"))
    train_proba = clf.predict_proba(train[STATIC_FEATURE_COLS])[:, 1]
    test_proba = clf.predict_proba(test[STATIC_FEATURE_COLS])[:, 1]
    y_train, y_test = train["delayed"].to_numpy(), test["delayed"].to_numpy()

    thresholds = np.linspace(0.01, 0.99, 197)
    best_thresh, best_f1 = 0.5, -1
    for thresh in thresholds:
        f1 = f1_score(y_train, (train_proba >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    pred_default = (test_proba >= 0.5).astype(int)
    pred_tuned = (test_proba >= best_thresh).astype(int)

    tuned_metrics = {
        "threshold": float(best_thresh),
        "train_f1_at_threshold": float(best_f1),
        "test_f1": float(f1_score(y_test, pred_tuned, zero_division=0)),
        "test_precision": float(precision_score(y_test, pred_tuned, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred_tuned, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, pred_tuned)),
    }
    default_metrics = {
        "threshold": 0.5,
        "test_f1": float(f1_score(y_test, pred_default, zero_division=0)),
        "test_precision": float(precision_score(y_test, pred_default, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred_default, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, pred_default)),
    }

    print(f"  tuned threshold (max F1 on TRAIN): {best_thresh:.3f}")
    print(f"  test @ 0.5:    F1={default_metrics['test_f1']:.4f} P={default_metrics['test_precision']:.4f} "
          f"R={default_metrics['test_recall']:.4f} acc={default_metrics['test_accuracy']:.4f}")
    print(f"  test @ tuned:  F1={tuned_metrics['test_f1']:.4f} P={tuned_metrics['test_precision']:.4f} "
          f"R={tuned_metrics['test_recall']:.4f} acc={tuned_metrics['test_accuracy']:.4f}")

    metrics_path = os.path.join(REPORTS_DIR, "phase7_metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["threshold_tuning"] = {"default_threshold_0.5": default_metrics, "tuned_on_train": tuned_metrics}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  updated -> {metrics_path}")
    return tuned_metrics


def part_c_shap(test_ck):
    print("\n=== Part C: Phase 10 SHAP ===")
    rng = np.random.default_rng(42)

    static_model = joblib.load(os.path.join(MODELS_DIR, "xgb_static_baseline.joblib"))
    dynamic_model = joblib.load(os.path.join(MODELS_DIR, "xgb_dynamic_with_pattern.joblib"))
    dyn_feature_cols = STATIC_FEATURE_COLS + ["log1p_stall", "elapsed_minutes", "pattern_label"]

    # --- 1. Global static summary ---
    print("  [1] static model global SHAP summary ...")
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    issue = pd.read_csv("data/raw/issue.csv", usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left").sort_values("Creation_Date", kind="mergesort")
    phase8_split = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_split.csv"))
    test_ids = set(phase8_split.loc[phase8_split["split"] == "test", "ID"])
    static_test = feat[feat["ID"].isin(test_ids)].copy()
    for col in CATEGORICAL_COLS:
        static_test[col] = static_test[col].astype("category")

    sample_idx = rng.choice(len(static_test), size=min(8000, len(static_test)), replace=False)
    X_static_sample = static_test.iloc[sample_idx][STATIC_FEATURE_COLS]

    explainer_static = shap.TreeExplainer(static_model)
    shap_static = explainer_static(X_static_sample)

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    shap.summary_plot(shap_static, X_static_sample, show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "phase10_shap_static_summary.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    static_importance = pd.Series(
        np.abs(shap_static.values).mean(axis=0), index=STATIC_FEATURE_COLS
    ).sort_values(ascending=False)
    print("    static model top features (mean |SHAP|):")
    print(static_importance.to_string())

    # --- 2. Global dynamic summary ---
    print("\n  [2] dynamic (with-pattern) model global SHAP summary ...")
    sample_idx2 = rng.choice(len(test_ck), size=min(8000, len(test_ck)), replace=False)
    X_dyn_sample = test_ck.iloc[sample_idx2][dyn_feature_cols]

    explainer_dyn = shap.TreeExplainer(dynamic_model)
    shap_dyn = explainer_dyn(X_dyn_sample)

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    shap.summary_plot(shap_dyn, X_dyn_sample, show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "phase10_shap_dynamic_summary.png"), dpi=150, facecolor=SURFACE)
    plt.close()

    dyn_importance = pd.Series(
        np.abs(shap_dyn.values).mean(axis=0), index=dyn_feature_cols
    ).sort_values(ascending=False)
    print("    dynamic model top features (mean |SHAP|):")
    print(dyn_importance.to_string())
    for feature in ["pattern_label", "log1p_stall", "elapsed_minutes"]:
        rank = list(dyn_importance.index).index(feature) + 1
        print(f"    '{feature}' ranks #{rank} of {len(dyn_importance)}")

    # --- 3. Importance shift early vs late ---
    print("\n  [3] importance shift M1-M3 vs M8-M10 ...")
    early = test_ck[test_ck["checkpoint_index"].isin([1, 2, 3])]
    late = test_ck[test_ck["checkpoint_index"].isin([8, 9, 10])]

    early_idx = rng.choice(len(early), size=min(4000, len(early)), replace=False)
    late_idx = rng.choice(len(late), size=min(4000, len(late)), replace=False)
    X_early = early.iloc[early_idx][dyn_feature_cols]
    X_late = late.iloc[late_idx][dyn_feature_cols]

    shap_early = explainer_dyn(X_early)
    shap_late = explainer_dyn(X_late)

    imp_early = pd.Series(np.abs(shap_early.values).mean(axis=0), index=dyn_feature_cols)
    imp_late = pd.Series(np.abs(shap_late.values).mean(axis=0), index=dyn_feature_cols)
    shift_df = pd.DataFrame({"early_M1_M3": imp_early, "late_M8_M10": imp_late})
    shift_df["delta_late_minus_early"] = shift_df["late_M8_M10"] - shift_df["early_M1_M3"]
    shift_df = shift_df.sort_values("delta_late_minus_early", ascending=False)
    print(shift_df.to_string())

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    order = shift_df.index
    y_pos = np.arange(len(order))
    ax.barh(y_pos - 0.18, shift_df.loc[order, "early_M1_M3"], height=0.36, color=COLOR_NO_PATTERN, label="M1-M3 (early)")
    ax.barh(y_pos + 0.18, shift_df.loc[order, "late_M8_M10"], height=0.36, color=COLOR_WITH_PATTERN, label="M8-M10 (late)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, color=INK_SECONDARY)
    ax.set_xlabel("mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title("Feature importance shift: early vs. late checkpoints", color=INK_PRIMARY)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(color=GRIDLINE, linewidth=0.8, axis="x")
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.legend(fontsize=9, facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "phase10_shap_importance_shift.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"    saved -> {os.path.join(REPORTS_DIR, 'phase10_shap_importance_shift.png')}")

    # --- 4. Local example ---
    print("\n  [4] local waterfall example ...")
    proba_all = dynamic_model.predict_proba(test_ck[dyn_feature_cols])[:, 1]
    pred_all = (proba_all >= 0.5).astype(int)
    correct_delayed_mask = (pred_all == 1) & (test_ck["delayed"].to_numpy() == 1)
    candidates = test_ck.index[correct_delayed_mask]
    chosen_idx = rng.choice(candidates, size=1)[0]
    chosen_row = test_ck.loc[[chosen_idx], dyn_feature_cols]
    chosen_shap = explainer_dyn(chosen_row)

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    shap.plots.waterfall(chosen_shap[0], show=False)
    plt.gcf().set_facecolor(SURFACE)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "phase10_shap_local_example.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print(f"    issue_id={test_ck.loc[chosen_idx, 'Issue_ID']} "
          f"checkpoint={test_ck.loc[chosen_idx, 'checkpoint_index']} "
          f"predicted_proba={proba_all[test_ck.index.get_loc(chosen_idx)]:.3f} "
          f"-> saved {os.path.join(REPORTS_DIR, 'phase10_shap_local_example.png')}")

    return static_importance, dyn_importance, shift_df


if __name__ == "__main__":
    test_ck = load_merged_test_checkpoints()
    part_a_fair_comparison(test_ck)
    part_b_threshold_tuning()
    part_c_shap(test_ck)
