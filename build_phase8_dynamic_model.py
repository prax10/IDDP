"""
Phase 8.4-8.6: dynamic pattern assignment (leakage-safe, checkpoint-by-
checkpoint), dynamic XGBoost models with/without pattern_label, and
milestone-wise (per-checkpoint) evaluation on the test split.

Stops after Step 7 -- Phase 9/10 are a separate, later task.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"

DYNAMIC_METRICS_OUT = os.path.join(REPORTS_DIR, "phase8_dynamic_metrics.csv")
ACCURACY_PLOT = os.path.join(REPORTS_DIR, "phase8_checkpoint_accuracy.png")

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]

XGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    enable_categorical=True,
    tree_method="hist",
    early_stopping_rounds=20,
    random_state=42,
)


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading inputs ...")
    ck = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_checkpoints.csv"))
    medoids_df = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_medoids.csv"))
    split = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_split.csv"))
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    print(f"  checkpoints: {len(ck)} rows, {ck['Issue_ID'].nunique()} issues")

    # ------------------------------------------------------------------
    # Step 1: log1p everywhere (fixed transform, no fitted params -> safe
    # to apply to train/val/test alike)
    # ------------------------------------------------------------------
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    print(f"\nStep 1: log1p applied to all {len(ck)} rows "
          f"({ck['log1p_stall'].isna().sum()} remain NaN, from missing dwell baseline)")

    # ------------------------------------------------------------------
    # Step 2: dynamic pattern assignment
    # ------------------------------------------------------------------
    print("\nStep 2: dynamic pattern assignment against the 4 saved medoids ...")
    print(f"  medoid issue IDs (unchanged from 8.3, training-subsample only): "
          f"{sorted(medoids_df['medoid_issue_id'].unique().tolist())}")

    medoids = {}
    for c, g in medoids_df.groupby("cluster"):
        medoids[c] = g.sort_values("checkpoint_index")["log1p_stall_ratio"].to_numpy(
            dtype=np.double
        ).copy()

    ck = ck.sort_values(["Issue_ID", "checkpoint_index"]).reset_index(drop=True)
    pattern_labels = np.empty(len(ck), dtype=object)

    for _, idx in ck.groupby("Issue_ID", sort=False).indices.items():
        prefix = []
        for i in idx:  # idx already in checkpoint_index order (frame pre-sorted)
            cidx = ck.at[i, "checkpoint_index"]
            val = ck.at[i, "log1p_stall"]
            if cidx <= 2:
                pattern_labels[i] = "insufficient_history"
                continue
            if not np.isnan(val):
                prefix.append(val)
            if len(prefix) == 0:
                pattern_labels[i] = "insufficient_history"
                continue
            arr = np.array(prefix, dtype=np.double)
            dists = {c: dtw.distance_fast(arr, m) for c, m in medoids.items()}
            best = min(dists, key=dists.get)
            pattern_labels[i] = f"cluster_{best}"

    ck["pattern_label"] = pattern_labels
    print(ck["pattern_label"].value_counts())

    # ------------------------------------------------------------------
    # Step 3: verification
    # ------------------------------------------------------------------
    print("\n=== Step 3: verification ===")
    print(
        "  Medoids used exactly as saved from 8.3 (loaded directly from "
        "phase8_medoids.csv, no recomputation) -- val/test never touched them."
    )

    print("\n  Spot-checks (pattern at checkpoint t uses only 1..t):")
    rng = np.random.default_rng(7)
    candidates = ck[ck["checkpoint_index"] >= 5]["Issue_ID"].unique()
    spot_ids = rng.choice(candidates, size=3, replace=False)
    for iid in spot_ids:
        g = ck[ck["Issue_ID"] == iid].sort_values("checkpoint_index")
        t = g["checkpoint_index"].max()
        # recompute independently, feeding ONLY rows with checkpoint_index<=t
        truncated = g[g["checkpoint_index"] <= t]
        prefix = [v for v in truncated["log1p_stall"] if not np.isnan(v)]
        if len(prefix) == 0:
            expected = "insufficient_history"
        else:
            arr = np.array(prefix, dtype=np.double)
            dists = {c: dtw.distance_fast(arr, m) for c, m in medoids.items()}
            expected = f"cluster_{min(dists, key=dists.get)}"
        actual = g[g["checkpoint_index"] == t]["pattern_label"].iloc[0]
        print(
            f"    issue={iid} t={t} recomputed-from-1..t={expected} "
            f"stored={actual} -> {'MATCH' if expected == actual else 'MISMATCH'}"
        )

    print("\n  Pattern-label stability across an issue's own checkpoints:")
    real_labels = ck[ck["checkpoint_index"] >= 3].copy()
    n_changes_per_issue = real_labels.groupby("Issue_ID")["pattern_label"].apply(
        lambda s: (s.values[1:] != s.values[:-1]).sum()
    )
    print(f"    issues with >=1 checkpoint from M3 on: {len(n_changes_per_issue)}")
    print(f"    mean label changes per issue: {n_changes_per_issue.mean():.2f}")
    print(f"    % of issues whose label never changes after first assignment: "
          f"{(n_changes_per_issue == 0).mean():.2%}")

    def last_change_checkpoint(g):
        g = g.sort_values("checkpoint_index")
        vals = g["pattern_label"].values
        idxs = g["checkpoint_index"].values
        changed = np.where(vals[1:] != vals[:-1])[0]
        if len(changed) == 0:
            return idxs[0]
        return idxs[changed[-1] + 1]

    stabilize_at = real_labels.groupby("Issue_ID").apply(last_change_checkpoint, include_groups=False)
    print(f"    median checkpoint of last label change (stabilization point): {stabilize_at.median():.1f}")

    # ------------------------------------------------------------------
    # Step 4: dynamic XGBoost models (a) without pattern, (b) with pattern
    # ------------------------------------------------------------------
    print("\nStep 4: training dynamic models ...")
    print(
        "  NOTE: Phase 7 was not available in this session (no saved model/params found), "
        "so standard reasonable XGBoost defaults were used instead of copying Phase 7's "
        "hyperparameters -- flagging this rather than guessing what Phase 7 used."
    )

    # ck already carries its own Project_ID/Type_Normalized/split columns
    # (from the Phase 8 foundation build) -- drop them before merging so the
    # phase6 static-feature versions aren't suffixed away.
    split_lookup = split.set_index("ID")["split"]
    assert (ck["Issue_ID"].map(split_lookup) == ck["split"]).all(), (
        "checkpoints.csv 'split' column disagrees with phase8_split.csv"
    )

    ck_for_merge = ck.drop(columns=["Project_ID", "Type_Normalized"])
    merged = ck_for_merge.merge(feat, left_on="Issue_ID", right_on="ID", how="left")

    for col in CATEGORICAL_COLS:
        merged[col] = merged[col].astype("category")
    merged["pattern_label"] = merged["pattern_label"].astype("category")

    feature_cols_a = STATIC_FEATURE_COLS + ["log1p_stall", "elapsed_minutes"]
    feature_cols_b = feature_cols_a + ["pattern_label"]

    train_mask = merged["split"] == "train"
    val_mask = merged["split"] == "val"
    test_mask = merged["split"] == "test"
    print(f"  train rows={train_mask.sum()} val rows={val_mask.sum()} test rows={test_mask.sum()}")

    y_train = merged.loc[train_mask, "delayed"]
    y_val = merged.loc[val_mask, "delayed"]

    os.makedirs("models", exist_ok=True)
    models = {}
    for name, feature_cols in [("no_pattern", feature_cols_a), ("with_pattern", feature_cols_b)]:
        print(f"  training model '{name}' with {len(feature_cols)} features ...")
        X_train = merged.loc[train_mask, feature_cols]
        X_val = merged.loc[val_mask, feature_cols]
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models[name] = model
        print(f"    best iteration: {model.best_iteration}")
        joblib.dump(model, os.path.join("models", f"xgb_dynamic_{name}.joblib"))

    checkpoints_with_pattern_path = os.path.join(FEATURES_DIR, "phase8_checkpoints_with_pattern.csv")
    ck[["Issue_ID", "checkpoint_index", "pattern_label"]].to_csv(
        checkpoints_with_pattern_path, index=False
    )
    print(f"  saved models/xgb_dynamic_{{no_pattern,with_pattern}}.joblib "
          f"and pattern labels -> {checkpoints_with_pattern_path}")

    # ------------------------------------------------------------------
    # Step 5: milestone-wise evaluation on test split
    # ------------------------------------------------------------------
    print("\nStep 5: milestone-wise evaluation (test split) ...")
    results = []
    for t in range(1, 11):
        test_t = merged[test_mask & (merged["checkpoint_index"] == t)]
        n = len(test_t)
        if n == 0:
            continue
        y_true = test_t["delayed"].to_numpy()
        row = {"checkpoint_index": t, "N": n}
        for name, feature_cols in [("no_pattern", feature_cols_a), ("with_pattern", feature_cols_b)]:
            X_t = test_t[feature_cols]
            proba = models[name].predict_proba(X_t)[:, 1]
            pred = (proba >= 0.5).astype(int)
            try:
                auc = roc_auc_score(y_true, proba)
            except ValueError:
                auc = np.nan
            row[f"{name}_auc"] = auc
            row[f"{name}_f1"] = f1_score(y_true, pred, zero_division=0)
            row[f"{name}_precision"] = precision_score(y_true, pred, zero_division=0)
            row[f"{name}_recall"] = recall_score(y_true, pred, zero_division=0)
        results.append(row)
        print(
            f"  M{t}: N={n} | no_pattern AUC={row['no_pattern_auc']:.4f} F1={row['no_pattern_f1']:.4f} "
            f"| with_pattern AUC={row['with_pattern_auc']:.4f} F1={row['with_pattern_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Step 6: save plot + metrics
    # ------------------------------------------------------------------
    results_df.to_csv(DYNAMIC_METRICS_OUT, index=False)
    print(f"\nSaved metrics table -> {DYNAMIC_METRICS_OUT}")

    plot_checkpoint_accuracy(results_df)

    print("\n=== Done. Stopping after Step 7 per instructions. ===")
    return results_df, n_changes_per_issue, stabilize_at


def plot_checkpoint_accuracy(results_df):
    surface, ink_primary, ink_secondary, ink_muted, gridline = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
    )
    color_no_pattern = "#2a78d6"
    color_with_pattern = "#eb6834"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(surface)

    for ax, metric, title in [(ax1, "auc", "AUC"), (ax2, "f1", "F1")]:
        ax.set_facecolor(surface)
        ax.plot(
            results_df["checkpoint_index"], results_df[f"no_pattern_{metric}"],
            color=color_no_pattern, linewidth=2, marker="o", markersize=5, label="Dynamic XGBoost, no pattern",
        )
        ax.plot(
            results_df["checkpoint_index"], results_df[f"with_pattern_{metric}"],
            color=color_with_pattern, linewidth=2, marker="o", markersize=5, label="Dynamic XGBoost, with pattern",
        )
        ax.set_xlabel("Checkpoint index (1-10)", color=ink_secondary)
        ax.set_ylabel(title, color=ink_secondary)
        ax.set_title(f"{title} by checkpoint (test split)", color=ink_primary)
        ax.set_xticks(range(1, 11))
        ax.tick_params(colors=ink_muted)
        ax.grid(color=gridline, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color(gridline)
        ax.legend(fontsize=8, facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)

        ax2_twin = ax.twinx()
        ax2_twin.bar(
            results_df["checkpoint_index"], results_df["N"], color=ink_muted, alpha=0.15, width=0.5,
        )
        ax2_twin.set_ylabel("N (test issues)", color=ink_muted, fontsize=8)
        ax2_twin.tick_params(colors=ink_muted, labelsize=7)

    fig.tight_layout()
    fig.savefig(ACCURACY_PLOT, dpi=150, facecolor=surface)
    print(f"  saved checkpoint-accuracy plot -> {ACCURACY_PLOT}")


if __name__ == "__main__":
    main()
