"""
Task B, Part 1: full 39-project leave-one-project-out generalization,
using the v2 (with-text) feature table and Phase 7's hyperparameters.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
RAW_DIR = "data/raw"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]

PHASE7_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
)


def main():
    print("=== Task B, Part 1: full 39-project LOPO (v2, with text) ===")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    issue = pd.read_csv(f"{RAW_DIR}/issue.csv", usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    all_projects = sorted(feat["Project_ID"].unique())
    print(f"  {len(all_projects)} projects total")

    rows = []
    for i, p in enumerate(all_projects, 1):
        train_df = feat[feat["Project_ID"] != p]
        test_df = feat[feat["Project_ID"] == p]
        n_test = len(test_df)
        delay_rate = test_df["delayed"].mean()

        clf = xgb.XGBClassifier(**PHASE7_PARAMS)
        clf.fit(train_df[STATIC_FEATURE_COLS_V2], train_df["delayed"])
        proba = clf.predict_proba(test_df[STATIC_FEATURE_COLS_V2])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            cross_auc = roc_auc_score(test_df["delayed"], proba)
        except ValueError:
            cross_auc = np.nan
        cross_f1 = f1_score(test_df["delayed"], pred, zero_division=0)

        proj_df = feat[feat["Project_ID"] == p].sort_values("Creation_Date", kind="mergesort")
        n = len(proj_df)
        n_train = int(round(n * 0.8))
        proj_train, proj_test = proj_df.iloc[:n_train], proj_df.iloc[n_train:]

        within_skipped, skip_reason = False, None
        within_auc, within_f1 = np.nan, np.nan
        n_within_test = len(proj_test)
        if len(proj_test) < 30:
            within_skipped, skip_reason = True, f"test set too small (n={len(proj_test)}<30)"
        elif proj_train["delayed"].nunique() < 2:
            within_skipped, skip_reason = True, "train split missing a class"
        elif proj_test["delayed"].nunique() < 2:
            within_skipped, skip_reason = True, "test split missing a class"
        else:
            clf_w = xgb.XGBClassifier(**PHASE7_PARAMS)
            clf_w.fit(proj_train[STATIC_FEATURE_COLS_V2], proj_train["delayed"])
            proba_w = clf_w.predict_proba(proj_test[STATIC_FEATURE_COLS_V2])[:, 1]
            pred_w = (proba_w >= 0.5).astype(int)
            within_auc = roc_auc_score(proj_test["delayed"], proba_w)
            within_f1 = f1_score(proj_test["delayed"], pred_w, zero_division=0)

        gap = (within_auc - cross_auc) if not within_skipped else np.nan
        rows.append({
            "Project_ID": p, "N": n_test, "delay_rate": delay_rate,
            "cross_project_auc": cross_auc, "cross_project_f1": cross_f1,
            "within_project_auc": within_auc, "within_project_f1": within_f1,
            "n_within_test": n_within_test, "within_project_skipped": within_skipped,
            "skip_reason": skip_reason,
            "gap_auc_within_minus_cross": gap,
            "cross_beats_within": (gap < 0) if not within_skipped else np.nan,
        })
        print(f"  [{i:2d}/{len(all_projects)}] project {p}: N={n_test} delay_rate={delay_rate:.3f} "
              f"cross_AUC={cross_auc:.4f} "
              f"within_AUC={'SKIPPED (' + skip_reason + ')' if within_skipped else f'{within_auc:.4f}'}")

    results_df = pd.DataFrame(rows)
    out_path = f"{REPORTS_DIR}/phase9_generalization_full.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")

    valid = results_df[~results_df["within_project_skipped"]]
    n_skipped = results_df["within_project_skipped"].sum()
    print(f"\n  projects with within-project comparison skipped: {n_skipped} "
          f"({results_df.loc[results_df['within_project_skipped'], 'Project_ID'].tolist()})")

    for label, col in [("cross-project AUC", "cross_project_auc"), ("within-project AUC", "within_project_auc")]:
        s = results_df[col].dropna()
        print(f"  {label}: mean={s.mean():.4f} median={s.median():.4f} "
              f"IQR=[{s.quantile(.25):.4f}, {s.quantile(.75):.4f}] (n={len(s)})")

    gaps = valid["gap_auc_within_minus_cross"]
    n_cross_wins = (gaps < 0).sum()
    print(f"\n  gap (within - cross): mean={gaps.mean():.4f} median={gaps.median():.4f}")
    print(f"  cross-project beats within-project: {n_cross_wins} / {len(valid)} projects "
          f"(10-project pilot: 8/10)")

    plot_full_generalization(results_df)
    return results_df


def plot_full_generalization(df):
    surface, ink_primary, ink_secondary, ink_muted, gridline = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
    )
    color_cross, color_within = "#2a78d6", "#eb6834"

    df = df.sort_values("N").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    x = np.arange(len(df))
    ax.plot(x, df["cross_project_auc"], color=color_cross, marker="o", markersize=5,
            linewidth=1.3, label="Cross-project (train on other 38)")
    within_valid = ~df["within_project_skipped"]
    ax.plot(x[within_valid], df.loc[within_valid, "within_project_auc"], color=color_within,
            marker="o", markersize=5, linewidth=1.3, label="Within-project (own 80/20 split)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}\nN={n}" for p, n in zip(df["Project_ID"], df["N"])],
                        rotation=90, fontsize=6, color=ink_muted)
    ax.set_ylabel("AUC", color=ink_secondary)
    ax.set_title("Phase 9 (full 39-project LOPO, v2/with-text): cross-project vs. within-project AUC, sorted by N",
                 color=ink_primary)
    ax.tick_params(colors=ink_muted)
    ax.grid(color=gridline, linewidth=0.8, axis="y")
    for spine in ax.spines.values():
        spine.set_color(gridline)
    ax.legend(fontsize=9, facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)
    fig.tight_layout()
    out_path = f"{REPORTS_DIR}/phase9_generalization_full.png"
    fig.savefig(out_path, dpi=150, facecolor=surface)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
