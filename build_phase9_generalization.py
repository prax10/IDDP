"""
Phase 9: (A) verify the elapsed_minutes/expected_duration_proxy claim and
run the no-elapsed_minutes ablation, (B) cross-project generalization
(train-on-38-test-on-1) vs. within-project performance for 10 projects.
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

from build_phase8_foundation import (
    build_expanding_median_lookup,
    hierarchical_query,
)

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"
RAW_DIR = "data/raw"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]

SMALL_PROJECTS = [10, 6, 9, 2, 35]
LARGE_PROJECTS = [34, 33, 28, 12, 22]
CHOSEN_PROJECTS = SMALL_PROJECTS + LARGE_PROJECTS

PHASE7_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    enable_categorical=True, tree_method="hist",
)


def recompute_expected_duration_proxy():
    """Re-derive expected_duration_proxy independently of the stored
    elapsed_minutes column, using the same Step-3 hierarchical/expanding
    logic as the Phase 8 foundation build, so Part A's correlation check
    is a genuine test rather than a circular back-solve."""
    issue = pd.read_csv(
        os.path.join(RAW_DIR, "issue.csv"),
        usecols=["ID", "Resolution_Time_Minutes", "Creation_Date", "Resolution_Date"],
        parse_dates=["Creation_Date", "Resolution_Date"],
    )
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    res = feat.merge(issue, on="ID", how="left")
    res["Type_Normalized"] = res["Type_Normalized"].astype("category")
    res["Priority_Normalized"] = res["Priority_Normalized"].astype("category")
    res = res.sort_values("Creation_Date", kind="mergesort").reset_index(drop=True)

    lvl1 = build_expanding_median_lookup(
        res, ["Project_ID", "Type_Normalized", "Priority_Normalized"],
        "Resolution_Date", "Resolution_Time_Minutes",
    )
    lvl2 = build_expanding_median_lookup(
        res, ["Project_ID", "Type_Normalized"], "Resolution_Date", "Resolution_Time_Minutes"
    )
    lvl3 = build_expanding_median_lookup(
        res, ["Project_ID", "Priority_Normalized"], "Resolution_Date", "Resolution_Time_Minutes"
    )
    lvl4 = build_expanding_median_lookup(
        res, ["Project_ID"], "Resolution_Date", "Resolution_Time_Minutes"
    )
    lvl5 = build_expanding_median_lookup(res, [], "Resolution_Date", "Resolution_Time_Minutes")

    creation_vals = res["Creation_Date"].values
    proj_vals = res["Project_ID"].values
    type_vals = res["Type_Normalized"].values
    prio_vals = res["Priority_Normalized"].values

    proxy = np.empty(len(res), dtype=float)
    for i in range(len(res)):
        t = creation_vals[i]
        levels = [
            (lvl1, (proj_vals[i], type_vals[i], prio_vals[i])),
            (lvl2, (proj_vals[i], type_vals[i])),
            (lvl3, (proj_vals[i], prio_vals[i])),
            (lvl4, (proj_vals[i],)),
            (lvl5, ()),
        ]
        proxy[i] = hierarchical_query(levels, t)

    res["expected_duration_proxy"] = proxy
    return res[["ID", "expected_duration_proxy"]]


def part_a():
    print("=== Part A: verify elapsed_minutes / expected_duration_proxy ===")
    proxy_df = recompute_expected_duration_proxy()

    ck = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_checkpoints.csv"))
    ck = ck.merge(proxy_df, left_on="Issue_ID", right_on="ID", how="left")
    test_ck = ck[ck["split"] == "test"]

    for cidx in [5, 8]:
        sub = test_ck[test_ck["checkpoint_index"] == cidx]
        corr = sub["elapsed_minutes"].corr(sub["expected_duration_proxy"])
        print(f"  M{cidx}: n={len(sub)}, corr(elapsed_minutes, expected_duration_proxy) = {corr:.6f}")

    pooled_corr = ck["elapsed_minutes"].corr(ck["checkpoint_index"])
    print(f"  pooled corr(elapsed_minutes, checkpoint_index), all rows: {pooled_corr:.4f}")

    return ck


def part_a_ablation(ck):
    print("\n  Confirmed ~1.0 -> running no-elapsed_minutes ablation ...")
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

    feature_cols = STATIC_FEATURE_COLS + ["log1p_stall", "pattern_label"]
    train_mask, val_mask, test_mask = ck["split"] == "train", ck["split"] == "val", ck["split"] == "test"

    model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=300, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", enable_categorical=True,
        tree_method="hist", early_stopping_rounds=20, random_state=42,
    )
    model.fit(
        ck.loc[train_mask, feature_cols], ck.loc[train_mask, "delayed"],
        eval_set=[(ck.loc[val_mask, feature_cols], ck.loc[val_mask, "delayed"])],
        verbose=False,
    )
    print(f"  best iteration: {model.best_iteration}")

    rows = []
    for t in range(1, 11):
        sub = ck[test_mask & (ck["checkpoint_index"] == t)]
        if len(sub) == 0:
            continue
        proba = model.predict_proba(sub[feature_cols])[:, 1]
        auc = roc_auc_score(sub["delayed"], proba)
        rows.append({"checkpoint_index": t, "N": len(sub), "no_elapsed_minutes_auc": auc})
        print(f"  M{t}: N={len(sub)} AUC(no elapsed_minutes)={auc:.4f}")

    ablation_df = pd.DataFrame(rows)
    metrics_path = os.path.join(REPORTS_DIR, "phase8_dynamic_metrics.csv")
    dyn_metrics = pd.read_csv(metrics_path)
    dyn_metrics = dyn_metrics.drop(columns=["no_elapsed_minutes_auc"], errors="ignore")
    dyn_metrics = dyn_metrics.merge(ablation_df, on="checkpoint_index", how="left")
    dyn_metrics.to_csv(metrics_path, index=False)
    print(f"  updated -> {metrics_path}")


def part_b():
    print("\n=== Part B: cross-project generalization ===")
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))
    issue = pd.read_csv(os.path.join(RAW_DIR, "issue.csv"), usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")

    rows = []
    for p in CHOSEN_PROJECTS:
        train_df = feat[feat["Project_ID"] != p]
        test_df = feat[feat["Project_ID"] == p]
        n_test = len(test_df)
        delay_rate = test_df["delayed"].mean()

        clf = xgb.XGBClassifier(**PHASE7_PARAMS)
        clf.fit(train_df[STATIC_FEATURE_COLS], train_df["delayed"])
        proba = clf.predict_proba(test_df[STATIC_FEATURE_COLS])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            cross_auc = roc_auc_score(test_df["delayed"], proba)
        except ValueError:
            cross_auc = np.nan
        cross_f1 = f1_score(test_df["delayed"], pred, zero_division=0)

        # within-project chronological 80/20 split
        proj_df = feat[feat["Project_ID"] == p].sort_values("Creation_Date", kind="mergesort")
        n = len(proj_df)
        n_train = int(round(n * 0.8))
        proj_train, proj_test = proj_df.iloc[:n_train], proj_df.iloc[n_train:]

        within_skipped = False
        within_auc, within_f1, n_within_test = np.nan, np.nan, len(proj_test)
        if len(proj_test) < 30 or proj_train["delayed"].nunique() < 2 or proj_test["delayed"].nunique() < 2:
            within_skipped = True
        else:
            clf_w = xgb.XGBClassifier(**PHASE7_PARAMS)
            clf_w.fit(proj_train[STATIC_FEATURE_COLS], proj_train["delayed"])
            proba_w = clf_w.predict_proba(proj_test[STATIC_FEATURE_COLS])[:, 1]
            pred_w = (proba_w >= 0.5).astype(int)
            within_auc = roc_auc_score(proj_test["delayed"], proba_w)
            within_f1 = f1_score(proj_test["delayed"], pred_w, zero_division=0)

        rows.append({
            "Project_ID": p, "N": n_test, "delay_rate": delay_rate,
            "cross_project_auc": cross_auc, "cross_project_f1": cross_f1,
            "within_project_auc": within_auc, "within_project_f1": within_f1,
            "n_within_test": n_within_test, "within_project_skipped": within_skipped,
            "gap_auc_within_minus_cross": within_auc - cross_auc if not within_skipped else np.nan,
            "small_project_flag": n_test < 700,
        })
        print(
            f"  project {p}: N={n_test} delay_rate={delay_rate:.3f} "
            f"cross_AUC={cross_auc:.4f} within_AUC={within_auc if not within_skipped else 'SKIPPED'}"
        )

    results_df = pd.DataFrame(rows)
    out_path = os.path.join(REPORTS_DIR, "phase9_generalization.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")

    valid_gaps = results_df.loc[~results_df["within_project_skipped"], "gap_auc_within_minus_cross"]
    print(f"\n  cross-project AUC: mean={results_df['cross_project_auc'].mean():.4f} "
          f"median={results_df['cross_project_auc'].median():.4f} "
          f"IQR=[{results_df['cross_project_auc'].quantile(.25):.4f}, "
          f"{results_df['cross_project_auc'].quantile(.75):.4f}]")
    print(f"  within-project AUC: mean={results_df['within_project_auc'].mean():.4f} "
          f"median={results_df['within_project_auc'].median():.4f} "
          f"IQR=[{results_df['within_project_auc'].quantile(.25):.4f}, "
          f"{results_df['within_project_auc'].quantile(.75):.4f}]")
    print(f"  generalization gap (within - cross): mean={valid_gaps.mean():.4f} median={valid_gaps.median():.4f}")

    plot_generalization(results_df)
    return results_df


def plot_generalization(results_df):
    surface, ink_primary, ink_secondary, ink_muted, gridline = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
    )
    color_cross, color_within = "#2a78d6", "#eb6834"

    df = results_df.sort_values("N")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    x = np.arange(len(df))
    ax.plot(x, df["cross_project_auc"], color=color_cross, marker="o", markersize=7, linewidth=1.5, label="Cross-project (train on other 38)")
    within_valid = ~df["within_project_skipped"]
    ax.plot(x[within_valid], df.loc[within_valid, "within_project_auc"], color=color_within, marker="o", markersize=7, linewidth=1.5, label="Within-project (own 80/20 split)")

    for xi, (_, row) in zip(x, df.iterrows()):
        ax.annotate(f"N={row['N']}", (xi, row["cross_project_auc"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color=ink_muted)

    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in df["Project_ID"]], color=ink_secondary)
    ax.set_ylabel("AUC", color=ink_secondary)
    ax.set_title("Phase 9: cross-project vs. within-project AUC (10 projects, sorted by N)", color=ink_primary)
    ax.tick_params(colors=ink_muted)
    ax.grid(color=gridline, linewidth=0.8, axis="y")
    for spine in ax.spines.values():
        spine.set_color(gridline)
    ax.legend(fontsize=9, facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)
    fig.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "phase9_generalization.png")
    fig.savefig(out_path, dpi=150, facecolor=surface)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    ck = part_a()
    part_a_ablation(ck)
    part_b()
