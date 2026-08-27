"""
Task A, Step 6: SHAP on the v2 (with-text) static and dynamic-with-pattern
models. Text PCA components (30 of them) are reported as a single
aggregated group (sum of mean|SHAP|) alongside individual features.
"""

import os

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

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

SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRIDLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)


def grouped_importance(shap_values, feature_cols):
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    text_total = imp[TEXT_COLS].sum()
    non_text = imp.drop(TEXT_COLS)
    grouped = pd.concat([non_text, pd.Series({"text_pc_GROUP(sum of 30)": text_total})])
    return grouped.sort_values(ascending=False)


def main():
    rng = np.random.default_rng(42)

    print("=== [1] Static v2 model SHAP ===")
    static_model = joblib.load(f"{MODELS_DIR}/xgb_static_v2.joblib")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    issue = pd.read_csv("data/raw/issue.csv", usecols=["ID", "Creation_Date"], parse_dates=["Creation_Date"])
    feat = feat.merge(issue, on="ID", how="left").sort_values("Creation_Date", kind="mergesort")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    test_ids = set(split.loc[split["split"] == "test", "ID"])
    static_test = feat[feat["ID"].isin(test_ids)].copy()
    for col in CATEGORICAL_COLS:
        static_test[col] = static_test[col].astype("category")

    sample_idx = rng.choice(len(static_test), size=min(8000, len(static_test)), replace=False)
    X_static_sample = static_test.iloc[sample_idx][STATIC_FEATURE_COLS_V2]
    explainer_static = shap.TreeExplainer(static_model)
    shap_static = explainer_static(X_static_sample)

    static_grouped = grouped_importance(shap_static.values, STATIC_FEATURE_COLS_V2)
    print(static_grouped.head(15).to_string())

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.gca()
    ax.set_facecolor(SURFACE)
    order = static_grouped.head(15).index[::-1]
    ax.barh(order, static_grouped.loc[order], color="#4C78A8")
    ax.set_xlabel("mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title("Phase 7 v2 (static + text): feature importance (text grouped)", color=INK_PRIMARY)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/phase10_shap_static_summary_v2.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {REPORTS_DIR}/phase10_shap_static_summary_v2.png")

    print("\n=== [2] Dynamic with-pattern v2 model SHAP ===")
    dynamic_model = joblib.load(f"{MODELS_DIR}/xgb_dynamic_with_pattern_v2.joblib")
    dyn_feature_cols = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]

    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    feat_v2 = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat_v2, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    test_ck = ck[ck["split"] == "test"].copy()

    sample_idx2 = rng.choice(len(test_ck), size=min(8000, len(test_ck)), replace=False)
    X_dyn_sample = test_ck.iloc[sample_idx2][dyn_feature_cols]
    explainer_dyn = shap.TreeExplainer(dynamic_model)
    shap_dyn = explainer_dyn(X_dyn_sample)

    dyn_grouped = grouped_importance(shap_dyn.values, dyn_feature_cols)
    print(dyn_grouped.head(15).to_string())

    for feature in ["text_pc_GROUP(sum of 30)", "assignee_prior_delay_rate", "elapsed_minutes",
                     "log1p_stall", "pattern_label"]:
        rank = list(dyn_grouped.index).index(feature) + 1
        print(f"  '{feature}' ranks #{rank} of {len(dyn_grouped)} (grouped)")

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.gca()
    ax.set_facecolor(SURFACE)
    order = dyn_grouped.head(15).index[::-1]
    ax.barh(order, dyn_grouped.loc[order], color="#eb6834")
    ax.set_xlabel("mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title("Phase 8 v2 dynamic (with pattern + text): feature importance (text grouped)", color=INK_PRIMARY)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/phase10_shap_dynamic_summary_v2.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {REPORTS_DIR}/phase10_shap_dynamic_summary_v2.png")

    return static_grouped, dyn_grouped


if __name__ == "__main__":
    main()
