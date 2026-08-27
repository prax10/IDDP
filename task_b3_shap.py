"""
Task B3, Step 2.4: SHAP on the v3 (delayed_v2-labeled) static model and
the M8 per-checkpoint dynamic model (representative late-mid checkpoint).
Text PCA components reported as a summed group and individually.
"""

import numpy as np
import pandas as pd
import joblib
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
    present_text = [c for c in TEXT_COLS if c in feature_cols]
    text_total = imp[present_text].sum()
    non_text = imp.drop(present_text)
    grouped = pd.concat([non_text, pd.Series({"text_pc_GROUP(sum)": text_total})])
    return grouped.sort_values(ascending=False), imp.sort_values(ascending=False)


def main():
    rng = np.random.default_rng(42)

    print("=== Static v3 model SHAP ===")
    static_model = joblib.load(f"{MODELS_DIR}/xgb_static_v3.joblib")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")
    for col in CATEGORICAL_COLS:
        feat[col] = feat[col].astype("category")
    static_test = feat[feat["split"] == "test"]

    sample_idx = rng.choice(len(static_test), size=min(8000, len(static_test)), replace=False)
    X_static_sample = static_test.iloc[sample_idx][STATIC_FEATURE_COLS_V2]
    explainer_static = shap.TreeExplainer(static_model)
    shap_static = explainer_static(X_static_sample)

    grouped, individual = grouped_importance(shap_static.values, STATIC_FEATURE_COLS_V2)
    print("Grouped (text summed):")
    print(grouped.head(15).to_string())
    print("\nIndividual (top 15, text components shown separately):")
    print(individual.head(15).to_string())

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.gca()
    ax.set_facecolor(SURFACE)
    order = grouped.head(15).index[::-1]
    ax.barh(order, grouped.loc[order], color="#4C78A8")
    ax.set_xlabel("mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title("Phase 7 v3 (train+val-median labels): feature importance", color=INK_PRIMARY)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/phase10_shap_static_summary_v3.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {REPORTS_DIR}/phase10_shap_static_summary_v3.png")

    print("\n=== Per-checkpoint M8 dynamic model SHAP ===")
    dyn_model_m8 = joblib.load(f"{MODELS_DIR}/xgb_dynamic_percheckpoint_M8_v3.joblib")
    dyn_feature_cols = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]

    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat.drop(columns=["split"]), left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    ck = ck.merge(split[["ID", "split"]], left_on="Issue_ID", right_on="ID", how="left", suffixes=("", "_s"))

    test_m8 = ck[(ck["split"] == "test") & (ck["checkpoint_index"] == 8)]
    sample_idx2 = rng.choice(len(test_m8), size=min(8000, len(test_m8)), replace=False)
    X_dyn_sample = test_m8.iloc[sample_idx2][dyn_feature_cols]

    explainer_dyn = shap.TreeExplainer(dyn_model_m8)
    shap_dyn = explainer_dyn(X_dyn_sample)

    dyn_grouped, dyn_individual = grouped_importance(shap_dyn.values, dyn_feature_cols)
    print("Grouped (text summed):")
    print(dyn_grouped.head(15).to_string())
    print("\nIndividual (top 15):")
    print(dyn_individual.head(15).to_string())

    for feature in ["text_pc_GROUP(sum)", "assignee_prior_delay_rate", "elapsed_minutes",
                     "log1p_stall", "pattern_label"]:
        if feature in dyn_grouped.index:
            rank = list(dyn_grouped.index).index(feature) + 1
            print(f"  '{feature}' ranks #{rank} of {len(dyn_grouped)} (grouped)")

    fig = plt.figure(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.gca()
    ax.set_facecolor(SURFACE)
    order = dyn_grouped.head(15).index[::-1]
    ax.barh(order, dyn_grouped.loc[order], color="#eb6834")
    ax.set_xlabel("mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title("Phase 8 v3 M8 per-checkpoint model (train+val-median labels): feature importance",
                 color=INK_PRIMARY)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/phase10_shap_percheckpoint_M8_v3.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {REPORTS_DIR}/phase10_shap_percheckpoint_M8_v3.png")

    return grouped, dyn_grouped


if __name__ == "__main__":
    main()
