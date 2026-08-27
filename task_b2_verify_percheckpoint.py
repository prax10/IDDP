"""
Task B2: verify the per-checkpoint result before adopting it as headline.
Check 1: full three-line M1-M10 table + exact crossover checkpoint.
Check 2: per-checkpoint models without elapsed_minutes (mechanism test).
Check 3: label-definition sensitivity (train-only project medians).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

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


def check1():
    print("=== Check 1: full three-line table + exact crossover ===")
    dyn = pd.read_csv(f"{REPORTS_DIR}/phase8_dynamic_metrics_v2.csv")
    per_ck = pd.read_csv(f"{REPORTS_DIR}/phase8_pooled_vs_percheckpoint.csv")

    table = dyn[["checkpoint_index", "N", "pct_delayed", "majority_baseline_f1",
                 "majority_baseline_accuracy", "static_v2_auc", "with_pattern_auc"]].merge(
        per_ck[["checkpoint_index", "per_checkpoint_auc"]], on="checkpoint_index", how="left"
    ).rename(columns={"with_pattern_auc": "pooled_dynamic_auc"})

    print(table.to_string(index=False))

    crossover = None
    for _, r in table.iterrows():
        if r["per_checkpoint_auc"] > r["static_v2_auc"]:
            crossover = int(r["checkpoint_index"])
            break
    print(f"\n  per-checkpoint dynamic first overtakes static at: M{crossover}")

    out_path = f"{REPORTS_DIR}/phase8_three_line_comparison.csv"
    table.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
    return table, crossover


def build_merged():
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])

    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    ck = ck.drop(columns=["Project_ID", "Type_Normalized"]).merge(
        feat, left_on="Issue_ID", right_on="ID", how="left"
    )
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    return ck


def check2():
    print("\n=== Check 2: per-checkpoint models WITHOUT elapsed_minutes ===")
    merged = build_merged()
    train_mask = merged["split"].isin(["train", "val"])
    test_mask = merged["split"] == "test"

    feature_cols_no_elapsed = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "pattern_label"]

    rows = []
    for t in range(1, 11):
        train_t = merged[train_mask & (merged["checkpoint_index"] == t)]
        test_t = merged[test_mask & (merged["checkpoint_index"] == t)]
        if len(test_t) == 0 or train_t["delayed"].nunique() < 2:
            continue

        clf = xgb.XGBClassifier(**PHASE7_PARAMS)
        clf.fit(train_t[feature_cols_no_elapsed], train_t["delayed"])
        proba = clf.predict_proba(test_t[feature_cols_no_elapsed])[:, 1]
        auc = roc_auc_score(test_t["delayed"], proba)
        rows.append({"checkpoint_index": t, "N": len(test_t), "per_checkpoint_no_elapsed_auc": auc})
        print(f"  M{t}: N={len(test_t)} AUC(no elapsed_minutes)={auc:.4f}")

    no_elapsed_df = pd.DataFrame(rows)
    per_ck = pd.read_csv(f"{REPORTS_DIR}/phase8_pooled_vs_percheckpoint.csv")
    dyn = pd.read_csv(f"{REPORTS_DIR}/phase8_dynamic_metrics_v2.csv")

    combined = per_ck[["checkpoint_index", "per_checkpoint_auc"]].merge(
        no_elapsed_df, on="checkpoint_index", how="left"
    ).merge(
        dyn[["checkpoint_index", "with_pattern_auc"]].rename(columns={"with_pattern_auc": "pooled_dynamic_auc"}),
        on="checkpoint_index", how="left",
    )
    combined["auc_drop_from_removing_elapsed"] = (
        combined["per_checkpoint_auc"] - combined["per_checkpoint_no_elapsed_auc"]
    )
    print("\n" + combined.to_string(index=False))

    out_path = f"{REPORTS_DIR}/phase8_percheckpoint_mechanism_check.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")

    late = combined[combined["checkpoint_index"] >= 8]
    collapses_toward_pooled = (
        (late["per_checkpoint_no_elapsed_auc"] - late["pooled_dynamic_auc"]).abs().mean() < 0.02
    )
    mean_drop = combined["auc_drop_from_removing_elapsed"].mean()
    print(f"\n  mean AUC drop from removing elapsed_minutes: {mean_drop:.4f}")
    print(f"  late-checkpoint (M8-M10) no-elapsed AUC collapses toward pooled level: {collapses_toward_pooled}")
    return combined


def check3():
    print("\n=== Check 3: label-definition sensitivity (train-only medians) ===")
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table.csv")
    issue = pd.read_csv(f"{RAW_DIR}/issue.csv", usecols=["ID", "Resolution_Time_Minutes"])
    feat = feat.merge(issue, on="ID", how="left")

    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")

    train_medians = (
        feat[feat["split"].isin(["train", "val"])]
        .groupby("Project_ID")["Resolution_Time_Minutes"]
        .median()
    )
    feat["proj_median_train_only"] = feat["Project_ID"].map(train_medians)

    missing = feat["proj_median_train_only"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} rows have no train-only median for their project "
              f"(project entirely in test) -- excluded from the comparison")

    test = feat[(feat["split"] == "test") & feat["proj_median_train_only"].notna()].copy()
    test["delayed_train_only_median"] = (
        test["Resolution_Time_Minutes"] > test["proj_median_train_only"]
    ).astype(int)

    n_changed = (test["delayed"] != test["delayed_train_only_median"]).sum()
    pct_changed = n_changed / len(test) * 100
    print(f"  test issues compared: {len(test)}")
    print(f"  labels changed: {n_changed} ({pct_changed:.3f}%)")

    verdict = "negligible" if pct_changed < 2 else "MATERIAL -- worth a re-run"
    print(f"  verdict: {verdict}")
    return pct_changed


if __name__ == "__main__":
    table, crossover = check1()
    mech = check2()
    pct_changed = check3()
