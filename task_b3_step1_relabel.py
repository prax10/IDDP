"""
Task B3, Step 1: relabel using project medians computed from train+val
issues only (the shared 80% complement of Phase 7/8's test split), applied
to ALL issues. Keeps the old label alongside for the robustness comparison.
"""

import pandas as pd

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"


def main():
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v2.csv")
    issue = pd.read_csv(f"{RAW_DIR}/issue.csv", usecols=["ID", "Resolution_Time_Minutes"])
    feat = feat.merge(issue, on="ID", how="left")

    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat = feat.merge(split[["ID", "split"]], on="ID", how="left")

    train_val = feat[feat["split"].isin(["train", "val"])]
    train_val_medians = train_val.groupby("Project_ID")["Resolution_Time_Minutes"].median()
    train_val_counts = train_val.groupby("Project_ID").size()

    feat["proj_median_train_val"] = feat["Project_ID"].map(train_val_medians)
    feat["delayed_v2"] = (feat["Resolution_Time_Minutes"] > feat["proj_median_train_val"]).astype(int)

    print("=== Overall class balance ===")
    print(f"  old label (delayed):    {feat['delayed'].mean():.4f} positive")
    print(f"  new label (delayed_v2): {feat['delayed_v2'].mean():.4f} positive")

    test = feat[feat["split"] == "test"]
    print("\n=== Test-set class balance ===")
    print(f"  old label (delayed):    {test['delayed'].mean():.4f} positive (n={len(test)})")
    print(f"  new label (delayed_v2): {test['delayed_v2'].mean():.4f} positive (n={len(test)})")

    n_changed = (feat["delayed"] != feat["delayed_v2"]).sum()
    print(f"\n  labels changed overall: {n_changed} / {len(feat)} ({n_changed/len(feat)*100:.3f}%)")
    n_changed_test = (test["delayed"] != test["delayed_v2"]).sum()
    print(f"  labels changed in test split: {n_changed_test} / {len(test)} ({n_changed_test/len(test)*100:.3f}%)")

    print("\n=== Small train+val projects (noisy median risk) ===")
    small = train_val_counts[train_val_counts < 100].sort_values()
    for pid, n in small.items():
        print(f"  Project_ID={pid}: train+val n={n}")
    print(f"  {len(small)} of {len(train_val_counts)} projects have <100 train+val issues")

    out_path = f"{FEATURES_DIR}/phase6_feature_table_v3.csv"
    feat = feat.drop(columns=["split", "proj_median_train_val"])
    feat.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({feat.shape[0]} rows x {feat.shape[1]} cols, "
          f"both 'delayed' [old] and 'delayed_v2' [new] present)")


if __name__ == "__main__":
    main()
