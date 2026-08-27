"""
Task A, Step 3: merge text_pc_0..29 into phase6_feature_table.csv ->
phase6_feature_table_v2.csv (original kept for with/without comparison).
Re-run the leakage test on the new table.
"""

import pandas as pd

FEATURES_DIR = "data/features"

TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
BANNED = {"Resolution_Date", "Resolution_Time_Minutes", "Status"}


def main():
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table.csv")
    text_pca = pd.read_csv(f"{FEATURES_DIR}/text_pca30.csv")

    merged = feat.merge(text_pca, on="ID", how="left")
    assert merged[TEXT_COLS].isna().sum().sum() == 0, "text PCA merge produced NaNs -- ID mismatch"

    out_path = f"{FEATURES_DIR}/phase6_feature_table_v2.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_path}")

    print("\n=== Leakage test on v2 table ===")
    feature_cols = [c for c in merged.columns if c not in ("ID", "delayed")]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]
    corrs = merged[numeric_cols].corrwith(merged["delayed"]).abs().sort_values(ascending=False)
    print("Top 10 |correlation| with delayed:")
    print(corrs.head(10).to_string())
    max_corr = corrs.max()
    corr_ok = max_corr <= 0.95
    print(f"Max |correlation|: {max_corr:.4f} -> {'PASS' if corr_ok else 'FAIL'}")

    present_banned = BANNED.intersection(merged.columns)
    banned_ok = len(present_banned) == 0
    print(f"Banned columns present: {present_banned or 'none'} -> {'PASS' if banned_ok else 'FAIL'}")

    overall = corr_ok and banned_ok
    print(f"\n=== Leakage test overall: {'PASS' if overall else 'FAIL'} ===")
    if not overall:
        raise SystemExit("STOP: leakage test failed on v2 table.")


if __name__ == "__main__":
    main()
