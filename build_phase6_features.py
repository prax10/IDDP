"""
Build the Phase 6 feature table from the raw TAWOS cache.

Reads data/raw/issue.csv, data/raw/issue_link.csv.
Writes data/features/phase6_feature_table.csv (203,290 rows, one per
resolved issue) including the `delayed` label.

SBERT text features (section 6.4) are intentionally skipped.
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
FEATURES_DIR = "data/features"
OUT_PATH = os.path.join(FEATURES_DIR, "phase6_feature_table.csv")

COMPLETION_RESOLUTIONS = [
    "Fixed", "Done", "Complete", "Completed", "Resolved",
    "Resolved Locally", "Implemented", "Deployed", "Handled by Support",
]
NON_WORK_STATUS = [
    "Won't Fix", "Invalid", "Duplicate", "Cannot Reproduce", "Won't Do",
    "Timed out", "Obsolete", "Inactive", "Not A Bug", "Rejected",
]

PRIORITY_MAP = {
    "Trivial": "1", "Trivial - P5": "1", "Lowest": "1",
    "Minor": "2", "Minor - P4": "2", "Low": "2",
    "Major": "3", "Major - P3": "3", "Medium": "3",
    "Critical": "4", "Critical - P2": "4", "High": "4",
    "Blocker": "5", "Blocker - P1": "5", "Highest": "5",
}

TYPE_MIN_COUNT = 100

BLOCKING_KEYWORDS = ["block", "depend", "require"]


def normalize_priority(series: pd.Series) -> pd.Series:
    mapped = series.map(PRIORITY_MAP)
    mapped = mapped.fillna("Unknown")
    return mapped.astype("category")


def normalize_type(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    keep = counts[counts >= TYPE_MIN_COUNT].index
    normalized = series.where(series.isin(keep), other="Other")
    return normalized.astype("category")


def compute_assignee_group_features(g: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-assignee sweep-line computation over the FULL raw
    Issue population: prior-resolved count and concurrent-open count at
    each row's own Creation_Date, using only strictly-earlier events."""
    g = g.sort_values("Creation_Date")
    creations = g["Creation_Date"].values.astype("datetime64[ns]")
    resolutions = g["Resolution_Date"].values.astype("datetime64[ns]")

    finite_mask = ~pd.isna(resolutions)
    finite_res_sorted = np.sort(resolutions[finite_mask])

    # count of same-assignee issues with Resolution_Date < this row's Creation_Date
    prior_resolved_count = np.searchsorted(finite_res_sorted, creations, side="left")

    # count of same-assignee issues with Creation_Date < this row's Creation_Date
    starts_count = np.searchsorted(creations, creations, side="left")
    # count of same-assignee issues with Resolution_Date <= this row's Creation_Date
    ends_count = np.searchsorted(finite_res_sorted, creations, side="right")
    concurrent_open = starts_count - ends_count

    g = g.copy()
    g["assignee_prior_resolved_count"] = prior_resolved_count
    g["assignee_concurrent_open"] = concurrent_open
    return g


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("Loading data/raw/issue.csv ...")
    full_df = pd.read_csv(
        os.path.join(RAW_DIR, "issue.csv"),
        parse_dates=["Creation_Date", "Resolution_Date", "Estimation_Date"],
    )
    print(f"  full population: {len(full_df)} rows")

    # ------------------------------------------------------------------
    # Step 1: EDA-validated resolved-issue filter (locked, do not modify)
    # ------------------------------------------------------------------
    base_resolved = full_df["Resolution_Date"].notna() | (
        full_df["Resolution"].notna() & (full_df["Resolution_Time_Minutes"] > 0)
    )
    genuine_completion = full_df["Resolution"].isin(COMPLETION_RESOLUTIONS) & ~full_df[
        "Status"
    ].isin(NON_WORK_STATUS)
    res = full_df[base_resolved & genuine_completion].copy()

    print(f"\nStep 1: resolved filter -> {len(res)} rows (expected 203290)")
    if len(res) != 203290:
        raise SystemExit(
            f"STOP: resolved filter produced {len(res)} rows, expected 203290. "
            "Reporting discrepancy instead of proceeding."
        )

    # ------------------------------------------------------------------
    # Step 2: label
    # ------------------------------------------------------------------
    res["proj_median"] = res.groupby("Project_ID")["Resolution_Time_Minutes"].transform(
        "median"
    )
    res["delayed"] = (res["Resolution_Time_Minutes"] > res["proj_median"]).astype(int)

    balance = res["delayed"].mean()
    print(f"\nStep 2: label balance -> delayed=1 rate = {balance:.4f}")

    # ------------------------------------------------------------------
    # Step 3.1: base features
    # ------------------------------------------------------------------
    res["Project_ID"] = res["Project_ID"].astype("category")
    res["Priority_Normalized"] = normalize_priority(res["Priority"])
    res["Type_Normalized"] = normalize_type(res["Type"])
    # Story_Point kept as-is (float, NaN preserved, no imputation)

    # ------------------------------------------------------------------
    # Step 3.2: workload features
    # ------------------------------------------------------------------
    print("\nStep 3.2: computing workload features (full population sweep-line) ...")
    assigned_full = full_df[full_df["Assignee_ID"].notna()].copy()
    assigned_full = assigned_full.groupby("Assignee_ID", group_keys=False).apply(
        compute_assignee_group_features
    )
    workload_lookup = assigned_full.set_index("ID")[
        ["assignee_prior_resolved_count", "assignee_concurrent_open"]
    ]

    res = res.join(workload_lookup, on="ID")

    # assignee_prior_delay_rate: computed within the resolved subset only,
    # sorted by Creation_Date per assignee, shifted so the current issue is
    # never included in its own history.
    res = res.sort_values("Creation_Date")
    res["assignee_prior_delay_rate"] = res.groupby("Assignee_ID")["delayed"].transform(
        lambda s: s.shift().expanding().mean()
    )

    res["is_unassigned"] = res["Assignee_ID"].isna().astype(int)
    # Explicit per spec: unassigned issues get concurrent workload = 0,
    # the other two workload features stay NaN.
    res.loc[res["is_unassigned"] == 1, "assignee_concurrent_open"] = 0

    # ------------------------------------------------------------------
    # Step 3.3: dependency features
    # ------------------------------------------------------------------
    print("Step 3.3: computing dependency features from Issue_Link ...")
    issue_link = pd.read_csv(os.path.join(RAW_DIR, "issue_link.csv"))

    n_links = issue_link.groupby("Issue_ID").size().rename("n_links")

    is_blocking = issue_link["Description"].fillna("").str.lower().apply(
        lambda d: any(kw in d for kw in BLOCKING_KEYWORDS)
    )
    n_blocking_links = (
        issue_link[is_blocking].groupby("Issue_ID").size().rename("n_blocking_links")
    )

    res = res.join(n_links, on="ID")
    res = res.join(n_blocking_links, on="ID")
    res["n_links"] = res["n_links"].fillna(0).astype(int)
    res["n_blocking_links"] = res["n_blocking_links"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Step 4: leakage tests
    # ------------------------------------------------------------------
    print("\n=== Step 4: leakage tests ===")

    feature_cols = [
        "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
        "assignee_prior_resolved_count", "assignee_prior_delay_rate",
        "assignee_concurrent_open", "is_unassigned",
        "n_links", "n_blocking_links",
    ]
    final_cols = ["ID"] + feature_cols + ["delayed"]
    final_df = res[final_cols].copy()

    leakage_pass = True

    # 1. correlation check
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(final_df[c])
    ]
    corrs = final_df[numeric_feature_cols].corrwith(final_df["delayed"])
    print("\nCorrelation with delayed:")
    print(corrs)
    max_corr = corrs.abs().max()
    corr_ok = max_corr <= 0.95
    leakage_pass &= corr_ok
    print(f"Max abs correlation: {max_corr:.4f} -> {'PASS' if corr_ok else 'FAIL'}")

    # 2. banned columns absent from feature matrix
    banned = {"Resolution_Date", "Resolution_Time_Minutes", "Status"}
    present_banned = banned.intersection(final_df.columns)
    banned_ok = len(present_banned) == 0
    leakage_pass &= banned_ok
    print(
        f"Banned columns present in feature matrix: {present_banned or 'none'} -> "
        f"{'PASS' if banned_ok else 'FAIL'}"
    )

    # 3. spot-check assignee_prior_delay_rate on a few issues
    print("\nSpot-check assignee_prior_delay_rate:")
    spot_ok = True
    sample_assignees = (
        res[res["Assignee_ID"].notna()]
        .groupby("Assignee_ID")
        .filter(lambda g: len(g) >= 5)["Assignee_ID"]
        .unique()
    )
    rng = np.random.default_rng(0)
    picks = rng.choice(sample_assignees, size=min(3, len(sample_assignees)), replace=False)
    for a in picks:
        g = res[res["Assignee_ID"] == a].sort_values("Creation_Date")
        idx = len(g) // 2
        row = g.iloc[idx]
        prior = g.iloc[:idx]
        expected = prior["delayed"].mean() if len(prior) else np.nan
        actual = row["assignee_prior_delay_rate"]
        ok = (pd.isna(expected) and pd.isna(actual)) or np.isclose(expected, actual)
        spot_ok &= ok
        print(
            f"  assignee={a} issue_id={row['ID']} creation={row['Creation_Date']} "
            f"n_prior={len(prior)} expected={expected} actual={actual} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )
    leakage_pass &= spot_ok

    print(f"\n=== Leakage tests overall: {'PASS' if leakage_pass else 'FAIL'} ===")
    if not leakage_pass:
        raise SystemExit("STOP: leakage test failed, not saving feature table.")

    # ------------------------------------------------------------------
    # Step 5: save and report
    # ------------------------------------------------------------------
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    print("\n=== Delay rate by normalized Priority ===")
    print(res.groupby("Priority_Normalized", observed=True)["delayed"].mean().sort_index())

    print("\n=== Delay rate by normalized Type ===")
    print(
        res.groupby("Type_Normalized", observed=True)["delayed"]
        .mean()
        .sort_values(ascending=False)
    )

    unassigned_rate = res["is_unassigned"].mean()
    print(f"\nis_unassigned=1 rate: {unassigned_rate:.4f} (expect ~0.429)")


if __name__ == "__main__":
    main()
