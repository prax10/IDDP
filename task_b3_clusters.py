"""
Task B3, Step 2.5: recompute the four clusters' delay rates under
delayed_v2. Clusters themselves are unchanged -- deterministically
reproduce the exact same stratified subsample + DTW clustering from
build_phase8_clustering.py (same seed, same code path), then recompute
delay rate per cluster with the new label.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"

RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 4000
MIN_PER_STRATUM = 1
CHOSEN_K = 4
LINKAGE_METHOD = "complete"


def stratified_sample(issues_df, target_n, seed):
    rng = np.random.default_rng(seed)
    strata = list(issues_df.groupby(["Project_ID", "Type_Normalized"], observed=True))
    sizes = np.array([len(g) for _, g in strata])
    n_strata = len(strata)

    alloc = np.minimum(sizes, MIN_PER_STRATUM)
    remaining_budget = target_n - alloc.sum()
    remaining_capacity = sizes - alloc

    if remaining_budget > 0 and remaining_capacity.sum() > 0:
        props = remaining_capacity / remaining_capacity.sum()
        extra = np.floor(props * remaining_budget).astype(int)
        extra = np.minimum(extra, remaining_capacity)
        alloc += extra
        shortfall = target_n - alloc.sum()
        capacity_left = sizes - alloc
        order = np.argsort(-capacity_left)
        idx = 0
        while shortfall > 0 and capacity_left[order].sum() > 0:
            s = order[idx % n_strata]
            if capacity_left[s] > 0:
                alloc[s] += 1
                capacity_left[s] -= 1
                shortfall -= 1
            idx += 1

    sampled_ids = []
    for (key, g), k in zip(strata, alloc):
        k = int(min(k, len(g)))
        if k <= 0:
            continue
        chosen = rng.choice(g["Issue_ID"].values, size=k, replace=False)
        sampled_ids.extend(chosen.tolist())
    return sampled_ids


def main():
    print("=== Task B3: recompute cluster delay rates under delayed_v2 ===")
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    feat_v3 = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")

    train_ids = set(split.loc[split["split"] == "train", "ID"])
    train_ck = ck[ck["Issue_ID"].isin(train_ids)].copy()
    train_ck = train_ck[train_ck["stall_ratio"].notna()].copy()
    train_ck["log1p_stall"] = np.log1p(train_ck["stall_ratio"])

    issues_df = train_ck.drop_duplicates("Issue_ID")[["Issue_ID", "Project_ID", "Type_Normalized"]]
    sampled_ids = stratified_sample(issues_df, TARGET_SAMPLE_SIZE, RANDOM_SEED)
    print(f"  reproduced subsample: {len(sampled_ids)} issues (same seed as 8.3)")

    sample_ck = train_ck[train_ck["Issue_ID"].isin(sampled_ids)].sort_values(["Issue_ID", "checkpoint_index"])
    trajectories, traj_issue_ids = [], []
    for issue_id, g in sample_ck.groupby("Issue_ID", sort=False):
        vals = g["log1p_stall"].to_numpy(dtype=np.double)
        if len(vals) == 0:
            continue
        trajectories.append(vals)
        traj_issue_ids.append(issue_id)
    print(f"  {len(trajectories)} trajectories reproduced")

    dist_matrix = dtw.distance_matrix_fast(trajectories)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method=LINKAGE_METHOD)
    labels = fcluster(Z, t=CHOSEN_K, criterion="maxclust")

    medoids_df = pd.read_csv(f"{FEATURES_DIR}/phase8_medoids.csv")
    expected_medoid_ids = set(medoids_df["medoid_issue_id"].unique())

    def medoid_index(cluster_indices):
        sub = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        return cluster_indices[np.argmin(sub.sum(axis=1))]

    reproduced_medoid_ids = set()
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = medoid_index(idx)
        reproduced_medoid_ids.add(traj_issue_ids[m])

    match = reproduced_medoid_ids == expected_medoid_ids
    print(f"  reproduced medoid IDs match saved medoids exactly: {match}")
    if not match:
        print(f"    expected: {expected_medoid_ids}")
        print(f"    got:      {reproduced_medoid_ids}")

    delayed_v2_lookup = feat_v3.set_index("ID")["delayed_v2"]
    delayed_v1_lookup = feat_v3.set_index("ID")["delayed"]
    traj_delayed_v2 = pd.Series(traj_issue_ids).map(delayed_v2_lookup)
    traj_delayed_v1 = pd.Series(traj_issue_ids).map(delayed_v1_lookup)

    summary = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        size = len(idx)
        summary.append({
            "cluster": c, "size": size,
            "delay_rate_old_label": traj_delayed_v1.iloc[idx].mean(),
            "delay_rate_new_label_v2": traj_delayed_v2.iloc[idx].mean(),
        })
    summary_df = pd.DataFrame(summary).sort_values("cluster")
    print(f"\n  overall subsample delay rate: old={traj_delayed_v1.mean():.4f} new={traj_delayed_v2.mean():.4f}")
    print(summary_df.to_string(index=False))

    out_path = f"{REPORTS_DIR}/phase8_cluster_delay_rates_v3.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")
    return summary_df


if __name__ == "__main__":
    main()
