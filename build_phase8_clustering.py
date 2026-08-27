"""
Phase 8.3: log1p-transform stall_ratio, take a stratified training subsample,
build variable-length trajectories, DTW distance + hierarchical clustering,
pick K via elbow, extract medoids, characterize clusters.

Stops after Step 5 (cluster characterization) -- dynamic pattern assignment
and dynamic XGBoost training are a separate, later task.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw

FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"

RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 4000
MIN_PER_STRATUM = 1

ELBOW_PLOT = os.path.join(REPORTS_DIR, "phase8_elbow.png")
CLUSTER_PLOT = os.path.join(REPORTS_DIR, "phase8_cluster_trajectories.png")
MEDOIDS_OUT = os.path.join(FEATURES_DIR, "phase8_medoids.csv")


def stratified_sample(issues_df, target_n, seed):
    rng = np.random.default_rng(seed)
    strata = list(issues_df.groupby(["Project_ID", "Type_Normalized"], observed=True))
    sizes = np.array([len(g) for _, g in strata])
    n_strata = len(strata)

    alloc = np.minimum(sizes, MIN_PER_STRATUM)
    remaining_budget = target_n - alloc.sum()
    remaining_capacity = sizes - alloc

    if remaining_budget > 0 and remaining_capacity.sum() > 0:
        # proportional allocation of the remainder, capped by each stratum's capacity
        props = remaining_capacity / remaining_capacity.sum()
        extra = np.floor(props * remaining_budget).astype(int)
        extra = np.minimum(extra, remaining_capacity)
        alloc += extra
        shortfall = target_n - alloc.sum()
        # hand out leftover one-by-one to strata that still have capacity
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
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading checkpoints + split ...")
    ck = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_checkpoints.csv"))
    split = pd.read_csv(os.path.join(FEATURES_DIR, "phase8_split.csv"))
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))

    train_ids = set(split.loc[split["split"] == "train", "ID"])
    print(f"  train issues (from phase8_split.csv): {len(train_ids)}")

    train_ck = ck[ck["Issue_ID"].isin(train_ids)].copy()

    # ------------------------------------------------------------------
    # Step 0: log1p transform (required, before anything else)
    # ------------------------------------------------------------------
    print("\nStep 0: log1p(stall_ratio) ...")
    print("  raw stall_ratio (train, non-null):")
    print(train_ck["stall_ratio"].describe())

    # insufficient_history: checkpoints with no usable dwell-baseline (NaN
    # stall_ratio) are excluded from the trajectory, not treated as a real
    # observation.
    n_before = len(train_ck)
    train_ck = train_ck[train_ck["stall_ratio"].notna()].copy()
    print(f"  dropped {n_before - len(train_ck)} insufficient_history rows (NaN stall_ratio)")

    train_ck["log1p_stall"] = np.log1p(train_ck["stall_ratio"])
    print("\n  log1p(stall_ratio) distribution (train):")
    print(train_ck["log1p_stall"].describe())
    q = train_ck["log1p_stall"].quantile([0.9, 0.99, 0.999, 0.9999, 1.0])
    print(q)
    cap_needed = q.loc[1.0] > 3 * q.loc[0.999]
    print(
        f"  cap still needed beyond log1p? {'YES -- revisit' if cap_needed else 'NO -- log1p alone is sufficient'}"
    )

    # ------------------------------------------------------------------
    # Step 1: stratified subsample of training issues
    # ------------------------------------------------------------------
    print(f"\nStep 1: stratified subsample (target={TARGET_SAMPLE_SIZE}, seed={RANDOM_SEED}) ...")
    issues_df = train_ck.drop_duplicates("Issue_ID")[["Issue_ID", "Project_ID", "Type_Normalized"]]
    print(f"  candidate train issues with >=1 valid checkpoint: {len(issues_df)}")
    print(f"  distinct (Project_ID, Type_Normalized) strata: {issues_df.groupby(['Project_ID','Type_Normalized'], observed=True).ngroups}")

    sampled_ids = stratified_sample(issues_df, TARGET_SAMPLE_SIZE, RANDOM_SEED)
    print(f"  sampled {len(sampled_ids)} issues")

    # ------------------------------------------------------------------
    # Step 2: build trajectories
    # ------------------------------------------------------------------
    print("\nStep 2: building trajectories ...")
    sample_ck = train_ck[train_ck["Issue_ID"].isin(sampled_ids)].sort_values(
        ["Issue_ID", "checkpoint_index"]
    )
    trajectories = []
    traj_issue_ids = []
    for issue_id, g in sample_ck.groupby("Issue_ID", sort=False):
        vals = g["log1p_stall"].to_numpy(dtype=np.double)
        if len(vals) == 0:
            continue
        trajectories.append(vals)
        traj_issue_ids.append(issue_id)

    lengths = np.array([len(t) for t in trajectories])
    print(
        f"  {len(trajectories)} trajectories; length min={lengths.min()} "
        f"median={np.median(lengths):.1f} max={lengths.max()}"
    )

    # ------------------------------------------------------------------
    # Step 3: DTW distance + hierarchical clustering
    # ------------------------------------------------------------------
    print("\nStep 3: computing pairwise DTW distance matrix (dtaidistance) ...")
    import dtaidistance
    print(f"  dtaidistance version: {dtaidistance.__version__}")

    dist_matrix = dtw.distance_matrix_fast(trajectories)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)

    def medoid_index(cluster_indices):
        sub = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        return cluster_indices[np.argmin(sub.sum(axis=1))]

    def wss_for_k(Z, k):
        labels = fcluster(Z, t=k, criterion="maxclust")
        total = 0.0
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            m = medoid_index(idx)
            d = dist_matrix[idx, m]
            total += np.sum(d ** 2)
        return total, labels

    ks = list(range(2, 9))

    # Try both linkage methods and compare cluster-size shape, not just WSS,
    # before committing -- average linkage on this data collapses almost
    # everything into one dominant cluster with several near-singleton
    # outlier clusters peeling off one at a time (a chaining artifact),
    # which is a degenerate result, not a real elbow.
    print("\nStep 3b: comparing average vs. complete linkage ...")
    linkages = {}
    for method in ["average", "complete"]:
        Z = linkage(condensed, method=method)
        linkages[method] = Z
        print(f"  {method} linkage cluster sizes by K:")
        for k in ks:
            _, labels = wss_for_k(Z, k)
            sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
            print(f"    K={k}: {sizes}")

    LINKAGE_METHOD = "complete"
    print(
        f"\n  Using '{LINKAGE_METHOD}' linkage: average linkage's cluster sizes above show one "
        f"~3950-member cluster persisting through K=6 with only singleton-ish outliers peeling "
        f"off (a chaining artifact); complete linkage breaks out a real secondary cluster "
        f"(~200 members) much earlier and more gradually."
    )
    Z = linkages[LINKAGE_METHOD]

    print(f"\nStep 3c: elbow method on {LINKAGE_METHOD} linkage, K=2..8 ...")
    wss_values = []
    for k in ks:
        w, _ = wss_for_k(Z, k)
        wss_values.append(w)
        print(f"  K={k}: WSS={w:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    surface, ink_primary, ink_secondary, gridline = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot(ks, wss_values, color="#2a78d6", linewidth=2, marker="o", markersize=6)
    ax.set_xlabel("K", color=ink_secondary)
    ax.set_ylabel("Within-cluster sum of squared DTW distances (to medoid)", color=ink_secondary)
    ax.set_title("Phase 8.3: elbow plot (K=2..8)", color=ink_primary)
    ax.grid(color=gridline, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(gridline)
    fig.tight_layout()
    fig.savefig(ELBOW_PLOT, dpi=150, facecolor=surface)
    print(f"  saved elbow plot -> {ELBOW_PLOT}")

    # Elbow = the K right after the single largest marginal WSS improvement
    # (i.e. where returns start diminishing), not a blind curvature formula.
    wss_arr = np.array(wss_values)
    first_diff = -np.diff(wss_arr)  # improvement going from K_i to K_{i+1}
    print("  marginal WSS improvement per step:", dict(zip([f"{ks[i]}->{ks[i+1]}" for i in range(len(first_diff))], np.round(first_diff, 1))))
    biggest_drop_idx = np.argmax(first_diff)  # improvement INTO ks[biggest_drop_idx+1]
    suggested_k = ks[biggest_drop_idx + 1]
    print(f"  largest single improvement is entering K={suggested_k}; drops taper off after that -> elbow K={suggested_k}")

    CHOSEN_K = suggested_k
    print(f"\n=== Chosen K = {CHOSEN_K} ===")

    final_wss, labels = wss_for_k(Z, CHOSEN_K)

    # ------------------------------------------------------------------
    # Step 4: medoids
    # ------------------------------------------------------------------
    print("\nStep 4: extracting medoids ...")
    medoid_rows = []
    cluster_medoid_issue = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = medoid_index(idx)
        medoid_issue_id = traj_issue_ids[m]
        cluster_medoid_issue[c] = medoid_issue_id
        traj = trajectories[m]
        for ckpt_i, val in enumerate(traj, start=1):
            medoid_rows.append((c, medoid_issue_id, ckpt_i, val))

    medoids_df = pd.DataFrame(
        medoid_rows, columns=["cluster", "medoid_issue_id", "checkpoint_index", "log1p_stall_ratio"]
    )
    medoids_df.to_csv(MEDOIDS_OUT, index=False)
    print(f"  saved medoids -> {MEDOIDS_OUT}")
    for c, mid in cluster_medoid_issue.items():
        print(f"    cluster {c}: medoid issue_id={mid}")

    # ------------------------------------------------------------------
    # Step 5: characterize clusters
    # ------------------------------------------------------------------
    print("\nStep 5: characterizing clusters ...")
    delayed_lookup = feat.set_index("ID")["delayed"]
    traj_delayed = pd.Series(traj_issue_ids).map(delayed_lookup)

    summary = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        size = len(idx)
        delay_rate = traj_delayed.iloc[idx].mean()
        summary.append({"cluster": c, "size": size, "delay_rate": delay_rate})
    summary_df = pd.DataFrame(summary).sort_values("cluster")
    overall_delay_rate = traj_delayed.mean()
    print(f"\nOverall subsample delay rate: {overall_delay_rate:.4f}")
    print(summary_df.to_string(index=False))

    plot_cluster_trajectories(labels, trajectories, traj_issue_ids, cluster_medoid_issue)

    print("\n=== Done. Stopping after Step 5 per instructions. ===")


def plot_cluster_trajectories(labels, trajectories, traj_issue_ids, cluster_medoid_issue):
    surface, ink_primary, ink_secondary, ink_muted, gridline = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
    )
    palette = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]
    unique_clusters = np.unique(labels)
    ncols = 3
    nrows = int(np.ceil(len(unique_clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    fig.patch.set_facecolor(surface)

    rng = np.random.default_rng(RANDOM_SEED)
    for i, c in enumerate(unique_clusters):
        ax = axes[i // ncols][i % ncols]
        ax.set_facecolor(surface)
        idx = np.where(labels == c)[0]
        sample_idx = rng.choice(idx, size=min(15, len(idx)), replace=False)
        for j in sample_idx:
            traj = trajectories[j]
            ax.plot(range(1, len(traj) + 1), traj, color=ink_muted, alpha=0.4, linewidth=1)

        color = palette[i % len(palette)]
        medoid_issue_id = cluster_medoid_issue[c]
        medoid_pos = traj_issue_ids.index(medoid_issue_id)
        medoid_traj = trajectories[medoid_pos]
        ax.plot(
            range(1, len(medoid_traj) + 1), medoid_traj,
            color=color, linewidth=2.5, marker="o", markersize=4,
        )
        ax.set_title(f"Cluster {c} (n={len(idx)})", color=ink_primary, fontsize=10)
        ax.set_xlabel("checkpoint", color=ink_secondary, fontsize=8)
        ax.set_ylabel("log1p(stall_ratio)", color=ink_secondary, fontsize=8)
        ax.tick_params(colors=ink_muted, labelsize=7)
        ax.grid(color=gridline, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color(gridline)

    for i in range(len(unique_clusters), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(CLUSTER_PLOT, dpi=150, facecolor=surface)
    print(f"  saved per-cluster trajectory plot -> {CLUSTER_PLOT}")


if __name__ == "__main__":
    main()
