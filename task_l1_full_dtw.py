"""
Task L1 (future-work #6): does the "93% in one cluster" result survive at FULL scale?

Phase 8.3 clustered a 4,000-issue stratified subsample (DTW, complete linkage,
K=4) and found one cluster holding 93%. The negative pattern-transfer result
rests on that. This script tests it on all training trajectories:

  A. assign ALL train trajectories to the 4 published medoids -> size distribution
  B. re-run the published method (complete-linkage hierarchical) on bigger and
     different subsamples (4k x 3 seeds, 8k, 12k) -> is the dominance stable?
  C. CLARA k-medoids (DTW): 5 subsamples of 5,000, K = 2..8, every candidate
     medoid set scored on the FULL training population; elbow on full-population
     cost -> cluster sizes on all training issues
  D. a SHAPE variant (z-normalised trajectories), which ignores the stall level and
     looks only at the trajectory's shape -- the closest analogue to Kula's patterns
  E. the best full-scale clustering is turned into dynamic pattern labels (prefix
     M1..Mt, t >= 3) and v5b is retrained with / without it (3 seeds, paired
     bootstrap) to re-test the pattern contribution

DTW = dtaidistance definition (sqrt of summed squared differences along the
optimal warping path, no window), re-implemented in numba for batch speed and
checked against dtaidistance.

Also documents a quirk in build_phase8_dynamic_model.py: the partial trajectory
used for dynamic assignment skipped checkpoints 1-2 (they `continue` before the
prefix is appended), so labels used M3..Mt, not M1..Mt as written in the paper.
"""

import json
import numpy as np
import pandas as pd
import numba
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_DIR = "data/features"
REPORTS = "reports"
MAXLEN = 10
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ DTW (numba)
@numba.njit(cache=True)
def dtw_pair(a, la, b, lb):
    INF = 1e300
    prev = np.full(lb + 1, INF)
    prev[0] = 0.0
    cur = np.full(lb + 1, INF)
    for i in range(1, la + 1):
        cur[0] = INF
        for j in range(1, lb + 1):
            d = (a[i - 1] - b[j - 1]) ** 2
            m = prev[j - 1]
            if prev[j] < m:
                m = prev[j]
            if cur[j - 1] < m:
                m = cur[j - 1]
            cur[j] = d + m
        for j in range(lb + 1):
            prev[j] = cur[j]
    return np.sqrt(prev[lb])


@numba.njit(parallel=True, cache=True)
def dtw_to_medoids(X, L, M, LM):
    n, k = X.shape[0], M.shape[0]
    out = np.empty((n, k))
    for i in numba.prange(n):
        for c in range(k):
            out[i, c] = dtw_pair(X[i], L[i], M[c], LM[c])
    return out


@numba.njit(parallel=True, cache=True)
def dtw_matrix(X, L):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in numba.prange(n):
        for j in range(i + 1, n):
            d = dtw_pair(X[i], L[i], X[j], L[j])
            D[i, j] = d
            D[j, i] = d
    return D


def pad(trajs):
    X = np.zeros((len(trajs), MAXLEN))
    L = np.zeros(len(trajs), dtype=np.int64)
    for i, t in enumerate(trajs):
        X[i, :len(t)] = t
        L[i] = len(t)
    return X, L


# ------------------------------------------------------------------ k-medoids
def kmedoids(D, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    med = [rng.integers(n)]
    for _ in range(1, k):                                  # k-medoids++ init
        dmin = D[:, med].min(axis=1)
        p = dmin ** 2 / (dmin ** 2).sum()
        med.append(rng.choice(n, p=p))
    med = np.array(med)
    for _ in range(iters):
        lab = D[:, med].argmin(axis=1)
        new = med.copy()
        for c in range(k):
            idx = np.where(lab == c)[0]
            if len(idx):
                new[c] = idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]
        if (new == med).all():
            break
        med = new
    return med


def sizes(lab, k):
    s = np.bincount(lab, minlength=k) / len(lab)
    return sorted(s.round(4).tolist(), reverse=True)


def main():
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(str(s))

    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv",
                     usecols=["Issue_ID", "checkpoint_index", "stall_ratio", "split"])
    ck["v"] = np.log1p(ck["stall_ratio"])
    tr = ck[(ck.split == "train") & ck.v.notna()].sort_values(["Issue_ID", "checkpoint_index"])
    groups = tr.groupby("Issue_ID")["v"].apply(np.array)
    ids = groups.index.to_numpy()
    X, L = pad(groups.tolist())
    say(f"train trajectories (>=1 valid checkpoint): {len(ids)}; median length {np.median(L):.0f}")

    # validate numba DTW against dtaidistance
    for _ in range(200):
        i, j = RNG.integers(len(ids), size=2)
        a, b = X[i, :L[i]], X[j, :L[j]]
        assert abs(dtw_pair(a, L[i], b, L[j]) - dtw.distance(a, b)) < 1e-8
    say("numba DTW matches dtaidistance on 200 random pairs")

    # ---------------- A. published medoids on the full population
    med = pd.read_csv(f"{FEATURES_DIR}/phase8_medoids.csv")
    old = [g.sort_values("checkpoint_index")["log1p_stall_ratio"].to_numpy() for _, g in med.groupby("cluster")]
    MX, ML = pad(old)
    labA = dtw_to_medoids(X, L, MX, ML).argmin(axis=1)
    sA = np.bincount(labA, minlength=4) / len(labA)
    say(f"\nA. published 4 medoids -> full training population ({len(ids)} issues): "
        + ", ".join(f"cluster_{c+1} {sA[c]:.1%}" for c in range(4)))

    # ---------------- B. published method on other / bigger samples
    say("\nB. complete-linkage hierarchical (published method), K=4, on different samples:")
    rows_b = []
    for n, seed in [(4000, 1), (4000, 2), (4000, 3), (8000, 1), (12000, 1)]:
        idx = np.random.default_rng(seed).choice(len(ids), n, replace=False)
        D = dtw_matrix(X[idx], L[idx])
        Z = linkage(squareform(D, checks=False), method="complete")
        lab = fcluster(Z, t=4, criterion="maxclust") - 1
        s = sizes(lab, 4)
        rows_b.append({"n": n, "seed": seed, "largest": s[0], "sizes": s})
        say(f"   n={n:5d} seed={seed}: sizes {s}")
        del D, Z
    pd.DataFrame(rows_b).to_csv(f"{REPORTS}/l1_hierarchical_samples.csv", index=False)

    # ---------------- C/D. CLARA k-medoids on full population, level and shape
    results, best_sets = [], {}
    variants = {"level (log1p stall)": (X, L)}
    Xs = X.copy()
    for i in range(len(Xs)):                               # z-normalise each trajectory (shape only)
        t = Xs[i, :L[i]]
        sd = t.std()
        Xs[i, :L[i]] = (t - t.mean()) / sd if sd > 1e-9 else 0.0
    variants["shape (z-normalised)"] = (Xs, L)
    for vname, (XX, LL) in variants.items():
        say(f"\nC. CLARA k-medoids, variant = {vname}")
        subs = [np.random.default_rng(100 + s).choice(len(ids), 5000, replace=False) for s in range(5)]
        Ds = [dtw_matrix(XX[s], LL[s]) for s in subs]
        for k in range(2, 9):
            best = None
            for si, (s, D) in enumerate(zip(subs, Ds)):
                m = s[kmedoids(D, k, seed=si)]
                dist = dtw_to_medoids(XX, LL, XX[m], LL[m])
                cost = dist.min(axis=1).sum()
                if best is None or cost < best[0]:
                    best = (cost, m, dist.argmin(axis=1))
            cost, m, lab = best
            s = sizes(lab, k)
            results.append({"variant": vname, "K": k, "full_population_cost": cost, "largest_cluster": s[0],
                            "sizes": s, "medoid_issue_ids": ids[m].tolist()})
            best_sets[(vname, k)] = (m, lab)
            say(f"   K={k}: full-pop cost {cost:,.0f}; sizes {s}")
        del Ds
    res = pd.DataFrame(results)
    res.to_csv(f"{REPORTS}/l1_clara_full_population.csv", index=False)

    # elbow per variant: K right after the largest relative cost drop
    chosen = {}
    for vname, g in res.groupby("variant"):
        c = g.sort_values("K")["full_population_cost"].to_numpy()
        drops = (c[:-1] - c[1:]) / c[:-1]
        k = int(g.sort_values("K")["K"].to_numpy()[1:][np.argmax(drops)])
        chosen[vname] = k
        say(f"\n   elbow ({vname}): K={k}; sizes {g[g.K == k]['sizes'].iloc[0]}")

    # delay rate per cluster (train) for the chosen clusterings
    lab_df = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv", usecols=["ID", "delayed_v2"]).set_index("ID")
    y = lab_df.reindex(ids)["delayed_v2"].to_numpy()
    profile = []
    for vname, k in chosen.items():
        m, lab = best_sets[(vname, k)]
        for c in range(k):
            msk = lab == c
            profile.append({"variant": vname, "K": k, "cluster": c + 1, "share": msk.mean(),
                            "delay_rate": y[msk].mean(), "medoid_issue_id": int(ids[m[c]]),
                            "medoid_trajectory": np.round(variants[vname][0][m[c], :L[m[c]]], 3).tolist()})
    prof = pd.DataFrame(profile)
    prof.to_csv(f"{REPORTS}/l1_cluster_profiles.csv", index=False)
    say("\nCluster profiles (chosen K):\n" + prof[["variant", "cluster", "share", "delay_rate"]].round(3).to_string(index=False))

    # save medoids for E
    out = {}
    for vname, k in chosen.items():
        m, _ = best_sets[(vname, k)]
        out[vname] = {"K": k, "medoid_issue_ids": ids[m].tolist(),
                      "medoids": [variants[vname][0][i, :L[i]].tolist() for i in m]}
    json.dump(out, open(f"{FEATURES_DIR}/l1_full_scale_medoids.json", "w"), indent=2)
    open(f"{REPORTS}/l1_run_summary.txt", "w").write("\n".join(log))

    # plot: elbow + medoids
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for vname, g in res.groupby("variant"):
        g = g.sort_values("K")
        axes[0].plot(g["K"], g["full_population_cost"] / g["full_population_cost"].iloc[0], marker="o", label=vname)
    axes[0].set_title("CLARA cost on ALL training issues (relative to K=2)")
    axes[0].set_xlabel("K")
    axes[0].legend(fontsize=8)
    for ax, (vname, k) in zip(axes[1:], chosen.items()):
        m, lab = best_sets[(vname, k)]
        XX = variants[vname][0]
        for c in range(k):
            ax.plot(range(1, L[m[c]] + 1), XX[m[c], :L[m[c]]], marker="o",
                    label=f"cluster {c+1} ({(lab == c).mean():.0%}, delay {y[lab == c].mean():.2f})")
        ax.set_title(f"Medoids, {vname}, K={k}")
        ax.set_xlabel("checkpoint")
        ax.legend(fontsize=7)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{REPORTS}/l1_full_scale_clusters.png", dpi=140)


if __name__ == "__main__":
    main()
