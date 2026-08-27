"""
Phase 8 foundation: chronological split, per-issue status timelines,
leakage-safe expected_duration_proxy, observation checkpoints, dwell-time
baseline / stall_ratio, and the 8E.0 trajectory sanity check.

Stops after the sanity-check plot -- DTW/clustering is a separate task.
"""

import heapq
import os

import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
FEATURES_DIR = "data/features"
REPORTS_DIR = "reports"

CHECKPOINTS_OUT = os.path.join(FEATURES_DIR, "phase8_checkpoints.csv")
SPLIT_OUT = os.path.join(FEATURES_DIR, "phase8_split.csv")
PLOT_OUT = os.path.join(REPORTS_DIR, "phase8_trajectory_sample.png")

MIN_GROUP_COUNT = 5
N_CHECKPOINTS = 10


# ---------------------------------------------------------------------
# Generic expanding (leakage-safe) hierarchical median machinery
# ---------------------------------------------------------------------
def build_expanding_median_lookup(df, group_cols, order_col, value_col):
    """For every group defined by group_cols, sort rows by order_col and
    build a running (streaming) median using a two-heap structure, so that
    median_at_k[k] = median of the first k rows (by order_col) in that
    group. Returns {group_key: (sorted_order_values, median_at_k)}.
    """
    lookup = {}
    if group_cols:
        grouped = df.groupby(group_cols, sort=False, observed=True)
    else:
        grouped = [((), df)]

    for key, g in grouped:
        g = g.sort_values(order_col, kind="mergesort")
        order_vals = g[order_col].values
        vals = g[value_col].to_numpy(dtype=float)

        n = len(vals)
        medians = np.empty(n + 1, dtype=float)
        medians[0] = np.nan

        lo, hi = [], []  # lo = max-heap (negated), hi = min-heap
        for i, v in enumerate(vals):
            if not lo or v <= -lo[0]:
                heapq.heappush(lo, -v)
            else:
                heapq.heappush(hi, v)

            if len(lo) > len(hi) + 1:
                heapq.heappush(hi, -heapq.heappop(lo))
            elif len(hi) > len(lo):
                heapq.heappush(lo, -heapq.heappop(hi))

            medians[i + 1] = -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2.0

        lookup[key if group_cols else ()] = (order_vals, medians)
    return lookup


def query_expanding_median(lookup, key, t, min_count=MIN_GROUP_COUNT):
    """Count of entries with order_val < t in this group, and the median of
    exactly those entries. Returns (count, median) -- median is NaN if
    count < min_count."""
    entry = lookup.get(key)
    if entry is None:
        return 0, np.nan
    order_vals, medians = entry
    k = np.searchsorted(order_vals, t, side="left")
    if k < min_count:
        return k, np.nan
    return k, medians[k]


def hierarchical_query(levels, t, min_count=MIN_GROUP_COUNT):
    """levels: list of (lookup, key) pairs, most specific first. Returns the
    first level's median with count >= min_count, falling back down the
    list; the LAST level is used regardless of its count (global fallback).
    """
    for lookup, key in levels[:-1]:
        count, med = query_expanding_median(lookup, key, t, min_count)
        if count >= min_count:
            return med
    lookup, key = levels[-1]
    entry = lookup.get(key)
    if entry is None:
        return np.nan
    order_vals, medians = entry
    k = np.searchsorted(order_vals, t, side="left")
    return medians[k]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading inputs ...")
    issue = pd.read_csv(
        os.path.join(RAW_DIR, "issue.csv"),
        usecols=["ID", "Project_ID", "Status", "Creation_Date", "Resolution_Date"],
        parse_dates=["Creation_Date", "Resolution_Date"],
    )
    feat = pd.read_csv(os.path.join(FEATURES_DIR, "phase6_feature_table.csv"))

    res = feat.merge(
        issue[["ID", "Status", "Creation_Date", "Resolution_Date"]], on="ID", how="left"
    )
    assert res["Creation_Date"].notna().all()
    assert res["Resolution_Date"].notna().all()
    print(f"  merged resolved population: {len(res)} rows")

    # ------------------------------------------------------------------
    # Step 1: chronological three-way split
    # ------------------------------------------------------------------
    res = res.sort_values("Creation_Date", kind="mergesort").reset_index(drop=True)
    n = len(res)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.80)) - n_train
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val))
    res["split"] = split
    print(
        f"\nStep 1: split -> train={sum(split=='train')} val={sum(split=='val')} "
        f"test={sum(split=='test')}"
    )
    res[["ID", "Creation_Date", "split"]].to_csv(SPLIT_OUT, index=False)
    print(f"  saved split assignment -> {SPLIT_OUT}")

    # ------------------------------------------------------------------
    # Step 2: reconstruct per-issue genuine status timelines
    # ------------------------------------------------------------------
    print("\nStep 2: reconstructing status timelines ...")
    cl = pd.read_csv(
        os.path.join(RAW_DIR, "change_log_status.csv"), parse_dates=["Creation_Date"]
    )
    # Rows where either side is null are Resolution-field entries mixed into
    # this Change_Type bucket (From=NaN/To=resolution value, or the reverse
    # when a resolution is cleared) -- not real workflow-status transitions.
    # Confirmed: To_String on null-From rows matches Issue.Resolution's
    # vocabulary almost exactly (Fixed, Won't Fix, Duplicate, ...).
    genuine = cl[cl["From_String"].notna() & cl["To_String"].notna()].copy()
    genuine = genuine[genuine["Issue_ID"].isin(res["ID"])]
    genuine = genuine.sort_values(["Issue_ID", "Creation_Date"], kind="mergesort")
    n_dropped = len(cl) - len(cl[cl["From_String"].notna() & cl["To_String"].notna()])
    print(
        f"  kept {len(genuine)} genuine transition rows "
        f"(dropped {n_dropped} resolution-field rows mixed into Change_Type='STATUS')"
    )

    timelines = {}
    for issue_id, g in genuine.groupby("Issue_ID", sort=False):
        timelines[issue_id] = (
            g["Creation_Date"].values,
            g["From_String"].values,
            g["To_String"].values,
        )
    missing_timeline = set(res["ID"]) - set(timelines.keys())
    print(f"  resolved issues with zero genuine transition rows: {len(missing_timeline)}")

    # ------------------------------------------------------------------
    # Step 3: expected_duration_proxy (leakage-safe, expanding, hierarchical)
    # ------------------------------------------------------------------
    print("\nStep 3: expected_duration_proxy ...")
    res["Type_Normalized"] = res["Type_Normalized"].astype("category")
    res["Priority_Normalized"] = res["Priority_Normalized"].astype("category")

    issue_full = pd.read_csv(
        os.path.join(RAW_DIR, "issue.csv"),
        usecols=["ID", "Resolution_Time_Minutes"],
    )
    res = res.merge(issue_full, on="ID", how="left")

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
    lvl5 = build_expanding_median_lookup(
        res, [], "Resolution_Date", "Resolution_Time_Minutes"
    )

    expected_duration = np.empty(len(res), dtype=float)
    creation_vals = res["Creation_Date"].values
    proj_vals = res["Project_ID"].values
    type_vals = res["Type_Normalized"].values
    prio_vals = res["Priority_Normalized"].values

    for i in range(len(res)):
        t = creation_vals[i]
        levels = [
            (lvl1, (proj_vals[i], type_vals[i], prio_vals[i])),
            (lvl2, (proj_vals[i], type_vals[i])),
            (lvl3, (proj_vals[i], prio_vals[i])),
            (lvl4, (proj_vals[i],)),
            (lvl5, ()),
        ]
        expected_duration[i] = hierarchical_query(levels, t)

    res["expected_duration_proxy"] = expected_duration
    n_nan_proxy = np.isnan(expected_duration).sum()
    print(
        f"  computed for all {len(res)} issues; {n_nan_proxy} NaN "
        f"(cold-start issues with zero prior resolved history anywhere)"
    )

    # ------------------------------------------------------------------
    # Build completed dwell-period corpus (needed for Step 5, but built
    # once up front from ALL resolved issues' timelines + Resolution_Date)
    # ------------------------------------------------------------------
    print("\nBuilding completed dwell-period corpus ...")
    proj_lookup = res.set_index("ID")["Project_ID"].to_dict()
    type_lookup = res.set_index("ID")["Type_Normalized"].to_dict()
    resolution_date_lookup = res.set_index("ID")["Resolution_Date"].to_dict()
    creation_date_lookup = res.set_index("ID")["Creation_Date"].to_dict()

    dwell_status, dwell_project, dwell_type = [], [], []
    dwell_start, dwell_end = [], []

    for issue_id, (times, froms, tos) in timelines.items():
        creation = creation_date_lookup[issue_id]
        resolution = resolution_date_lookup[issue_id]
        project_id = proj_lookup[issue_id]
        type_norm = type_lookup[issue_id]

        # segment 0: initial status, Creation_Date -> first transition
        seg_starts = [creation] + list(times)
        seg_ends = list(times) + [resolution]
        seg_statuses = [froms[0]] + list(tos)

        # Status logging sometimes continues after the recorded Resolution_Date
        # (e.g. administrative transitions like Resolved -> Closed happen later).
        # Drop segments that start at/after resolution entirely, and clip the
        # segment straddling resolution so it ends exactly at Resolution_Date --
        # otherwise these produce negative "durations".
        resolution_np = np.datetime64(resolution)
        for s, e, status in zip(seg_starts, seg_ends, seg_statuses):
            if np.datetime64(s) >= resolution_np:
                continue
            e = min(np.datetime64(e), resolution_np)
            dwell_status.append(status)
            dwell_project.append(project_id)
            dwell_type.append(type_norm)
            dwell_start.append(s)
            dwell_end.append(e)

    dwell_df = pd.DataFrame(
        {
            "Status": dwell_status,
            "Project_ID": dwell_project,
            "Type_Normalized": dwell_type,
            "start": pd.to_datetime(dwell_start),
            "end": pd.to_datetime(dwell_end),
        }
    )
    dwell_df["duration_minutes"] = (
        dwell_df["end"] - dwell_df["start"]
    ).dt.total_seconds() / 60.0
    n_negative = (dwell_df["duration_minutes"] < 0).sum()
    n_zero = (dwell_df["duration_minutes"] == 0).sum()
    print(
        f"  {len(dwell_df)} completed dwell segments; "
        f"{n_zero} zero-duration ({n_zero / len(dwell_df):.2%}, mostly bulk-migrated "
        f"issues with identical Change_Log timestamps), {n_negative} negative (should be 0)"
    )
    dwell_df = dwell_df[dwell_df["duration_minutes"] >= 0].copy()

    dwellA = build_expanding_median_lookup(
        dwell_df, ["Status", "Project_ID", "Type_Normalized"], "end", "duration_minutes"
    )
    dwellB = build_expanding_median_lookup(
        dwell_df, ["Status", "Project_ID"], "end", "duration_minutes"
    )
    dwellC = build_expanding_median_lookup(
        dwell_df, ["Status", "Type_Normalized"], "end", "duration_minutes"
    )
    dwellD = build_expanding_median_lookup(dwell_df, ["Status"], "end", "duration_minutes")

    # ------------------------------------------------------------------
    # Step 4 + 5: checkpoints, current status, dwell baseline, stall_ratio
    # ------------------------------------------------------------------
    print("\nStep 4/5: building checkpoints ...")
    rows = []
    n_zero_baseline = 0
    n_missing_baseline = 0

    for i in range(len(res)):
        issue_id = res["ID"].values[i]
        creation = creation_vals[i]
        resolution = resolution_date_lookup[issue_id]
        proxy = expected_duration[i]
        if np.isnan(proxy) or proxy <= 0:
            continue

        project_id = proj_vals[i]
        type_norm = type_vals[i]

        tl = timelines.get(issue_id)
        if tl is not None:
            trans_times, trans_from, trans_to = tl
        else:
            trans_times, trans_from, trans_to = (
                np.array([], dtype="datetime64[ns]"),
                np.array([]),
                np.array([]),
            )
        initial_status = trans_from[0] if len(trans_from) else res["Status"].values[i]

        still_active_past_m10 = False
        for k in range(1, N_CHECKPOINTS + 1):
            elapsed_minutes = (k / N_CHECKPOINTS) * proxy
            checkpoint_time = creation + pd.Timedelta(minutes=elapsed_minutes)

            if resolution <= checkpoint_time:
                continue  # already resolved by this checkpoint -- skip (no row)

            if k == N_CHECKPOINTS:
                still_active_past_m10 = True

            if len(trans_times):
                idx = np.searchsorted(trans_times, np.datetime64(checkpoint_time), side="right") - 1
            else:
                idx = -1

            if idx == -1:
                current_status = initial_status
                last_change_time = creation
            else:
                current_status = trans_to[idx]
                last_change_time = trans_times[idx]

            levels = [
                (dwellA, (current_status, project_id, type_norm)),
                (dwellB, (current_status, project_id)),
                (dwellC, (current_status, type_norm)),
                (dwellD, (current_status,)),
            ]
            baseline = None
            for lookup, key in levels:
                count, med = query_expanding_median(lookup, key, np.datetime64(checkpoint_time))
                if count >= MIN_GROUP_COUNT:
                    baseline = med
                    break

            time_in_status = (
                pd.Timestamp(checkpoint_time) - pd.Timestamp(last_change_time)
            ).total_seconds() / 60.0

            if baseline is None:
                n_missing_baseline += 1
                stall_ratio = np.nan
            elif baseline <= 0:
                n_zero_baseline += 1
                stall_ratio = np.nan
            else:
                stall_ratio = time_in_status / baseline

            rows.append(
                (
                    issue_id, k, elapsed_minutes, current_status, last_change_time,
                    stall_ratio, project_id, type_norm, res["split"].values[i],
                )
            )

        if still_active_past_m10:
            pass  # flag applied below via a separate per-issue pass

    checkpoints = pd.DataFrame(
        rows,
        columns=[
            "Issue_ID", "checkpoint_index", "elapsed_minutes", "current_status",
            "last_status_change_time", "stall_ratio", "Project_ID", "Type_Normalized", "split",
        ],
    )

    # overran_expected_duration: issue still unresolved past its own M10
    m10_time = pd.Series(creation_vals, index=res["ID"].values) + pd.to_timedelta(
        expected_duration, unit="m"
    )
    resolution_series = pd.Series(
        [resolution_date_lookup[i] for i in res["ID"].values], index=res["ID"].values
    )
    overran = (resolution_series > m10_time).astype(int)
    overran.name = "overran_expected_duration"
    checkpoints = checkpoints.merge(overran, left_on="Issue_ID", right_index=True, how="left")

    checkpoints.to_csv(CHECKPOINTS_OUT, index=False)
    print(f"  saved {len(checkpoints)} checkpoint rows -> {CHECKPOINTS_OUT}")
    print(
        f"  stall_ratio NaN due to missing baseline group: {n_missing_baseline}; "
        f"due to zero/negative baseline: {n_zero_baseline}"
    )

    n_issues_with_checkpoints = checkpoints["Issue_ID"].nunique()
    avg_checkpoints = checkpoints.groupby("Issue_ID").size().mean()
    print(
        f"\nIssues with >=1 checkpoint: {n_issues_with_checkpoints} / {len(res)}; "
        f"avg checkpoints/issue: {avg_checkpoints:.2f}"
    )
    print(f"overran_expected_duration=1 rate: {overran.mean():.4f}")

    # ------------------------------------------------------------------
    # Step 6: 8E.0 trajectory sanity check
    # ------------------------------------------------------------------
    print("\nStep 6: building sanity-check plot ...")
    plot_sanity_check(checkpoints, res)

    print("\n=== Done. Stopping after Step 6 per instructions. ===")


def plot_sanity_check(checkpoints, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_ids = res.loc[res["split"] == "train", "ID"]
    eligible = checkpoints[checkpoints["Issue_ID"].isin(train_ids)]

    hand_picked_ids = []
    meta = res.set_index("ID")
    seen_combo = set()
    for _, row in res.sample(frac=1.0, random_state=42).iterrows():
        combo = (row["Project_ID"], row["Type_Normalized"], row["delayed"])
        if combo in seen_combo:
            continue
        if row["ID"] not in checkpoints["Issue_ID"].values:
            continue
        seen_combo.add(combo)
        hand_picked_ids.append(row["ID"])
        if len(hand_picked_ids) >= 8:
            break

    rng = np.random.default_rng(42)
    candidate_ids = eligible["Issue_ID"].unique()
    random_ids = rng.choice(
        candidate_ids, size=min(40, len(candidate_ids)), replace=False
    )

    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#898781"
    gridline = "#e1e0d9"
    baseline_color = "#c3c2b7"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for iid in random_ids:
        g = checkpoints[checkpoints["Issue_ID"] == iid].sort_values("checkpoint_index")
        ax.plot(g["checkpoint_index"], g["stall_ratio"], color=ink_muted, alpha=0.35, linewidth=1)

    # Fixed-order categorical palette (validated: dataviz skill references/palette.md)
    palette = [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948",
    ]
    for i, iid in enumerate(hand_picked_ids):
        g = checkpoints[checkpoints["Issue_ID"] == iid].sort_values("checkpoint_index")
        r = meta.loc[iid]
        label = f"{iid} (proj {r['Project_ID']}, {r['Type_Normalized']}, delayed={r['delayed']})"
        ax.plot(
            g["checkpoint_index"], g["stall_ratio"],
            color=palette[i % len(palette)], linewidth=2.2, marker="o", markersize=4,
            label=label,
        )

    ax.axhline(1.0, color=baseline_color, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Checkpoint index (1-10)", color=ink_secondary)
    ax.set_ylabel("stall_ratio (log scale)", color=ink_secondary)
    ax.set_title("Phase 8: stall_ratio trajectories (8E.0 sanity check)", color=ink_primary)
    ax.set_xticks(range(1, 11))
    ax.tick_params(colors=ink_muted)
    ax.grid(color=gridline, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(gridline)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), facecolor=surface, edgecolor=gridline, labelcolor=ink_primary)
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, facecolor=surface)
    print(f"  saved plot -> {PLOT_OUT}")

    print("\nHand-picked issues:", hand_picked_ids)
    print("Random training sample size:", len(random_ids))


if __name__ == "__main__":
    main()
