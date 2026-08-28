"""
Human-readable explanations: grouped SHAP (collapsing 30 text PCA
components + assignee features into named groups) and templated
natural-language bullets generated from SHAP + raw feature values,
reusing reference_stats' dwell/duration lookups rather than recomputing.
"""

import numpy as np
import pandas as pd

import demo_data as dd

FEATURE_GROUPS = {
    "elapsed_minutes": "Time progress & stall",
    "log1p_stall": "Time progress & stall",
    "assignee_prior_delay_rate": "Assignee history & workload",
    "assignee_prior_resolved_count": "Assignee history & workload",
    "assignee_concurrent_open": "Assignee history & workload",
    "is_unassigned": "Assignee history & workload",
    "n_links": "Dependencies",
    "n_blocking_links": "Dependencies",
    "Project_ID": "Issue type & context",
    "Type_Normalized": "Issue type & context",
    "Priority_Normalized": "Issue type & context",
    "Story_Point": "Issue type & context",
    "pattern_label": "Trajectory pattern",
}
for _i in range(30):
    FEATURE_GROUPS[f"text_pc_{_i}"] = "Issue description content"

MATERIALITY_THRESHOLD = 0.03  # min |grouped SHAP| to generate a bullet


def grouped_shap(contrib):
    """contrib: pd.Series feature -> SHAP value. Returns grouped (summed,
    sign-preserving) pd.Series sorted by |value| descending."""
    groups = contrib.index.map(lambda f: FEATURE_GROUPS.get(f, f))
    grouped = contrib.groupby(groups).sum()
    return grouped.reindex(grouped.abs().sort_values(ascending=False).index)


def generate_explanation(row, contrib, ref_stats, project_avg_delay_rate, current_status=None,
                          last_status_change_time=None, checkpoint_time=None):
    """row: a portfolio/issue row (raw feature values). contrib: per-feature
    SHAP series (not grouped). Returns a list of bullet strings, generated
    only for groups whose contribution is material, ranked by |contribution|.
    """
    grouped = grouped_shap(contrib)
    bullets = []

    for group, value in grouped.items():
        if abs(value) < MATERIALITY_THRESHOLD or len(bullets) >= 4:
            continue

        if group == "Time progress & stall" and current_status and last_status_change_time is not None \
                and checkpoint_time is not None:
            days_in_status = (pd.Timestamp(checkpoint_time) - pd.Timestamp(last_status_change_time)).total_seconds() / 86400
            baseline_min = ref_stats.dwell_baseline_minutes(current_status, row["Project_ID"], row["Type_Normalized"])
            if baseline_min and baseline_min > 0 and not np.isnan(baseline_min):
                baseline_days = baseline_min / 1440
                ratio = days_in_status / baseline_days if baseline_days > 0 else np.nan
                if not np.isnan(ratio):
                    bullets.append(
                        f"It has been in *{current_status}* for **{days_in_status:.1f} days** -- "
                        f"about **{ratio:.1f}x** {'longer' if ratio > 1 else 'shorter'} than the typical "
                        f"{baseline_days:.1f} days for this status in this project."
                    )
                    continue

        if group == "Assignee history & workload":
            delay_rate = row.get("assignee_prior_delay_rate")
            concurrent = row.get("assignee_concurrent_open")
            if pd.notna(delay_rate) and pd.notna(concurrent):
                bullets.append(
                    f"Its assignee currently holds **{int(concurrent)} open issues** and has finished "
                    f"late on **{delay_rate:.0%}** of their past work (project average: "
                    f"{project_avg_delay_rate:.0%})."
                )
                continue
            elif row.get("is_unassigned"):
                bullets.append("This issue is **unassigned**, which historically correlates with higher delay risk.")
                continue

        if group == "Dependencies":
            n_blocking = row.get("n_blocking_links", 0)
            n_links = row.get("n_links", 0)
            n = n_blocking if n_blocking else n_links
            label = "blocking dependencies" if n_blocking else "linked issues"
            pct = ref_stats.n_links_percentile(row["Project_ID"], n)
            if pct is not None:
                if value > 0:
                    bullets.append(
                        f"It has **{int(n)} {label}**, more than **{pct:.0%}** of issues in this project -- "
                        f"this raises its predicted risk."
                    )
                else:
                    bullets.append(
                        f"It has only **{int(n)} {label}**, fewer than **{1-pct:.0%}** of issues in this "
                        f"project -- this lowers its predicted risk."
                    )
                continue

        if group == "Issue type & context":
            expected_min = ref_stats.expected_duration_minutes(row)
            elapsed_min = row.get("elapsed_minutes")
            if expected_min and not np.isnan(expected_min) and pd.notna(elapsed_min):
                expected_days = expected_min / 1440
                elapsed_days = elapsed_min / 1440
                bullets.append(
                    f"Issues of this type and priority in this project typically resolve in "
                    f"**{expected_days:.1f} days**; this one is at **day {elapsed_days:.1f}**."
                )
                continue

        if group == "Issue description content":
            direction = "more similar to issues that were delayed" if value > 0 else \
                "more similar to issues that finished on time"
            bullets.append(f"The wording of its title and description is {direction} historically in this project.")
            continue

        if group == "Trajectory pattern":
            direction = "a historically higher-delay" if value > 0 else "a historically lower-delay"
            bullets.append(f"Its stall trajectory so far most resembles {direction} pattern from similar issues.")
            continue

    return bullets


def project_avg_delay_rate(feat_full, project_id):
    sub = feat_full[feat_full["Project_ID"] == project_id]
    return sub["assignee_prior_delay_rate"].mean()


def cohort_trajectories(all_checkpoints, feat_full, project_id, type_norm, dynamic_model, max_n=300, seed=0):
    """Median + IQR of dynamic risk per checkpoint index, for a sample of
    issues in the same (project, type) cohort. Used for the trajectory
    chart's cohort band."""
    cohort_ids = feat_full[
        (feat_full["Project_ID"] == project_id) & (feat_full["Type_Normalized"] == type_norm)
    ]["ID"]
    if len(cohort_ids) > max_n:
        cohort_ids = cohort_ids.sample(n=max_n, random_state=seed)

    ck = all_checkpoints[all_checkpoints["Issue_ID"].isin(cohort_ids)].drop(
        columns=["Project_ID", "Type_Normalized"], errors="ignore"
    ).copy()
    static_lookup = feat_full.set_index("ID")[dd.STATIC_FEATURE_COLS_V2]
    ck = ck.join(static_lookup, on="Issue_ID")
    for col in dd.CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    for col in dd.STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"]:
        if col not in dd.CATEGORICAL_COLS:
            ck[col] = ck[col].astype(float)
    ck["risk"] = dynamic_model.predict_proba(ck[dd.DYN_FEATURE_COLS])[:, 1]

    summary = ck.groupby("checkpoint_index")["risk"].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="size"
    ).reset_index()
    return summary


def peer_resolution_times(feat_full, project_id, type_norm, priority_norm):
    peers = feat_full[
        (feat_full["Project_ID"] == project_id) & (feat_full["Type_Normalized"] == type_norm)
        & (feat_full["Priority_Normalized"] == priority_norm)
    ]
    return peers["Resolution_Time_Minutes"].dropna() / 1440  # days
