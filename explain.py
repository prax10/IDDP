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

MATERIALITY_THRESHOLD = 0.03  # min |grouped SHAP| to still count as material once top-3 are covered
MIN_GUARANTEED = 3            # top-N groups always shown, regardless of template availability

_RANK_PHRASES = ["the single largest factor", "the second-largest factor", "the third-largest factor"]


def _rank_phrase(idx):
    if idx < len(_RANK_PHRASES):
        return _RANK_PHRASES[idx]
    return "a smaller factor"


def grouped_shap(contrib):
    """contrib: pd.Series feature -> SHAP value. Returns grouped (summed,
    sign-preserving) pd.Series sorted by |value| descending."""
    groups = contrib.index.map(lambda f: FEATURE_GROUPS.get(f, f))
    grouped = contrib.groupby(groups).sum()
    return grouped.reindex(grouped.abs().sort_values(ascending=False).index)


def _short_reason(group, value):
    """Very short phrase for the one-line summary sentence."""
    phrases = {
        "Time progress & stall": "how long it has already been open",
        "Assignee history & workload": "the assignee's current workload and track record",
        "Dependencies": "its dependency count" if value > 0 else "having few dependencies",
        "Issue type & context": "this type/priority combination's track record in this project",
        "Issue description content": (
            "description content resembling historically delayed issues" if value > 0
            else "description content resembling issues that finished on time"
        ),
        "Trajectory pattern": "its stall trajectory pattern",
    }
    return phrases.get(group, group)


def _bullet_for_group(group, value, rank, row, ref_stats, project_avg_delay_rate,
                       current_status, last_status_change_time, checkpoint_time):
    """Always returns a non-empty, direction-aware bullet string for the
    given group -- no silent skipping, per Fix 2."""
    verb = "raising" if value > 0 else "lowering"
    rank_phrase = _rank_phrase(rank)

    if group == "Time progress & stall":
        elapsed_min = row.get("elapsed_minutes")
        if current_status and last_status_change_time is not None and checkpoint_time is not None:
            days_in_status = (pd.Timestamp(checkpoint_time) - pd.Timestamp(last_status_change_time)).total_seconds() / 86400
            baseline_min = ref_stats.dwell_baseline_minutes(current_status, row["Project_ID"], row["Type_Normalized"])
            if baseline_min and baseline_min > 0 and not np.isnan(baseline_min):
                baseline_days = baseline_min / 1440
                ratio = days_in_status / baseline_days if baseline_days > 0 else np.nan
                if not np.isnan(ratio):
                    return (
                        f"It has been in *{current_status}* for **{days_in_status:.1f} days** -- "
                        f"about **{ratio:.1f}x** the typical {baseline_days:.1f} days for this status in "
                        f"this project -- {rank_phrase}, {verb} this estimate."
                    )
        expected_min = ref_stats.expected_duration_minutes(row)
        if expected_min and not np.isnan(expected_min) and pd.notna(elapsed_min) and expected_min > 0:
            expected_days = expected_min / 1440
            elapsed_days = elapsed_min / 1440
            pct_of_expected = elapsed_days / expected_days
            return (
                f"It has been open for **{elapsed_days:.1f} days**, which is **{pct_of_expected:.0%}** of the "
                f"{expected_days:.1f} days issues like this typically take -- {rank_phrase}, {verb} this estimate."
            )
        if pd.notna(elapsed_min):
            return (
                f"It has been open for **{elapsed_min/1440:.1f} days** -- {rank_phrase}, {verb} this estimate."
            )
        return f"Its time-in-progress pattern is {rank_phrase}, {verb} this estimate."

    if group == "Assignee history & workload":
        delay_rate = row.get("assignee_prior_delay_rate")
        concurrent = row.get("assignee_concurrent_open")
        if pd.notna(delay_rate) and pd.notna(concurrent):
            cmp_word = "above" if delay_rate > project_avg_delay_rate else "below"
            return (
                f"Its assignee currently holds **{int(concurrent)} open issues** and has finished late on "
                f"**{delay_rate:.0%}** of their past work ({cmp_word} the project average of "
                f"{project_avg_delay_rate:.0%}) -- {rank_phrase}, {verb} this estimate."
            )
        if row.get("is_unassigned"):
            return f"This issue is **unassigned** -- {rank_phrase}, {verb} this estimate."
        return f"Assignee history & workload is {rank_phrase}, {verb} this estimate."

    if group == "Dependencies":
        n_blocking = row.get("n_blocking_links", 0) or 0
        n_links = row.get("n_links", 0) or 0
        n = n_blocking if n_blocking else n_links
        label = "blocking dependencies" if n_blocking else "linked issues"
        if n == 0:
            pct_any = ref_stats.n_links_fraction_with_any(row["Project_ID"])
            if pct_any is None:
                return f"Its dependency count is {rank_phrase}, {verb} this estimate."
            return (
                f"It has no {label}; **{pct_any:.0%}** of issues in this project have at least one -- "
                f"{rank_phrase}, {verb} this estimate."
            )
        pct = ref_stats.n_links_percentile(row["Project_ID"], n)
        if pct is None:
            return f"Its dependency count is {rank_phrase}, {verb} this estimate."
        if value > 0:
            return (
                f"It has **{int(n)} {label}** -- more than **{pct:.0%}** of issues in this project -- "
                f"{rank_phrase}, {verb} this estimate."
            )
        return (
            f"It has only **{int(n)} {label}** -- fewer than **{1-pct:.0%}** of issues in this project -- "
            f"{rank_phrase}, {verb} this estimate."
        )

    if group == "Issue type & context":
        tp_rate = ref_stats.type_priority_delay_rate(row["Project_ID"], row["Type_Normalized"], row["Priority_Normalized"])
        if tp_rate is not None:
            return (
                f"Its **{row['Type']}** / **{row['Priority']}**-priority combination in this project has "
                f"historically been delayed **{tp_rate:.0%}** of the time -- {rank_phrase}, {verb} this estimate."
            )
        return (
            f"Its **{row['Type']}** / **{row['Priority']}**-priority combination in this project is "
            f"{rank_phrase}, {verb} this estimate."
        )

    if group == "Issue description content":
        similar_to = "issues that have historically run late" if value > 0 else "issues that finished on time"
        return (
            f"The wording of the title and description resembles {similar_to} -- {rank_phrase} here. "
            f"(This signal is learned from text patterns and is not directly interpretable.)"
        )

    if group == "Trajectory pattern":
        direction = "a historically higher-delay" if value > 0 else "a historically lower-delay"
        return (
            f"Its stall trajectory so far most resembles {direction} pattern from similar issues -- "
            f"{rank_phrase}, {verb} this estimate."
        )

    return f"{group} is {rank_phrase}, {verb} this estimate."


def generate_explanation(row, contrib, ref_stats, project_avg_delay_rate, current_status=None,
                          last_status_change_time=None, checkpoint_time=None, predicted_risk=None):
    """Returns {"summary": str, "raising": [str], "lowering": [str]}.

    Fix 1: raising/lowering are always kept in separate lists.
    Fix 2: the top MIN_GUARANTEED groups by |SHAP| always get a bullet.
    Fix 4: widens the shown set until the displayed net direction matches
    the actual prediction (risk > 0.5 -> shown raising must outweigh shown
    lowering, and vice versa), rather than emitting a self-contradicting
    explanation.
    """
    grouped = grouped_shap(contrib)
    ranked = list(grouped.items())

    def make_bullet(i):
        group, value = ranked[i]
        return group, value, _bullet_for_group(
            group, value, i, row, ref_stats, project_avg_delay_rate,
            current_status, last_status_change_time, checkpoint_time,
        )

    n_shown = max(MIN_GUARANTEED, (np.abs(grouped.to_numpy()) >= MATERIALITY_THRESHOLD).sum())
    n_shown = min(n_shown, len(ranked))

    if predicted_risk is None:
        predicted_risk = 0.5 + grouped.sum()  # fallback: sign of total contribution

    while n_shown <= len(ranked):
        shown = [make_bullet(i) for i in range(n_shown)]
        raising_sum = sum(v for _, v, _ in shown if v > 0)
        lowering_sum = sum(-v for _, v, _ in shown if v < 0)
        net_matches = (predicted_risk >= 0.5 and raising_sum >= lowering_sum) or \
                      (predicted_risk < 0.5 and lowering_sum >= raising_sum)
        if net_matches or n_shown == len(ranked):
            break
        n_shown += 1

    raising = [b for _, v, b in shown if v > 0]
    lowering = [b for _, v, b in shown if v < 0]

    top_raising = next(((g, v) for g, v, _ in shown if v > 0), None)
    top_lowering = next(((g, v) for g, v, _ in shown if v < 0), None)

    verdict_phrase = "likely to be delayed" if predicted_risk >= 0.5 else "not likely to be delayed"
    summary = f"**{predicted_risk:.0%} {verdict_phrase}**"
    if top_raising:
        summary += f" -- driven mainly by {_short_reason(*top_raising)}"
    if top_lowering:
        summary += f", partly offset by {_short_reason(*top_lowering)}"
    summary += "."

    if not net_matches:
        # Every named factor pulls the same direction, opposite the verdict --
        # this happens when the model's baseline rate for issues like this one
        # is already on the other side of 0.5, and the listed factors explain
        # a (partial) deviation from that baseline rather than the baseline
        # itself. State this plainly rather than implying the bullets alone
        # add up to the verdict.
        base_direction = "high" if predicted_risk >= 0.5 else "low"
        summary += (
            f" (Note: issues like this one start from an already-{base_direction} baseline risk; "
            f"the factors below explain how this issue deviates from that baseline, not the "
            f"baseline itself.)"
        )

    return {"summary": summary, "raising": raising, "lowering": lowering}


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
