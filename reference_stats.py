"""
Static (non-expanding) reference lookups reused from Phase 8's leakage-safe
design, for human-readable explanations only -- NOT used as model inputs,
so there's no leakage concern in using the full population here (unlike
training, where these had to be computed expanding/train-only).

- expected duration by (Project, Type, Priority) hierarchy
- dwell-time baseline by (Status, Project, Type) hierarchy
- peer cohort resolution-time distributions by (Project, Type, Priority)
"""

import numpy as np
import pandas as pd

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"

MIN_GROUP = 5


def _hierarchical_median_lookup(df, group_levels, value_col):
    """group_levels: list of tuples (cols,) most specific first. Returns a
    function key_tuple_per_level -> median, falling back down levels."""
    medians = []
    for cols in group_levels:
        if cols:
            m = df.groupby(list(cols), observed=True)[value_col].median()
            n = df.groupby(list(cols), observed=True)[value_col].size()
        else:
            m = pd.Series({(): df[value_col].median()})
            n = pd.Series({(): len(df)})
        medians.append((cols, m, n))
    return medians


class ReferenceStats:
    def __init__(self, feat_full, dwell_df):
        self.feat_full = feat_full
        self.duration_medians = _hierarchical_median_lookup(
            feat_full,
            [("Project_ID", "Type_Normalized", "Priority_Normalized"),
             ("Project_ID", "Type_Normalized"),
             ("Project_ID", "Priority_Normalized"),
             ("Project_ID",), ()],
            "Resolution_Time_Minutes",
        )
        self.dwell_medians = _hierarchical_median_lookup(
            dwell_df,
            [("Status", "Project_ID", "Type_Normalized"),
             ("Status", "Project_ID"),
             ("Status", "Type_Normalized"),
             ("Status",)],
            "duration_minutes",
        )
        # dependency-count percentile lookup, per project
        self.n_links_by_project = {
            p: g["n_links"].to_numpy() for p, g in feat_full.groupby("Project_ID", observed=True)
        }

    def expected_duration_minutes(self, row):
        return _query_hierarchical_safe(self.duration_medians, row,
                                         ["Project_ID", "Type_Normalized", "Priority_Normalized"])

    def dwell_baseline_minutes(self, status, project_id, type_norm):
        row = {"Status": status, "Project_ID": project_id, "Type_Normalized": type_norm}
        return _query_hierarchical_safe(self.dwell_medians, row, ["Status", "Project_ID", "Type_Normalized"])

    def n_links_percentile(self, project_id, n_links_value):
        arr = self.n_links_by_project.get(project_id)
        if arr is None or len(arr) == 0:
            return None
        return float((arr < n_links_value).mean())


def _query_hierarchical_safe(medians, row, all_cols):
    for cols, m, n in medians:
        if not cols:
            key = ()
        elif len(cols) == 1:
            key = row[cols[0]]
        else:
            key = tuple(row[c] for c in cols)
        if key in m.index and n.get(key, 0) >= MIN_GROUP:
            return m[key]
    cols, m, n = medians[-1]
    key = () if not cols else (row[cols[0]] if len(cols) == 1 else tuple(row[c] for c in cols))
    return m.get(key, np.nan)


def build_dwell_corpus(feat_full):
    """Rebuild the completed-dwell-period corpus (status segments), same
    genuine-status-transition filtering as Phase 8 foundation, but as a
    single static table (no expanding structure needed for display use)."""
    cl = pd.read_csv(f"{RAW_DIR}/change_log_status.csv", parse_dates=["Creation_Date"])
    genuine = cl[cl["From_String"].notna() & cl["To_String"].notna()].copy()
    genuine = genuine[genuine["Issue_ID"].isin(feat_full["ID"])]
    genuine = genuine.sort_values(["Issue_ID", "Creation_Date"], kind="mergesort")

    issue_meta = feat_full.set_index("ID")

    dwell_status, dwell_project, dwell_type, dwell_minutes = [], [], [], []
    for issue_id, g in genuine.groupby("Issue_ID", sort=False):
        if issue_id not in issue_meta.index:
            continue
        meta = issue_meta.loc[issue_id]
        creation = meta["Creation_Date"]
        resolution = meta["Resolution_Date"]
        project_id = meta["Project_ID"]
        type_norm = meta["Type_Normalized"]

        times = g["Creation_Date"].values
        froms = g["From_String"].values
        tos = g["To_String"].values

        seg_starts = [creation] + list(times)
        seg_ends = list(times) + [resolution]
        seg_statuses = [froms[0]] + list(tos)

        for s, e, status in zip(seg_starts, seg_ends, seg_statuses):
            dwell_status.append(status)
            dwell_project.append(project_id)
            dwell_type.append(type_norm)
            dwell_minutes.append((pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 60.0)

    dwell_df = pd.DataFrame({
        "Status": dwell_status, "Project_ID": dwell_project, "Type_Normalized": dwell_type,
        "duration_minutes": dwell_minutes,
    })
    return dwell_df[dwell_df["duration_minutes"] >= 0]


def load_reference_stats():
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    issue = pd.read_csv(
        f"{RAW_DIR}/issue.csv",
        usecols=["ID", "Creation_Date", "Resolution_Date"],
        parse_dates=["Creation_Date", "Resolution_Date"],
    )
    feat = feat.merge(issue, on="ID", how="left")
    feat["Type_Normalized"] = feat["Type_Normalized"].astype("category")
    feat["Priority_Normalized"] = feat["Priority_Normalized"].astype("category")

    dwell_df = build_dwell_corpus(feat)
    return ReferenceStats(feat, dwell_df)
