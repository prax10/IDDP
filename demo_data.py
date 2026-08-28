"""
Shared data/model loading for the Phase 11/12 demo. No retraining, no new
feature engineering -- this only assembles already-computed artifacts
into a "current portfolio" view.

Demo simplification (stated in the app's "How it works" panel too): the
project has no live feed of genuinely in-progress issues with computed
features, since Phase 6/8 only ever built features for the 203,290
already-resolved issues. The demo instead uses the held-out TEST-split
resolved issues, at each one's *last available checkpoint*, as a stand-in
"current" portfolio -- their true outcome is known (for our own
validation) but the app only ever surfaces the model's prediction.
"""

import numpy as np
import pandas as pd
import joblib

FEATURES_DIR = "data/features"
RAW_DIR = "data/raw"
MODELS_DIR = "models"

STATIC_FEATURE_COLS = [
    "Project_ID", "Priority_Normalized", "Type_Normalized", "Story_Point",
    "assignee_prior_resolved_count", "assignee_prior_delay_rate",
    "assignee_concurrent_open", "is_unassigned", "n_links", "n_blocking_links",
]
TEXT_COLS = [f"text_pc_{i}" for i in range(30)]
STATIC_FEATURE_COLS_V2 = STATIC_FEATURE_COLS + TEXT_COLS
CATEGORICAL_COLS = ["Project_ID", "Priority_Normalized", "Type_Normalized"]
ASSIGNEE_FEATURE_COLS = [
    "assignee_prior_delay_rate", "assignee_prior_resolved_count", "assignee_concurrent_open",
]
DYN_FEATURE_COLS = STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes", "pattern_label"]

STATIC_MODEL_PATH = f"{MODELS_DIR}/xgb_static_v3_tuned.joblib"
DYNAMIC_MODEL_PATH = f"{MODELS_DIR}/xgb_dynamic_pooled_v3_tuned.joblib"

LABEL_COL = "delayed_v2"
DEFAULT_THRESHOLD = 0.5
F1_THRESHOLD = 0.405
CROSSOVER_CHECKPOINT = 3


def load_models():
    static_model = joblib.load(STATIC_MODEL_PATH)
    dynamic_model = joblib.load(DYNAMIC_MODEL_PATH)
    return static_model, dynamic_model


def _apply_categorical_dtypes(df):
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def load_feature_table():
    """v3 feature table: v2 (with-text) features + both label definitions."""
    feat = pd.read_csv(f"{FEATURES_DIR}/phase6_feature_table_v3.csv")
    return feat


def load_raw_display_fields():
    return pd.read_csv(
        f"{RAW_DIR}/issue.csv",
        usecols=["ID", "Title", "Priority", "Type", "Status", "Assignee_ID",
                 "Project_ID", "Creation_Date"],
        parse_dates=["Creation_Date"],
    )


def build_portfolio():
    """Assemble the demo's 'current portfolio': test-split resolved issues,
    at their last available checkpoint, with static + dynamic features and
    display fields attached. This is the single source of truth every tab
    and reassignment.py reads from."""
    feat = load_feature_table()
    split = pd.read_csv(f"{FEATURES_DIR}/phase8_split.csv")
    raw = load_raw_display_fields()

    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])

    test_ids = set(split.loc[split["split"] == "test", "ID"])
    ck_test = ck[ck["Issue_ID"].isin(test_ids)]
    last_ck = ck_test.sort_values("checkpoint_index").groupby("Issue_ID", as_index=False).tail(1)
    last_ck = last_ck.drop(columns=["Project_ID", "Type_Normalized"])

    portfolio = feat.merge(last_ck, left_on="ID", right_on="Issue_ID", how="inner")
    portfolio = portfolio.merge(
        raw[["ID", "Title", "Priority", "Type", "Status", "Assignee_ID", "Creation_Date"]],
        on="ID", how="left",
    )
    portfolio = _apply_categorical_dtypes(portfolio)
    portfolio["pattern_label"] = portfolio["pattern_label"].astype("category")
    return portfolio


def score_portfolio(portfolio, static_model, dynamic_model):
    portfolio = portfolio.copy()
    portfolio["static_risk"] = static_model.predict_proba(portfolio[STATIC_FEATURE_COLS_V2])[:, 1]
    portfolio["dynamic_risk"] = dynamic_model.predict_proba(portfolio[DYN_FEATURE_COLS])[:, 1]
    return portfolio


def load_all_checkpoints():
    """Full checkpoint table (all issues, all checkpoints they reached),
    with pattern_label and log1p_stall attached. Used to build per-issue
    risk trajectories -- filter to one Issue_ID at call time."""
    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])
    return ck


def build_issue_trajectory(issue_id, static_row, all_checkpoints, dynamic_model):
    """Score the dynamic model at every checkpoint this issue actually
    reached, holding its static features fixed (they don't vary by
    checkpoint) and using each checkpoint's own log1p_stall/elapsed_minutes
    /pattern_label."""
    issue_ck = all_checkpoints[all_checkpoints["Issue_ID"] == issue_id].sort_values("checkpoint_index")
    if issue_ck.empty:
        return pd.DataFrame(columns=["checkpoint_index", "risk"])

    rows = issue_ck[["checkpoint_index", "log1p_stall", "elapsed_minutes", "pattern_label"]].copy()
    for col in STATIC_FEATURE_COLS_V2:
        rows[col] = static_row[col]

    for col in CATEGORICAL_COLS:
        rows[col] = rows[col].astype("category")
    rows["pattern_label"] = rows["pattern_label"].astype("category")
    for col in STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"]:
        if col not in CATEGORICAL_COLS:
            rows[col] = rows[col].astype(float)

    rows["risk"] = dynamic_model.predict_proba(rows[DYN_FEATURE_COLS])[:, 1]
    return rows[["checkpoint_index", "risk"]].reset_index(drop=True)


MISSION_CONTROL_PROJECT = 43
MISSION_CONTROL_START = pd.Timestamp("2020-01-01")
MISSION_CONTROL_END = pd.Timestamp("2020-04-30")


def load_project_for_replay(project_id, static_model, dynamic_model):
    """ALL resolved issues in this project (any split -- Mission Control
    needs the true set of what was open on a given day, which requires
    issues regardless of which split they landed in), with static risk
    scored and their full checkpoint trajectory (dynamic risk at each
    checkpoint they reached, plus each checkpoint's absolute timestamp).
    """
    feat = load_feature_table()
    raw = load_raw_display_fields()
    proj_ids = raw.loc[raw["Project_ID"] == project_id, "ID"]

    issues = feat[feat["ID"].isin(proj_ids)].merge(
        raw[["ID", "Title", "Priority", "Type", "Assignee_ID", "Creation_Date"]], on="ID", how="left"
    )
    resolution = pd.read_csv(f"{RAW_DIR}/issue.csv", usecols=["ID", "Resolution_Date"], parse_dates=["Resolution_Date"])
    issues = issues.merge(resolution, on="ID", how="left")
    issues = _apply_categorical_dtypes(issues)
    issues["static_risk"] = static_model.predict_proba(issues[STATIC_FEATURE_COLS_V2])[:, 1]

    ck = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints.csv")
    pattern = pd.read_csv(f"{FEATURES_DIR}/phase8_checkpoints_with_pattern.csv")
    ck = ck.merge(pattern, on=["Issue_ID", "checkpoint_index"], how="left")
    ck = ck[ck["Issue_ID"].isin(issues["ID"])].drop(columns=["Project_ID", "Type_Normalized"]).copy()
    ck["log1p_stall"] = np.log1p(ck["stall_ratio"])

    creation_lookup = issues.set_index("ID")["Creation_Date"]
    ck["checkpoint_time"] = ck["Issue_ID"].map(creation_lookup) + pd.to_timedelta(ck["elapsed_minutes"], unit="m")

    static_cols_lookup = issues.set_index("ID")[STATIC_FEATURE_COLS_V2]
    ck = ck.join(static_cols_lookup, on="Issue_ID")
    for col in CATEGORICAL_COLS:
        ck[col] = ck[col].astype("category")
    ck["pattern_label"] = ck["pattern_label"].astype("category")
    for col in STATIC_FEATURE_COLS_V2 + ["log1p_stall", "elapsed_minutes"]:
        if col not in CATEGORICAL_COLS:
            ck[col] = ck[col].astype(float)
    ck["dynamic_risk"] = dynamic_model.predict_proba(ck[DYN_FEATURE_COLS])[:, 1]

    ck = ck.sort_values("checkpoint_time")
    checkpoints = ck[["Issue_ID", "checkpoint_index", "checkpoint_time", "dynamic_risk"]].reset_index(drop=True)
    return issues, checkpoints


def risk_at_time(issues, checkpoints, T):
    """For every issue, its predicted risk as of T: the dynamic model's
    prediction at the most recent checkpoint with checkpoint_time <= T, or
    the static (creation-time) prediction if no checkpoint has been
    reached yet or the latest one reached is still M1/M2."""
    T = pd.Timestamp(T)
    query = issues[["ID"]].rename(columns={"ID": "Issue_ID"}).copy()
    query["checkpoint_time"] = T
    query["checkpoint_time"] = query["checkpoint_time"].astype(checkpoints["checkpoint_time"].dtype)
    query = query.sort_values("checkpoint_time")

    matched = pd.merge_asof(
        query, checkpoints, on="checkpoint_time", by="Issue_ID", direction="backward",
    )
    matched = matched.set_index("Issue_ID")

    out = issues[["ID", "static_risk"]].set_index("ID").copy()
    out["checkpoint_index"] = matched["checkpoint_index"].reindex(out.index)
    out["dynamic_risk"] = matched["dynamic_risk"].reindex(out.index)

    use_dynamic = out["checkpoint_index"].fillna(0) >= CROSSOVER_CHECKPOINT
    out["risk"] = np.where(use_dynamic, out["dynamic_risk"], out["static_risk"])
    out["source"] = np.where(use_dynamic, "dynamic", "static")
    return out.reset_index()


def open_at_time(issues, T):
    T = pd.Timestamp(T)
    return (issues["Creation_Date"] <= T) & (issues["Resolution_Date"] > T)


def risk_band(risk):
    if risk < 0.4:
        return "low"
    elif risk < 0.7:
        return "medium"
    return "high"


RISK_COLORS = {"low": "#1baf7a", "medium": "#eda100", "high": "#e34948"}


def risk_color(risk):
    return RISK_COLORS[risk_band(risk)]
