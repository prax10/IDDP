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
