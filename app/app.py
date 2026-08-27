"""
Phase 12: Streamlit prototype. Demo wrapper around already-final models --
no retraining, no live Jira sync, no task editing, no notifications.

Run with: streamlit run app/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

import demo_data as dd
import reassignment as ra

st.set_page_config(page_title="TAWOS Delay Risk", layout="wide")

RISK_THRESHOLD = dd.DEFAULT_THRESHOLD  # 0.5, stated in the UI


@st.cache_resource
def get_models():
    return dd.load_models()


@st.cache_data
def get_portfolio_data():
    portfolio = dd.build_portfolio()
    feat_full = dd.load_feature_table()
    issue_raw = pd.read_csv(
        "data/raw/issue.csv", usecols=["ID", "Assignee_ID", "Creation_Date"],
        parse_dates=["Creation_Date"],
    )
    feat_full = feat_full.merge(issue_raw, on="ID", how="left")
    return portfolio, feat_full


@st.cache_data
def get_scored_portfolio(_static_model, _dynamic_model, portfolio):
    return dd.score_portfolio(portfolio, _static_model, _dynamic_model)


static_model, dynamic_model = get_models()
portfolio_raw, feat_full = get_portfolio_data()
portfolio = get_scored_portfolio(static_model, dynamic_model, portfolio_raw)
portfolio["at_risk"] = portfolio["dynamic_risk"] >= RISK_THRESHOLD

st.title("TAWOS Issue Delay Risk")
st.caption(
    "Prototype demo -- predictions run on held-out test-split issues from the TAWOS dataset, "
    "used as a stand-in for a 'current' portfolio. See 'How it works' below for details."
)

with st.expander("How it works", expanded=False):
    st.markdown(f"""
- **Label definition:** an issue is "delayed" if its resolution time exceeds its own project's
  *median* resolution time -- and that median is computed **only from training-split issues**
  (never from validation or test issues), so the label itself doesn't leak future information.
- **Leakage-safe by design:** every feature (assignee history, workload, dependency counts,
  text embeddings) reflects only information available at or before the issue's own creation time,
  or, for in-progress predictions, at or before the specific checkpoint being scored.
- **Predictions for in-progress issues are checkpoint-based:** the model re-evaluates an issue
  at 10 checkpoints across its expected lifetime (M1 = 10% of expected duration, M10 = 100%+),
  using only the trajectory observed up to that point.
- **The key finding driving this app:** dynamic (checkpoint-based) predictions only become more
  reliable than a creation-time-only prediction after roughly **30% of an issue's expected duration
  has elapsed (checkpoint M3)** -- confirmed via bootstrap confidence intervals, not just a point
  estimate. Before M3, trust the static/creation-time model; from M3 onward, the dynamic model is
  the better bet.
- **Decision threshold:** this demo uses **{RISK_THRESHOLD}** for at-risk/on-track calls (matches
  standard accuracy-style decisions); a threshold of {dd.F1_THRESHOLD} was separately tuned to
  optimize F1 if that's the operating point you care about.
- **Demo data note:** there is no live feed of in-progress issues with computed features in this
  project, so the app uses held-out (test-split) *resolved* issues, frozen at their last observed
  checkpoint, as a stand-in "current" portfolio. Predictions shown are the model's -- true outcomes
  are known only for our own validation, never surfaced in the UI.
""")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Project Risk Dashboard", "Issue Risk & Explanation", "Team Workload", "Reassignment Suggester"]
)

# ---------------------------------------------------------------------
# Tab 1: Project Risk Dashboard
# ---------------------------------------------------------------------
with tab1:
    st.header("Project Risk Dashboard")
    projects = sorted(portfolio["Project_ID"].unique())
    project_id = st.selectbox("Project", projects, key="tab1_project")

    proj_df = portfolio[portfolio["Project_ID"] == project_id]
    n_at_risk = int(proj_df["at_risk"].sum())
    n_on_track = len(proj_df) - n_at_risk

    col1, col2, col3 = st.columns(3)
    col1.metric("At-risk issues", n_at_risk)
    col2.metric("On-track issues", n_on_track)
    col3.metric("Total in portfolio", len(proj_df))

    top_n = st.slider("Show top N highest-risk issues", 5, 50, 15, key="tab1_topn")
    top_risk = proj_df.sort_values("dynamic_risk", ascending=False).head(top_n)
    st.dataframe(
        top_risk[["ID", "Title", "Type", "Priority", "Assignee_ID", "dynamic_risk", "checkpoint_index"]]
        .rename(columns={"dynamic_risk": "predicted_risk"})
        .reset_index(drop=True),
        width="stretch",
    )

# ---------------------------------------------------------------------
# Tab 2: Issue Risk & Explanation
# ---------------------------------------------------------------------
with tab2:
    st.header("Issue Risk & Explanation")
    search = st.text_input("Search by Issue ID or Title keyword", key="tab2_search")

    if search:
        try:
            search_id = int(search)
            matches = portfolio[portfolio["ID"] == search_id]
        except ValueError:
            matches = portfolio[portfolio["Title"].str.contains(search, case=False, na=False)]
    else:
        matches = portfolio.sort_values("dynamic_risk", ascending=False).head(20)

    if matches.empty:
        st.warning("No matching issues found in the current portfolio.")
    else:
        options = matches["ID"].tolist()
        labels = {
            row["ID"]: f"{row['ID']} - {row['Title'][:60]} (risk={row['dynamic_risk']:.3f})"
            for _, row in matches.iterrows()
        }
        selected_id = st.selectbox(
            "Select issue", options, format_func=lambda i: labels[i], key="tab2_select"
        )

        row = portfolio[portfolio["ID"] == selected_id].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted risk (dynamic)", f"{row['dynamic_risk']:.3f}")
        col2.metric("Predicted risk (static)", f"{row['static_risk']:.3f}")
        col3.metric("Checkpoint reached", int(row["checkpoint_index"]))

        st.write(f"**Type:** {row['Type']}  |  **Priority:** {row['Priority']}  |  "
                 f"**Assignee:** {row['Assignee_ID']}  |  **Project:** {row['Project_ID']}")

        tree_explainer = ra._get_explainer(dynamic_model)
        row_features = row[dd.DYN_FEATURE_COLS].to_frame().T
        for col in dd.DYN_FEATURE_COLS:
            if col in dd.CATEGORICAL_COLS + ["pattern_label"]:
                row_features[col] = row_features[col].astype(portfolio[col].dtype)
            else:
                row_features[col] = row_features[col].astype(float)
        shap_values = tree_explainer(row_features[dd.DYN_FEATURE_COLS])

        st.subheader("Why the model predicts this")
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig, clear_figure=True)

        contrib = pd.Series(shap_values.values[0], index=dd.DYN_FEATURE_COLS).sort_values(
            key=abs, ascending=False
        )
        top3 = contrib.head(3)
        direction = lambda v: "increases" if v > 0 else "decreases"
        plain = "; ".join(f"**{f}** {direction(v)} risk" for f, v in top3.items())
        st.markdown(f"In plain language: {plain}.")

# ---------------------------------------------------------------------
# Tab 3: Team Workload
# ---------------------------------------------------------------------
with tab3:
    st.header("Team Workload")
    project_id_3 = st.selectbox("Project", projects, key="tab3_project")

    proj_issues = feat_full[feat_full["Project_ID"] == project_id_3].dropna(subset=["Assignee_ID"])
    proj_issues = proj_issues.sort_values("Creation_Date")
    latest_per_dev = proj_issues.groupby("Assignee_ID", as_index=False).tail(1)
    latest_per_dev = latest_per_dev.sort_values("assignee_concurrent_open", ascending=False)

    st.caption(
        "Each developer's most recently observed concurrent-open-issue count and historical "
        "delay rate, as of their latest issue in this project -- this is exactly the workload "
        "signal the model uses."
    )
    st.dataframe(
        latest_per_dev[["Assignee_ID", "assignee_concurrent_open", "assignee_prior_delay_rate",
                         "assignee_prior_resolved_count"]]
        .rename(columns={
            "assignee_concurrent_open": "concurrent_open_issues",
            "assignee_prior_delay_rate": "historical_delay_rate",
            "assignee_prior_resolved_count": "prior_resolved_count",
        })
        .reset_index(drop=True),
        width="stretch",
    )

# ---------------------------------------------------------------------
# Tab 4: Reassignment Suggester
# ---------------------------------------------------------------------
with tab4:
    st.header("Reassignment Suggester")
    at_risk_df = portfolio[portfolio["at_risk"]].sort_values("dynamic_risk", ascending=False)
    st.caption(f"{len(at_risk_df)} at-risk issues in the portfolio (risk >= {RISK_THRESHOLD}).")

    options4 = at_risk_df["ID"].head(200).tolist()
    labels4 = {
        row["ID"]: f"{row['ID']} - {row['Title'][:60]} (risk={row['dynamic_risk']:.3f})"
        for _, row in at_risk_df.head(200).iterrows()
    }
    selected_id_4 = st.selectbox(
        "Select an at-risk issue", options4, format_func=lambda i: labels4[i], key="tab4_select"
    )

    result = ra.suggest_reassignment(selected_id_4, portfolio, feat_full, dynamic_model, top_n=5)

    st.metric("Current predicted risk", f"{result['current_risk']:.3f}")
    st.write(f"**Top risk drivers:** " + ", ".join(f"{f} ({v:+.3f})" for f, v in result["top_drivers"]))

    if result["verdict"] == "reassignment_may_help":
        st.success(f"✅ Reassignment may help — {result['explanation']}")
        cand_df = pd.DataFrame(result["candidates"])
        cand_df["predicted_risk"] = cand_df["predicted_risk"].round(3)
        cand_df["risk_delta"] = cand_df["risk_delta"].round(3)
        st.dataframe(cand_df, width="stretch")
    else:
        st.warning(f"⚠️ Reassignment unlikely to help — {result['explanation']}")
        if result["candidates"]:
            with st.expander("Show candidates anyway (none clear the improvement threshold)"):
                cand_df = pd.DataFrame(result["candidates"])
                cand_df["predicted_risk"] = cand_df["predicted_risk"].round(3)
                cand_df["risk_delta"] = cand_df["risk_delta"].round(3)
                st.dataframe(cand_df, width="stretch")
    if result.get("candidate_pool_threshold"):
        st.caption(f"Candidate pool: developers with >= {result['candidate_pool_threshold']} "
                   f"prior resolved issues in this project.")
