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

st.set_page_config(page_title="TAWOS Delay Risk", layout="wide", page_icon="🚦")

RISK_THRESHOLD = dd.DEFAULT_THRESHOLD  # 0.5, stated in the UI
CROSSOVER_CHECKPOINT = dd.CROSSOVER_CHECKPOINT  # M3
GREEN, AMBER, RED = dd.RISK_COLORS["low"], dd.RISK_COLORS["medium"], dd.RISK_COLORS["high"]
STATIC_TEST_AUC = 0.743
DYNAMIC_M10_AUC = 0.859


def risk_badge(risk):
    band = dd.risk_band(risk)
    color = dd.RISK_COLORS[band]
    label = {"low": "On track", "medium": "Watch", "high": "At risk"}[band]
    return f'<span style="color:{color}; font-weight:600;">{label} ({risk:.0%})</span>'


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


@st.cache_data
def get_all_checkpoints():
    return dd.load_all_checkpoints()


static_model, dynamic_model = get_models()
portfolio_raw, feat_full = get_portfolio_data()
portfolio = get_scored_portfolio(static_model, dynamic_model, portfolio_raw)
portfolio["at_risk"] = portfolio["dynamic_risk"] >= RISK_THRESHOLD
all_checkpoints = get_all_checkpoints()

st.title("TAWOS Issue Delay Risk")
st.caption(
    f"XGBoost · trained on 203,290 resolved issues across 39 projects · "
    f"AUC {STATIC_TEST_AUC:.3f} static / {DYNAMIC_M10_AUC:.3f} dynamic at M10"
)
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
  the better bet. The risk-trajectory chart in Tab 2 renders this directly.
- **Decision threshold:** this demo uses **{RISK_THRESHOLD}** for at-risk/on-track calls (matches
  standard accuracy-style decisions); a threshold of {dd.F1_THRESHOLD} was separately tuned to
  optimize F1 if that's the operating point you care about.
- **Risk colour coding (used everywhere in this app):**
  <span style="color:{GREEN};font-weight:600;">green &lt; 0.4</span>,
  <span style="color:{AMBER};font-weight:600;">amber 0.4-0.7</span>,
  <span style="color:{RED};font-weight:600;">red &gt; 0.7</span>.
- **Demo data note:** there is no live feed of in-progress issues with computed features in this
  project, so the app uses held-out (test-split) *resolved* issues, frozen at their last observed
  checkpoint, as a stand-in "current" portfolio. Predictions shown are the model's -- true outcomes
  are known only for our own validation, never surfaced except in the trajectory chart's final
  annotation.
""", unsafe_allow_html=True)

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
    project_avg_risk = proj_df["dynamic_risk"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("At-risk issues", n_at_risk, f"{n_at_risk/len(proj_df):.0%} of portfolio")
    col2.metric("On-track issues", n_on_track, f"{n_on_track/len(proj_df):.0%} of portfolio")
    col3.metric("Avg predicted risk", f"{project_avg_risk:.0%}")

    st.subheader("Predicted risk distribution")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axvspan(0, 0.4, color=GREEN, alpha=0.08)
    ax.axvspan(0.4, 0.7, color=AMBER, alpha=0.08)
    ax.axvspan(0.7, 1.0, color=RED, alpha=0.08)
    ax.hist(proj_df["dynamic_risk"], bins=25, range=(0, 1), color="#4C78A8", edgecolor="white")
    ax.axvline(RISK_THRESHOLD, color="black", linestyle="--", linewidth=1, label=f"threshold ({RISK_THRESHOLD})")
    ax.set_xlabel("Predicted risk", fontsize=11)
    ax.set_ylabel("Issue count", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Top-risk issues")
    top_n = st.slider("Show top N highest-risk issues", 5, 50, 15, key="tab1_topn")
    top_risk = proj_df.sort_values("dynamic_risk", ascending=False).head(top_n)
    display_df = top_risk[["ID", "Title", "Type", "Priority", "Assignee_ID", "dynamic_risk", "checkpoint_index"]].rename(
        columns={"dynamic_risk": "predicted_risk"}
    ).reset_index(drop=True)

    def _style_risk(v):
        return f"background-color: {dd.risk_color(v)}33; color: {dd.risk_color(v)}; font-weight: 600;"

    styled = display_df.style.map(_style_risk, subset=["predicted_risk"]).format({"predicted_risk": "{:.0%}"})
    st.dataframe(styled, width="stretch")

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
            row["ID"]: f"{row['ID']} - {row['Title'][:60]} (risk={row['dynamic_risk']:.0%})"
            for _, row in matches.iterrows()
        }
        selected_id = st.selectbox(
            "Select issue", options, format_func=lambda i: labels[i], key="tab2_select"
        )

        row = portfolio[portfolio["ID"] == selected_id].iloc[0]
        proj_avg = portfolio.loc[portfolio["Project_ID"] == row["Project_ID"], "dynamic_risk"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted risk (dynamic)", f"{row['dynamic_risk']:.0%}",
                    f"{row['dynamic_risk'] - proj_avg:+.0%} vs. project avg")
        col2.metric("Predicted risk (static)", f"{row['static_risk']:.0%}")
        col3.metric("Checkpoint reached", f"M{int(row['checkpoint_index'])}")
        st.markdown(risk_badge(row["dynamic_risk"]), unsafe_allow_html=True)

        st.write(f"**Type:** {row['Type']}  |  **Priority:** {row['Priority']}  |  "
                 f"**Assignee:** {row['Assignee_ID']}  |  **Project:** {row['Project_ID']}")

        # --- Risk trajectory chart (the headline visual) ---
        st.subheader("Risk trajectory across checkpoints")
        traj = dd.build_issue_trajectory(selected_id, row, all_checkpoints, dynamic_model)

        if traj.empty:
            st.info("No checkpoint history available for this issue (resolved before M1).")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.axhspan(0, 0.4, color=GREEN, alpha=0.06)
            ax.axhspan(0.4, 0.7, color=AMBER, alpha=0.06)
            ax.axhspan(0.7, 1.0, color=RED, alpha=0.06)

            max_ck = traj["checkpoint_index"].max()
            if max_ck >= 1:
                ax.axvspan(1, min(2, max_ck), color="grey", alpha=0.15)
                ax.text(1.5, 0.03, "insufficient history --\nstatic preferred", fontsize=11,
                        ha="center", va="bottom", color="#555555")

            ax.plot(traj["checkpoint_index"], traj["risk"], color="#eb6834", linewidth=2.4,
                    marker="o", markersize=7, label="Dynamic predicted risk", zorder=5)
            ax.axhline(row["static_risk"], color="#2a78d6", linestyle="--", linewidth=2,
                       label=f"Static prediction (creation-time): {row['static_risk']:.0%}")
            ax.axhline(RISK_THRESHOLD, color="black", linestyle=":", linewidth=1.3,
                       label=f"Decision threshold ({RISK_THRESHOLD})")
            ax.axvline(CROSSOVER_CHECKPOINT, color="#54A24B", linestyle="-", linewidth=1.6, alpha=0.8)
            ax.annotate(
                "dynamic becomes more\nreliable than static from here",
                xy=(CROSSOVER_CHECKPOINT, 0.95), fontsize=11, color="#2c6b3f",
                ha="left", va="top",
            )

            actual_delayed = int(row["delayed_v2"])
            outcome_text = "DELAYED" if actual_delayed else "ON TIME"
            outcome_color = RED if actual_delayed else GREEN
            ax.scatter([max_ck], [traj.iloc[-1]["risk"]], s=140, facecolors="none",
                       edgecolors=outcome_color, linewidths=2.5, zorder=6)
            ax.annotate(
                f"Actual outcome: {outcome_text}",
                xy=(max_ck, traj.iloc[-1]["risk"]), xytext=(-10, 20 if not actual_delayed else -30),
                textcoords="offset points", fontsize=11, fontweight="bold", color=outcome_color,
                ha="right",
            )

            ax.set_xlabel("Checkpoint index (M1-M10)", fontsize=12)
            ax.set_ylabel("Predicted delay risk", fontsize=12)
            ax.set_xticks(range(1, 11))
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=10, loc="lower right")
            ax.set_title(f"Issue {selected_id}: risk trajectory", fontsize=13)
            st.pyplot(fig, clear_figure=True)

            first_risk = traj.iloc[0]["risk"]
            first_ck = int(traj.iloc[0]["checkpoint_index"])
            last_risk = traj.iloc[-1]["risk"]
            last_ck = int(traj.iloc[-1]["checkpoint_index"])
            outcome_phrase = "It was ultimately delayed." if actual_delayed else "It ultimately finished on time."
            st.markdown(
                f"*At creation this issue was predicted **{row['static_risk']:.0%}** likely to be delayed "
                f"(static). By checkpoint **M{last_ck}** the dynamic model's estimate had moved to "
                f"**{last_risk:.0%}**. {outcome_phrase}*"
            )

        st.subheader("Why the model predicts this (at its latest checkpoint)")
        tree_explainer = ra._get_explainer(dynamic_model)
        row_features = row[dd.DYN_FEATURE_COLS].to_frame().T
        for col in dd.DYN_FEATURE_COLS:
            if col in dd.CATEGORICAL_COLS + ["pattern_label"]:
                row_features[col] = row_features[col].astype(portfolio[col].dtype)
            else:
                row_features[col] = row_features[col].astype(float)
        shap_values = tree_explainer(row_features[dd.DYN_FEATURE_COLS])

        fig2 = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2, clear_figure=True)

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
    workload_df = latest_per_dev[["Assignee_ID", "assignee_concurrent_open", "assignee_prior_delay_rate",
                                   "assignee_prior_resolved_count"]].rename(columns={
        "assignee_concurrent_open": "concurrent_open_issues",
        "assignee_prior_delay_rate": "historical_delay_rate",
        "assignee_prior_resolved_count": "prior_resolved_count",
    }).reset_index(drop=True)

    styled_workload = workload_df.style.map(
        lambda v: f"background-color: {dd.risk_color(v)}33; color: {dd.risk_color(v)}; font-weight: 600;",
        subset=["historical_delay_rate"],
    ).format({"historical_delay_rate": "{:.0%}"})
    st.dataframe(styled_workload, width="stretch")

# ---------------------------------------------------------------------
# Tab 4: Reassignment Suggester
# ---------------------------------------------------------------------
with tab4:
    st.header("Reassignment Suggester")
    at_risk_df = portfolio[portfolio["at_risk"]].sort_values("dynamic_risk", ascending=False)
    st.caption(f"{len(at_risk_df)} at-risk issues in the portfolio (risk >= {RISK_THRESHOLD}).")

    options4 = at_risk_df["ID"].head(200).tolist()
    labels4 = {
        row["ID"]: f"{row['ID']} - {row['Title'][:60]} (risk={row['dynamic_risk']:.0%})"
        for _, row in at_risk_df.head(200).iterrows()
    }
    selected_id_4 = st.selectbox(
        "Select an at-risk issue", options4, format_func=lambda i: labels4[i], key="tab4_select"
    )

    result = ra.suggest_reassignment(selected_id_4, portfolio, feat_full, dynamic_model, top_n=5)
    current_assignee = result["current_assignee_id"]

    st.metric("Current predicted risk", f"{result['current_risk']:.0%}")
    st.markdown(risk_badge(result["current_risk"]), unsafe_allow_html=True)
    st.write(f"**Top risk drivers:** " + ", ".join(f"{f} ({v:+.3f})" for f, v in result["top_drivers"]))

    if result["verdict"] == "reassignment_may_help":
        st.success(f"### ✅ Reassignment may help\n{result['explanation']}")
    else:
        st.warning(f"### ⚠️ Reassignment unlikely to help\n{result['explanation']}")

    if result["candidates"]:
        st.subheader("Candidate comparison")
        cand_df = pd.DataFrame(result["candidates"])
        bars = pd.concat([
            pd.DataFrame([{"assignee_id": f"{int(current_assignee)} (current)",
                            "predicted_risk": result["current_risk"], "is_current": True}]),
            cand_df.assign(
                assignee_id=cand_df["assignee_id"].apply(lambda a: str(int(a))), is_current=False
            )[["assignee_id", "predicted_risk", "is_current"]],
        ]).sort_values("predicted_risk", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(bars))))
        colors = ["#0b0b0b" if is_cur else dd.risk_color(r)
                  for r, is_cur in zip(bars["predicted_risk"], bars["is_current"])]
        ax.barh(bars["assignee_id"], bars["predicted_risk"], color=colors)
        ax.axvline(RISK_THRESHOLD, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted risk", fontsize=11)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Black bar = current assignee. Bar colour follows the same risk bands as elsewhere in the app.")

        with st.expander("Show as table"):
            cand_df_display = cand_df.copy()
            cand_df_display["predicted_risk"] = cand_df_display["predicted_risk"].round(3)
            cand_df_display["risk_delta"] = cand_df_display["risk_delta"].round(3)
            st.dataframe(cand_df_display, width="stretch")

    if result.get("candidate_pool_threshold"):
        st.caption(f"Candidate pool: developers with >= {result['candidate_pool_threshold']} "
                   f"prior resolved issues in this project.")
