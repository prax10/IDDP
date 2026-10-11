"""
Delay Radar v5 -- Streamlit app for the improved models (Task J).

Run from the capstone folder:   streamlit run app/app_v5.py
The original app/app.py (v3 models) is untouched.

What is new vs app.py
  * pick ANY project and ANY date: every issue open on that date is scored with
    features computed from events before that date (status moves, comments,
    reassignments, sprint changes, edits, team track record)
  * "why" for every issue (TreeSHAP, grouped, plain English)
  * risk history of an issue over its life
  * Scrum mode: running sprints on that date, each committed issue's spill risk
  * model card with the honest numbers (incl. censoring-safe and undetermined rows)
This is a historical replay on TAWOS data, not a live Jira feed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import streamlit as st

from app_v5_backend import Backend

st.set_page_config(page_title="Delay Radar v5", layout="wide", page_icon="📡")


@st.cache_resource(show_spinner="Loading change history and models (first run takes ~1 minute)...")
def backend():
    return Backend()


b = backend()
projects = b.projects()
projects = projects[projects["n"] >= 200].sort_values("n", ascending=False)

with st.sidebar:
    st.title("📡 Delay Radar v5")
    pid = st.selectbox("Project", projects.index.tolist(),
                       format_func=lambda p: f"Project {p}  ({projects.loc[p, 'n']:,} issues)")
    first = pd.to_datetime(projects.loc[pid, "first"], unit="s").date()
    last = pd.to_datetime(projects.loc[pid, "last"], unit="s").date()
    default = (pd.Timestamp(last) - pd.Timedelta(days=180)).date()
    date = st.date_input("As of date", value=default, min_value=first, max_value=last)
    st.caption("Everything is computed from events before this date. "
               "'Actually late' is shown only to check the model after the fact.")
    show_truth = st.toggle("Show actual outcomes", value=False)
    show_overdue = st.toggle("Include already-overdue issues", value=False,
                             help="Issues already older than the project's typical duration are late by definition; "
                                  "hidden by default so the list focuses on issues that can still be saved.")

tab_port, tab_issue, tab_scrum, tab_card = st.tabs(["Portfolio", "Issue detail", "Scrum mode", "Model card"])

with tab_port:
    df_all = b.open_issues(pid, pd.Timestamp(date))
    df = df_all if (show_overdue or df_all.empty) else df_all[~df_all["overdue"]].reset_index(drop=True)
    if df.empty:
        st.info("No open issues on this date.")
    else:
        late = (df["flag"] == "likely late").sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open issues", f"{len(df_all):,}")
        c2.metric("Already overdue", f"{int(df_all['overdue'].sum()):,}")
        c3.metric("Flagged likely late", f"{late:,}", f"{late / max((~df['overdue']).sum(), 1):.0%} of not-yet-overdue")
        c4.metric("Median age (days)", f"{df['age_days'].median():.0f}")
        cols = ["Issue_ID", "Title", "stage", "age_days", "risk", "flag", "why"] + (["actually_late"] if show_truth else [])
        st.dataframe(
            df[cols], width="stretch", hide_index=True,
            column_config={
                "risk": st.column_config.ProgressColumn("risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "age_days": st.column_config.NumberColumn("age (days)", format="%.0f"),
                "why": st.column_config.TextColumn("top reasons (↑ raises risk, ↓ lowers)", width="large"),
            })
        if show_truth:
            d = df.dropna(subset=["actually_late"])
            if d["actually_late"].nunique() == 2:
                from sklearn.metrics import roc_auc_score
                st.caption(f"Check on this snapshot: AUC {roc_auc_score(d['actually_late'], d['risk']):.3f} "
                           f"over {len(d)} issues whose outcome is known.")

with tab_issue:
    if df.empty:
        st.info("No open issues on this date.")
    else:
        iid = st.selectbox("Issue", df["Issue_ID"].tolist(),
                           format_func=lambda i: f"{i} — {df.set_index('Issue_ID').loc[i, 'Title'][:80]}")
        row = df.set_index("Issue_ID").loc[iid]
        st.subheader(f"Risk {row['risk']:.2f} — {row['flag']}")
        left, right = st.columns([3, 2])
        with left:
            traj = b.trajectory(int(iid), pd.Timestamp(date))
            if len(traj):
                st.line_chart(traj.set_index("date")["risk"], height=280)
                st.caption("How the model's risk for this issue changed as events happened.")
        with right:
            ex = b.explain_issue(int(iid), pd.Timestamp(date))
            ex["direction"] = np.where(ex["push"] > 0, "raises risk", "lowers risk")
            st.dataframe(ex, hide_index=True, width="stretch",
                         column_config={"push": st.column_config.NumberColumn("effect (log-odds)", format="%+.2f")})

with tab_scrum:
    sprints = b.active_sprints(pid, pd.Timestamp(date))
    if sprints.empty:
        st.info("No sprint from the Sprint table was running in this project on this date. "
                "Scrum mode covers ~33k sprint-tracked issues across the projects that use Jira sprints.")
    else:
        sk = st.selectbox("Sprint", sprints["sprint_key"].tolist(),
                          format_func=lambda k: sprints.set_index("sprint_key").loc[k, "name"])
        s = sprints.set_index("sprint_key").loc[sk]
        days_left = (pd.to_datetime(s["end"], unit="s") - pd.Timestamp(date)).days
        st.caption(f"{pd.to_datetime(s['start'], unit='s').date()} → {pd.to_datetime(s['end'], unit='s').date()} · "
                   f"{days_left} days left · {s['committed']} committed issues")
        si = b.sprint_issues(sk, pd.Timestamp(date))
        if si.empty:
            st.info("All committed issues are already done or moved out.")
        else:
            st.metric("Likely to spill", f"{(si['flag'] == 'likely to spill').sum()} of {len(si)} still open")
            cols = ["Issue_ID", "Title", "stage", "spill_risk", "flag"] + (["actually_spilled"] if show_truth else [])
            st.dataframe(si[cols], hide_index=True, width="stretch",
                         column_config={"spill_risk": st.column_config.ProgressColumn(
                             "spill risk", min_value=0.0, max_value=1.0, format="%.2f")})

with tab_card:
    st.markdown("### What the models do and how good they are")
    path = "md files/task-J-overnight-summary.md"
    if os.path.exists(path):
        st.markdown(open(path).read())
    else:
        st.info("Run the Task J scripts to generate the model card.")
