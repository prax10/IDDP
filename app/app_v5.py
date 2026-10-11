"""
Delay Radar v5 -- Streamlit app for the improved models (Tasks J/K).

Run from the capstone folder:   streamlit run app/app_v5.py
The original app/app.py (v3 models) is untouched.

Two modes, chosen by the project manager in the sidebar:

  FLOW MODE   -- "will this issue take longer than this project normally takes?"
                 Pick a project and a date; every open issue is scored with features
                 computed from events before that date, with plain-English reasons.
  SCRUM MODE  -- the PM runs sprints, so the deadline is the sprint end.
                 Sprint board (KPIs, burndown, spill risk per committed issue),
                 issue detail, sprint history (predicted vs actual spills).

This is a historical replay on TAWOS data, not a live Jira feed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from app_v5_backend import Backend
from scrum_backend import ScrumBackend

st.set_page_config(page_title="Delay Radar v5", layout="wide", page_icon="📡")
RED, BLUE, GREY = "#d1495b", "#2a78d6", "#898781"
RECO_NOTE = ("Each row re-scores this issue with one thing changed (e.g. status moved to review): the model's "
             "estimate of how its risk would move, not a guaranteed causal effect. Checked against history "
             "(test period, issues still open a week later, matched on risk): every action points the same way "
             "in real outcomes, but status moves (start work / review) are overstated about 2x, reassignment is "
             "about right, and commenting is understated.")


@st.cache_resource(show_spinner="Loading change history and models (first run takes ~1 minute)...")
def backend():
    return Backend()


@st.cache_resource(show_spinner="Preparing Scrum mode (sprint memberships)...")
def scrum_backend(_b):
    return ScrumBackend(_b.fe, _b.titles)


b = backend()

with st.sidebar:
    st.title("📡 Delay Radar v5")
    mode = st.radio("Mode", ["Flow mode", "Scrum mode"], horizontal=True,
                    help="Flow: deadline = the project's typical duration. "
                         "Scrum: deadline = the end of the sprint the issue is committed to.")


# =====================================================================================
# FLOW MODE
# =====================================================================================
def flow_mode():
    projects = b.projects()
    projects = projects[projects["n"] >= 200].sort_values("n", ascending=False)
    with st.sidebar:
        pid = st.selectbox("Project", projects.index.tolist(),
                           format_func=lambda p: f"Project {p}  ({projects.loc[p, 'n']:,} issues)")
        first = pd.to_datetime(projects.loc[pid, "first"], unit="s").date()
        last = pd.to_datetime(projects.loc[pid, "last"], unit="s").date()
        default = (pd.Timestamp(last) - pd.Timedelta(days=180)).date()
        date = st.date_input("As of date", value=default, min_value=first, max_value=last)
        st.caption("Everything is computed from events before this date.")
        show_truth = st.toggle("Show actual outcomes", value=False)
        show_overdue = st.toggle("Include already-overdue issues", value=False,
                                 help="Issues already older than the project's typical duration are late by "
                                      "definition; hidden by default so the list focuses on issues that can still be saved.")

    tab_port, tab_issue, tab_card = st.tabs(["Portfolio", "Issue detail", "Model card"])
    df_all = b.open_issues(pid, pd.Timestamp(date))
    df = df_all if (show_overdue or df_all.empty) else df_all[~df_all["overdue"]].reset_index(drop=True)

    with tab_port:
        if df.empty:
            st.info("No open issues on this date.")
        else:
            late = (df["flag"] == "likely late").sum()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Open issues", f"{len(df_all):,}")
            c2.metric("Already overdue", f"{int(df_all['overdue'].sum()):,}")
            c3.metric("Flagged likely late", f"{late:,}", f"{late / max((~df['overdue']).sum(), 1):.0%} of not-yet-overdue")
            c4.metric("Confident late calls", f"{(df['confidence'] == 'confident: late').sum():,}",
                      help="Issues whose 90% conformal prediction set contains only 'late'. On held-out test data "
                           "the sets contain the true outcome ~92% of the time, and a 'confident: late' call on a not-yet-"
                           "overdue issue was right ~97% of the time. 'Uncertain' means the model cannot rule either "
                           "outcome out at that level.")
            c5.metric("Median age (days)", f"{df['age_days'].median():.0f}")
            cols = ["Issue_ID", "Title", "stage", "age_days", "risk", "flag", "confidence", "finish_in", "why"] + \
                   (["actually_late"] if show_truth else [])
            st.dataframe(df[cols], width="stretch", hide_index=True, column_config={
                "risk": st.column_config.ProgressColumn("risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "age_days": st.column_config.NumberColumn("age (days)", format="%.0f"),
                "finish_in": st.column_config.TextColumn("likely done in (80% range)",
                                                         help="Conformalised quantile regression; on test data the "
                                                              "80% range held the true finish ~84% of the time."),
                "why": st.column_config.TextColumn("top reasons (↑ raises risk, ↓ lowers)", width="large")})
            if show_truth:
                d = df.dropna(subset=["actually_late"])
                if d["actually_late"].nunique() == 2:
                    from sklearn.metrics import roc_auc_score
                    top = d.head(max(len(d) // 10, 1))
                    st.caption(f"Check on this snapshot: AUC {roc_auc_score(d['actually_late'], d['risk']):.3f}; "
                               f"{top['actually_late'].mean():.0%} of the top-10% flags really ended up late "
                               f"(base rate {d['actually_late'].mean():.0%}).")

    with tab_issue:
        if df.empty:
            st.info("No open issues on this date.")
        else:
            iid = st.selectbox("Issue", df["Issue_ID"].tolist(),
                               format_func=lambda i: f"{i} — {df.set_index('Issue_ID').loc[i, 'Title'][:80]}")
            row = df.set_index("Issue_ID").loc[iid]
            st.subheader(f"Risk {row['risk']:.2f} — {row['flag']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Model confidence (90%)", row["confidence"].replace("confident: ", "") if row["confidence"] else "–",
                      help="From a 90% conformal prediction set calibrated on the validation period.")
            m2.metric("Likely done in", row["finish_in"] or "–", help="80% range for the days still remaining.")
            m3.metric("Open for", f"{row['age_days']:.0f} days")
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
            what_would_help(int(iid), pd.Timestamp(date), row)

    with tab_card:
        card("md files/task-J-overnight-summary.md")
        with st.expander("Further evidence (Task L): external companies, baselines, uncertainty, recommendations"):
            card("md files/task-L-results.md")


def what_would_help(iid, date, row):
    st.markdown("#### What would help?")
    if row["overdue"]:
        st.caption("This issue is already past the project's typical duration, so it is late by definition; "
                   "actions now affect how late, not whether.")
    try:
        base, rec = b.recommend(iid, date)
    except Exception as e:      # never break the page over a recommendation
        st.caption(f"No recommendations available ({e}).")
        return
    if rec.empty:
        st.caption("No candidate actions for this issue's stage.")
        return
    rec = rec.assign(verdict=np.where(rec["change"] <= -0.02, "helps", np.where(rec["change"] >= 0.02, "hurts", "little effect")))
    st.dataframe(rec, hide_index=True, width="stretch", column_config={
        "new_risk": st.column_config.ProgressColumn("risk if done", min_value=0.0, max_value=1.0, format="%.2f"),
        "change": st.column_config.NumberColumn("change", format="%+.2f")})
    st.caption(RECO_NOTE)


# =====================================================================================
# SCRUM MODE
# =====================================================================================
def scrum_mode():
    sb = scrum_backend(b)
    projects = sb.projects()
    with st.sidebar:
        pid = st.selectbox("Scrum project", projects.index.tolist(),
                           format_func=lambda p: f"Project {p}  ({projects.loc[p, 'sprints']} sprints)")
        sprints = sb.sprints(pid)
        label = lambda k: (lambda r: f"{r['name']}  ({pd.to_datetime(r['start'], unit='s'):%d %b %Y} → "
                                     f"{pd.to_datetime(r['planned_end'], unit='s'):%d %b})")(
            sprints.set_index("sprint_key").loc[k])
        key = st.selectbox("Sprint", sprints["sprint_key"].tolist(), index=min(3, len(sprints) - 1), format_func=label)
        s = sprints.set_index("sprint_key").loc[key]
        start = pd.to_datetime(s["start"], unit="s").to_pydatetime()
        end = pd.to_datetime(min(s["close"], s["planned_end"]) - 3600, unit="s").to_pydatetime()
        if end <= start:
            end = start + pd.Timedelta(hours=2)
        day = st.slider("Day of the sprint", min_value=start, max_value=end,
                        value=start + (end - start) / 2, format="DD MMM, HH:mm")
        show_truth = st.toggle("Show actual outcomes", value=False)
    T = int(pd.Timestamp(day).timestamp())

    tab_board, tab_issue, tab_hist, tab_card = st.tabs(["Sprint board", "Issue detail", "Sprint history", "Model card"])
    kpi, tbl = sb.snapshot(key, T)

    with tab_board:
        st.subheader(s["name"])
        c = st.columns(6)
        c[0].metric("Committed", kpi["committed"])
        c[1].metric("Done", kpi["done"])
        c[2].metric("Still open", kpi["open"])
        iv = kpi.get("spill_interval")
        c[3].metric("Predicted to spill", f"{kpi['expected_spills']:.1f}",
                    f"90% range {iv[0]}–{iv[1]}" if iv else None, delta_color="off",
                    help="Sum of spill risks of open issues, plus issues already moved out unfinished. "
                         "The range is split-conformal, calibrated on validation-period sprints "
                         "(it held the true count in ~96% of test sprints).")
        c[4].metric("Predicted completion", f"{1 - kpi['expected_spills'] / max(kpi['committed'], 1):.0%}")
        c[5].metric("Days left", f"{max(kpi['days_left'], 0):.1f}")

        bd = sb.burndown(key, T)
        if len(bd):
            base = alt.Chart(bd).mark_line(color=BLUE, point=True).encode(
                x=alt.X("date:T", title=None), y=alt.Y("open_commitments:Q", title="open committed issues"))
            proj = pd.DataFrame({"date": [bd["date"].iloc[-1], pd.to_datetime(s["planned_end"], unit="s")],
                                 "open_commitments": [bd["open_commitments"].iloc[-1], kpi["expected_spills"]]})
            fc = alt.Chart(proj).mark_line(color=RED, strokeDash=[6, 4]).encode(x="date:T", y="open_commitments:Q")
            st.altair_chart((base + fc).properties(height=240), width="stretch")
            st.caption("Blue: open commitments each day. Red dashed: where the model expects the sprint to end "
                       "(predicted spill count at the planned end).")

        if tbl.empty:
            st.success("Nothing left open in this sprint at this moment.")
        else:
            cols = ["Issue_ID", "Title", "stage", "story_points", "spill_risk", "flag", "why"] + \
                   (["actually_spilled"] if show_truth else [])
            st.dataframe(tbl[cols], hide_index=True, width="stretch", column_config={
                "spill_risk": st.column_config.ProgressColumn("spill risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "story_points": st.column_config.NumberColumn("SP", format="%.0f"),
                "why": st.column_config.TextColumn("top reasons (↑ raises risk, ↓ lowers)", width="large")})
            if show_truth:
                spilled = int(tbl["actually_spilled"].sum()) + kpi["removed"]
                st.caption(f"Actual outcome: {spilled} of {kpi['committed']} committed issues spilled "
                           f"(model predicted {kpi['expected_spills']:.1f}).")

    with tab_issue:
        if tbl.empty:
            st.info("No open committed issues at this moment.")
        else:
            iid = st.selectbox("Issue", tbl["Issue_ID"].tolist(),
                               format_func=lambda i: f"{i} — {tbl.set_index('Issue_ID').loc[i, 'Title'][:80]}")
            row = tbl.set_index("Issue_ID").loc[iid]
            st.subheader(f"Spill risk {row['spill_risk']:.2f} — {row['flag']}")
            left, right = st.columns([3, 2])
            with left:
                tr = sb.issue_trajectory(int(iid), key, T)
                if len(tr):
                    st.line_chart(tr.set_index("date")["spill_risk"], height=260)
                    st.caption("Spill risk for this issue since it joined the sprint.")
            with right:
                ex = sb.explain(int(iid), key, T)
                ex["direction"] = np.where(ex["push"] > 0, "raises risk", "lowers risk")
                st.dataframe(ex, hide_index=True, width="stretch",
                             column_config={"push": st.column_config.NumberColumn("effect (log-odds)", format="%+.2f")})

    with tab_hist:
        h = sb.history(pid, T)
        if h.empty:
            st.info("No closed sprints before this date.")
        else:
            m = h.dropna(subset=["predicted_spills"])
            corr = np.corrcoef(m["predicted_spills"], m["actual_spills_open_at_start"])[0, 1] if len(m) > 2 else np.nan
            mae = (m["predicted_spills"] - m["actual_spills_open_at_start"]).abs().mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Past sprints", len(h))
            c2.metric("Avg completion rate", f"{h['completion_rate'].mean():.0%}")
            c3.metric("Spill-count error (at sprint start)", f"±{mae:.1f} issues", f"corr {corr:.2f}")
            long = m.melt(id_vars=["start_date", "name"], value_vars=["actual_spills_open_at_start", "predicted_spills"],
                          var_name="series", value_name="spills")
            long["series"] = long["series"].map({"actual_spills_open_at_start": "actual", "predicted_spills": "predicted at start"})
            ch = alt.Chart(long).mark_line(point=True).encode(
                x=alt.X("start_date:T", title="sprint start"), y=alt.Y("spills:Q", title="issues spilled"),
                color=alt.Color("series:N", scale=alt.Scale(range=[GREY, RED])), tooltip=["name", "series", "spills"])
            st.altair_chart(ch.properties(height=260), width="stretch")
            st.dataframe(h[["name", "start_date", "committed", "spilled", "completion_rate", "predicted_spills"]],
                         hide_index=True, width="stretch", column_config={
                             "completion_rate": st.column_config.NumberColumn("completion", format="percent"),
                             "predicted_spills": st.column_config.NumberColumn("predicted spills (at start)", format="%.1f")})

    with tab_card:
        card("md files/scrum-mode-section.md")


def card(path):
    if os.path.exists(path):
        st.markdown(open(path).read())
    else:
        st.info(f"{path} not found.")


if mode == "Flow mode":
    flow_mode()
else:
    scrum_mode()
