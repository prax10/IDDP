"""
Intelligent Dynamic Delay Prediction -- Streamlit prototype. Demo wrapper
around already-final models -- no retraining, no live Jira sync, no task
editing, no notifications. Mission Control replays REAL historical data
with real leakage-safe predictions -- it is a replay, not a live feed,
and the UI says so.

Run with: streamlit run app/app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

import demo_data as dd
import reassignment as ra
import reference_stats as rs
import explain as ex

APP_NAME = "Intelligent Dynamic Delay Prediction"
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🚦")

RISK_THRESHOLD = dd.DEFAULT_THRESHOLD  # 0.5, stated in the UI
CROSSOVER_CHECKPOINT = dd.CROSSOVER_CHECKPOINT  # M3
GREEN, AMBER, RED = dd.RISK_COLORS["low"], dd.RISK_COLORS["medium"], dd.RISK_COLORS["high"]
STATIC_TEST_AUC = 0.743
DYNAMIC_M10_AUC = 0.859

MC_PROJECT_DEFAULT = dd.MISSION_CONTROL_PROJECT
TEST_PERIOD_START = dd.TEST_PERIOD_START


def risk_badge(risk):
    band = dd.risk_band(risk)
    color = dd.RISK_COLORS[band]
    label = {"low": "On track", "medium": "Watch", "high": "At risk"}[band]
    return f'<span style="color:{color}; font-weight:600;">{label} ({risk:.0%})</span>'


def fmt_int(x):
    """Step 8: whole numbers, no decimals -- for IDs, counts."""
    return "" if pd.isna(x) else f"{int(round(x))}"


def fmt_num(x, places=2):
    """Step 8: non-integers to at most `places` decimals."""
    if pd.isna(x):
        return ""
    if float(x) == int(x):
        return f"{int(x)}"
    return f"{x:.{places}f}"


@st.cache_resource
def get_models():
    return dd.load_models()


@st.cache_resource
def get_reference_stats():
    return rs.load_reference_stats()


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


@st.cache_data
def get_checkpoints_with_status():
    return pd.read_csv(
        "data/features/phase8_checkpoints.csv",
        usecols=["Issue_ID", "checkpoint_index", "current_status", "last_status_change_time"],
        parse_dates=["last_status_change_time"],
    )


@st.cache_data
def get_mission_control_data(project_id, _static_model, _dynamic_model):
    return dd.load_project_for_replay(project_id, _static_model, _dynamic_model)


@st.cache_data
def get_blocking_graph(project_id, issue_ids):
    return dd.load_blocking_graph(project_id, issue_ids)


@st.cache_data
def get_project_date_range(project_id):
    return dd.project_date_range(project_id)


@st.cache_data
def get_projects_with_test_period():
    return dd.projects_with_test_period()


@st.cache_data
def get_project_options():
    """{project_id: n_issues} for every project Mission Control can offer --
    i.e. projects with at least one genuinely held-out (test-split) issue."""
    raw = dd.load_raw_display_fields()
    feat = dd.load_feature_table()
    raw = raw[raw["ID"].isin(feat["ID"])]
    eligible = get_projects_with_test_period()
    counts = raw[raw["Project_ID"].isin(eligible)].groupby("Project_ID").size()
    return counts.to_dict()


@st.cache_data
def get_mission_control_timeline(project_id, issues, checkpoints, start, end):
    span_days = (end - start).days
    freq = "D" if span_days <= 400 else ("W" if span_days <= 1500 else "MS")
    dates = pd.date_range(start, end, freq=freq)
    rows = []
    for d in dates:
        cls = dd.classify_at_time(issues, checkpoints, d)
        open_mask = dd.open_at_time(issues, d)
        open_cls = cls[cls["ID"].isin(issues.loc[open_mask, "ID"])]
        counts = open_cls["status"].value_counts()
        rows.append({
            "date": d,
            "on_track": counts.get("on_track", 0),
            "at_risk": counts.get("at_risk", 0),
            "overrunning": counts.get("overrunning", 0),
        })
    return pd.DataFrame(rows)


static_model, dynamic_model = get_models()
ref_stats = get_reference_stats()
portfolio_raw, feat_full = get_portfolio_data()
portfolio = get_scored_portfolio(static_model, dynamic_model, portfolio_raw)
portfolio["at_risk"] = portfolio["dynamic_risk"] >= RISK_THRESHOLD
all_checkpoints = get_all_checkpoints()
checkpoint_status = get_checkpoints_with_status()

st.title(APP_NAME)
st.caption(
    f"XGBoost · trained on 203,290 resolved issues across 39 projects · "
    f"AUC {STATIC_TEST_AUC:.3f} static / {DYNAMIC_M10_AUC:.3f} dynamic at M10"
)

with st.expander("How it works", expanded=False):
    st.markdown(f"""
- **Label definition:** an issue is "delayed" if its resolution time exceeds its own project's
  *median* resolution time -- and that median is computed **only from training-split issues**
  (never from validation or test issues), so the label itself doesn't leak future information.
- **Leakage-safe by design:** every feature reflects only information available at or before the
  issue's own creation time, or, for in-progress predictions, at or before the specific checkpoint
  being scored.
- **Predictions for in-progress issues are checkpoint-based**, re-evaluated at 10 checkpoints
  across an issue's expected lifetime.
- **The key finding driving this app:** dynamic (checkpoint-based) predictions only become more
  reliable than a creation-time-only prediction after roughly **30% of an issue's expected duration
  has elapsed (checkpoint M3)** -- confirmed via bootstrap confidence intervals. Mission Control and
  the trajectory chart both enforce this rule directly: before M3 you see the static estimate,
  labelled as such.
- **Overrunning vs. at risk (Mission Control):** an issue that has already exceeded its own project's
  historical median resolution time **for its own issue type** is a **fact**, not a forecast -- it's
  tracked separately from the genuine at-risk/on-track prediction queue. That baseline is capped at
  the (project, type) level with a minimum of {dd.BASELINE_MIN_GROUP} prior resolved issues and
  **never falls through to a project-wide or global median** -- a type with too little history is
  labelled "insufficient history" and is never classified overrunning, rather than being compared
  against a baseline that isn't actually representative of it. We call this "overrunning" or "past
  expected duration," never "past deadline" -- TAWOS has no deadline field, nothing in the data
  represents a commitment.
- **Decision threshold:** **{RISK_THRESHOLD}** for at-risk/on-track calls; {dd.F1_THRESHOLD} was
  separately tuned to optimize F1 if that's the operating point you care about.
- **Risk colour coding (used everywhere):**
  <span style="color:{GREEN};font-weight:600;">green &lt; 0.4</span>,
  <span style="color:{AMBER};font-weight:600;">amber 0.4-0.7</span>,
  <span style="color:{RED};font-weight:600;">red &gt; 0.7</span>.
- **Mission Control is a REPLAY of real historical data, not a live feed.** It shows what the
  model would genuinely have predicted, day by day, using only information available as of each
  date shown -- no data has been fabricated or simulated. Each project's full history is shown; the
  portion before {TEST_PERIOD_START.date()} is shaded on the chart as **in-sample** (the model
  trained on it), so only the unshaded portion is a genuine out-of-sample prediction. Only projects
  with at least one issue created on/after {TEST_PERIOD_START.date()} are offered, so every project
  in the selector has some genuine held-out portion to show.
- **Manager targets never enter the model.** A target is a policy statement ("we want Bugs done in
  5 days"), not evidence about how long work actually takes -- feeding it into the model would let
  changing what you *want* change the model's risk estimate, which is backwards. Targets are pure
  arithmetic against elapsed time, computed entirely outside the model.
- **Blocking relationships:** directed link types ("depends on" / "has to be done before" /
  "blocks" and equivalents) let us tell which issue blocks which -- verified directional and
  reciprocal in the raw data, not inferred from keyword guesses.
- **Other tabs' data note:** outside Mission Control, the app uses held-out (test-split) resolved
  issues at their last observed checkpoint as a stand-in "current" portfolio, since there is no
  live feed of in-progress issues with computed features in this project.
""", unsafe_allow_html=True)

tab_mc, tab2, tab3 = st.tabs(["🎛️ Mission Control", "Issue Risk & Explanation", "Team"])

# ---------------------------------------------------------------------
# Tab: Mission Control (time-travel replay)
# ---------------------------------------------------------------------
with tab_mc:

    def _project_default_step(pid):
        start, end = get_project_date_range(pid)
        span_years = (end - start).days / 365.25
        return "1 month" if span_years > 2 else "1 week"

    def _project_default_cursor(pid):
        start, end = get_project_date_range(pid)
        return TEST_PERIOD_START + (end - TEST_PERIOD_START) / 3

    def _apply_project_switch(pid):
        st.session_state.mc_active_project = pid
        st.session_state.mc_targets = {}
        st.session_state.mc_closed_ids = set()
        st.session_state.mc_table_filter = None
        st.session_state.mc_cursor = _project_default_cursor(pid)
        st.session_state.mc_step = _project_default_step(pid)
        st.session_state.mc_playing = False

    if "mc_active_project" not in st.session_state:
        _apply_project_switch(MC_PROJECT_DEFAULT)
    if "mc_confirm_full_reset" not in st.session_state:
        st.session_state.mc_confirm_full_reset = False

    project_counts = get_project_options()
    project_choices = sorted(project_counts.keys())

    # --- Step 1: project selector (with confirmation only when there are
    # unsaved manager modifications to lose) ---
    ui_project = st.selectbox(
        "Project", project_choices,
        index=project_choices.index(st.session_state.mc_active_project),
        format_func=lambda p: f"{p} ({project_counts[p]} issues)",
        key="mc_project_select",
    )
    has_mods = bool(st.session_state.mc_targets) or bool(st.session_state.mc_closed_ids)
    if ui_project != st.session_state.mc_active_project:
        if not has_mods:
            _apply_project_switch(ui_project)
        else:
            st.warning(
                f"Switching to project {ui_project} will clear {len(st.session_state.mc_targets)} "
                f"target(s) and {len(st.session_state.mc_closed_ids)} closed issue(s) set for "
                f"project {st.session_state.mc_active_project}."
            )
            cc1, cc2 = st.columns(2)
            if cc1.button("Confirm switch", key="mc_confirm_switch_yes"):
                _apply_project_switch(ui_project)
                st.rerun()
            if cc2.button("Cancel", key="mc_confirm_switch_cancel"):
                del st.session_state["mc_project_select"]
                st.rerun()

    project_id = st.session_state.mc_active_project
    project_start, project_end = get_project_date_range(project_id)

    mc_issues, mc_checkpoints = get_mission_control_data(project_id, static_model, dynamic_model)
    blocking_graph = get_blocking_graph(project_id, tuple(mc_issues["ID"]))
    title_lookup = mc_issues.set_index("ID")["Title"]
    type_lookup = mc_issues.set_index("ID")["Type_Normalized"]

    # --- Step 2: sticky context header ---
    st.markdown(
        f"""
        <div style="position: sticky; top: 0; z-index: 999; background-color: rgba(240,240,240,0.97);
                    padding: 10px 14px; border-radius: 6px; border: 1px solid #ccc; margin-bottom: 10px;
                    font-size: 1.05rem;">
        🎛️ <b>Mission Control</b> &nbsp;·&nbsp; Project <b>{project_id}</b>
        &nbsp;·&nbsp; Replay date: <b>{pd.Timestamp(st.session_state.mc_cursor).date()}</b>
        &nbsp;·&nbsp; Step: <b>{st.session_state.mc_step}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Replaying real historical issues with real leakage-safe predictions, not a live feed -- "
        f"full history {project_start.date()} to {project_end.date()}, held-out test period starts "
        f"{TEST_PERIOD_START.date()}."
    )

    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        cursor_date = st.slider(
            "Replay date", min_value=project_start.to_pydatetime(), max_value=project_end.to_pydatetime(),
            value=pd.Timestamp(st.session_state.mc_cursor).to_pydatetime(), format="YYYY-MM-DD",
            key=f"mc_slider_{project_id}",
        )
        st.session_state.mc_cursor = pd.Timestamp(cursor_date)
    with col_b:
        step_choices = ["1 day", "1 week", "1 month"]
        step_label = st.selectbox("Step size", step_choices,
                                   index=step_choices.index(st.session_state.mc_step),
                                   key=f"mc_step_select_{project_id}")
        st.session_state.mc_step = step_label
        step = {"1 day": pd.Timedelta(days=1), "1 week": pd.Timedelta(weeks=1),
                "1 month": pd.DateOffset(months=1)}[step_label]
    with col_c:
        playing = st.checkbox("▶ Play", value=st.session_state.mc_playing, key=f"mc_play_checkbox_{project_id}")
        st.session_state.mc_playing = playing
    with col_d:
        if st.button("⟲ Reset to start"):
            st.session_state.mc_cursor = project_start
            st.session_state.mc_playing = False
            st.rerun()
        if not st.session_state.mc_confirm_full_reset:
            if st.button("🔄 Full reset"):
                st.session_state.mc_confirm_full_reset = True
                st.rerun()
        else:
            st.warning("Full reset clears all targets and closed issues, and resets the cursor.")
            rc1, rc2 = st.columns(2)
            if rc1.button("Yes, reset", key="mc_full_reset_yes"):
                _apply_project_switch(project_id)
                st.session_state.mc_confirm_full_reset = False
                st.rerun()
            if rc2.button("Cancel", key="mc_full_reset_cancel"):
                st.session_state.mc_confirm_full_reset = False
                st.rerun()

    T = pd.Timestamp(st.session_state.mc_cursor)
    T_prev = max(project_start, T - step)

    # --- Step 3 (config panel): historical baselines & manager targets ---
    with st.expander("⚙️ Configuration: historical baselines & manager targets", expanded=False):
        st.caption(
            "Targets change how issues are categorised against your standard. They do not change "
            "the model's risk estimate, which is computed from this project's historical behaviour."
        )
        baseline = dd.project_type_duration_baseline(feat_full, project_id)
        mc_targets = st.session_state.mc_targets
        for type_norm in sorted(baseline.index.astype(str)):
            brow = baseline.loc[type_norm]
            bcol1, bcol2, bcol3 = st.columns([2, 2, 3])
            with bcol1:
                if brow["sufficient"]:
                    st.write(f"**{type_norm}**: {brow['median_days']:.1f}-day historical median "
                             f"(n={int(brow['n'])})")
                else:
                    st.write(f"**{type_norm}**: insufficient history (n={int(brow['n'])})")
            with bcol2:
                use_default = st.checkbox(
                    "Use default (no target)", value=(type_norm not in mc_targets),
                    key=f"mc_target_default_{project_id}_{type_norm}",
                )
                if use_default:
                    mc_targets.pop(type_norm, None)
                else:
                    default_val = mc_targets.get(
                        type_norm, brow["median_days"] if brow["sufficient"] else 5.0
                    )
                    target_val = st.number_input(
                        "Target (days)", min_value=0.1, value=float(default_val), step=0.5,
                        key=f"mc_target_val_{project_id}_{type_norm}",
                    )
                    mc_targets[type_norm] = float(target_val)
            with bcol3:
                if type_norm in mc_targets:
                    rc = dd.target_reality_check(feat_full, project_id, type_norm, mc_targets[type_norm])
                    if rc is None:
                        st.caption("Insufficient history to compute a reality check.")
                    else:
                        median_days, pct_missed, n = rc
                        st.caption(
                            f"Target: {mc_targets[type_norm]:.1f} days · History: {median_days:.1f}-day "
                            f"median · {pct_missed:.0%} of past {n} {type_norm}s in this project would "
                            f"have missed this target."
                        )
        if mc_targets:
            deltas = []
            for t, tgt in mc_targets.items():
                if t in baseline.index and baseline.loc[t, "sufficient"]:
                    med = baseline.loc[t, "median_days"]
                    deltas.append((med - tgt) / med)
            if deltas:
                avg_pct = sum(deltas) / len(deltas) * 100
                direction = "tighter" if avg_pct >= 0 else "looser"
                st.markdown(
                    f"**Your targets are on average {abs(avg_pct):.0f}% {direction} than this "
                    f"project's historical delivery.**"
                )

    mc_targets = st.session_state.mc_targets
    mc_closed_ids = st.session_state.mc_closed_ids

    cls_cur = dd.classify_at_time(mc_issues, mc_checkpoints, T).set_index("ID")
    cls_prev = dd.classify_at_time(mc_issues, mc_checkpoints, T_prev).set_index("ID")
    open_cur_mask = dd.open_at_time(mc_issues, T)
    open_prev_mask = dd.open_at_time(mc_issues, T_prev)

    # Design constraint 2: manager-closed issues leave the open pool entirely.
    open_cur_ids = set(mc_issues.loc[open_cur_mask, "ID"]) - mc_closed_ids
    open_prev_ids = set(mc_issues.loc[open_prev_mask, "ID"]) - mc_closed_ids

    def _counts(cls, ids):
        if not ids:
            return {"overrunning": 0, "at_risk": 0, "on_track": 0}
        sub = cls.loc[list(ids), "status"].value_counts()
        return {k: int(sub.get(k, 0)) for k in ["overrunning", "at_risk", "on_track"]}

    cur_counts = _counts(cls_cur, open_cur_ids)
    prev_counts = _counts(cls_prev, open_prev_ids)

    non_overrun_cur = [i for i in open_cur_ids if cls_cur.loc[i, "status"] != "overrunning"]
    non_overrun_prev = [i for i in open_prev_ids if cls_prev.loc[i, "status"] != "overrunning"]
    avg_risk = float(cls_cur.loc[non_overrun_cur, "risk"].mean()) if non_overrun_cur else float("nan")
    avg_risk_prev = float(cls_prev.loc[non_overrun_prev, "risk"].mean()) if non_overrun_prev else float("nan")

    def _vs_target_status(ids, cls):
        out = {}
        for iid in ids:
            elapsed_d = cls.loc[iid, "elapsed_minutes"] / 1440.0
            out[iid] = dd.target_status(elapsed_d, str(type_lookup.get(iid)), mc_targets)
        return out

    vs_target_cur = _vs_target_status(open_cur_ids, cls_cur)
    vs_target_prev = _vs_target_status(open_prev_ids, cls_prev)
    past_target_cur = sum(1 for v in vs_target_cur.values() if v == "past_target")
    past_target_prev = sum(1 for v in vs_target_prev.values() if v == "past_target")

    # --- Step 5 (Task F): 5 (or 6) metric cards ---
    n_open, n_open_prev = len(open_cur_ids), len(open_prev_ids)
    ncols = 6 if mc_targets else 5
    mcols = st.columns(ncols)
    mcols[0].metric("Open", n_open, f"{n_open - n_open_prev:+d} vs. last step")
    mcols[1].metric("Overrunning", cur_counts["overrunning"],
                     f"{cur_counts['overrunning'] - prev_counts['overrunning']:+d} vs. last step")
    mcols[2].metric("At risk", cur_counts["at_risk"],
                     f"{cur_counts['at_risk'] - prev_counts['at_risk']:+d} vs. last step")
    mcols[3].metric("On track", cur_counts["on_track"],
                     f"{cur_counts['on_track'] - prev_counts['on_track']:+d} vs. last step")
    avg_delta = (avg_risk - avg_risk_prev) if not (np.isnan(avg_risk) or np.isnan(avg_risk_prev)) else 0.0
    mcols[4].metric("Avg risk (excl. overrunning)", f"{avg_risk:.0%}" if not np.isnan(avg_risk) else "n/a",
                     f"{avg_delta:+.0%} vs. last step")
    if mc_targets:
        mcols[5].metric("Past target", past_target_cur, f"{past_target_cur - past_target_prev:+d} vs. last step")
    st.caption(
        "Overrunning issues (already past their own project+type historical median) are excluded "
        "from the average and tracked separately below -- they're a fact about the past, not a "
        "prediction. Manager-closed issues leave the open pool entirely."
    )

    # --- filter row (metric cards act as drill-down filters) ---
    filter_defs = [(None, "Show all"), ("overrunning", "Overrunning"), ("at_risk", "At risk"),
                   ("on_track", "On track")]
    if mc_targets:
        filter_defs.append(("past_target", "Past target"))
    fcols = st.columns(len(filter_defs))
    for i, (fkey, flabel) in enumerate(filter_defs):
        if fcols[i].button(flabel, key=f"mc_filter_btn_{fkey}"):
            st.session_state.mc_table_filter = fkey
    if st.session_state.mc_table_filter:
        st.caption(f"Filtering: **{st.session_state.mc_table_filter}** -- click 'Show all' to clear.")

    # --- Step 4 (Task E): fixed running scoreboard, now excluding manager-closed issues ---
    resolved_so_far = mc_issues[
        (mc_issues["Resolution_Date"] > project_start) & (mc_issues["Resolution_Date"] <= T)
        & (~mc_issues["ID"].isin(mc_closed_ids))
    ]
    if len(resolved_so_far):
        rs_risk = dd.risk_at_time(mc_issues, mc_checkpoints, T).set_index("ID")
        rs_risk_sub = rs_risk.loc[resolved_so_far["ID"]]
        predicted_delayed = rs_risk_sub["risk"] >= RISK_THRESHOLD
        actual_delayed = resolved_so_far.set_index("ID")["delayed_v2"].astype(bool)
        correct = int((predicted_delayed.values == actual_delayed.values).sum())
        majority_class = int(actual_delayed.mean() >= 0.5)
        majority_correct = int((actual_delayed.values == bool(majority_class)).sum())
        st.info(
            f"📊 **Running scoreboard (since {project_start.date()}):** {correct} / {len(resolved_so_far)} "
            f"resolved issues correctly called ({correct/len(resolved_so_far):.0%} accuracy) · "
            f"predicting '{'delayed' if majority_class else 'on time'}' for everything would score "
            f"{majority_correct/len(resolved_so_far):.0%}. Manager-closed issues are excluded."
        )
    else:
        st.caption("📊 Running scoreboard: no issues resolved yet since the replay start.")

    # --- Timeline chart, with in-sample shading (design constraint 3) ---
    st.subheader("Open-issue volume over time (by status)")
    timeline = get_mission_control_timeline(project_id, mc_issues, mc_checkpoints, project_start, project_end)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if project_start < TEST_PERIOD_START:
        shade_end = min(TEST_PERIOD_START, project_end)
        ax.axvspan(project_start, shade_end, color="grey", alpha=0.15)
        ax.text(project_start + (shade_end - project_start) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                "in-sample -- model trained on this period", fontsize=8, color="#555555",
                ha="center", va="top")
    ax.stackplot(timeline["date"], timeline["on_track"], timeline["at_risk"], timeline["overrunning"],
                 colors=[GREEN, AMBER, RED], alpha=0.6, labels=["On track", "At risk", "Overrunning"])
    ax.axvline(T, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(TEST_PERIOD_START, color="#2a78d6", linewidth=1.2, linestyle=":")
    ax.set_ylabel("Open issues", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)
    st.pyplot(fig, clear_figure=True)

    # --- Event feed (moved below the chart, Step 8) ---
    st.subheader("Event feed")
    events = []
    risk_cur_all = cls_cur["risk"]
    risk_prev_all = cls_prev["risk"]

    newly_created = mc_issues[(mc_issues["Creation_Date"] > T_prev) & (mc_issues["Creation_Date"] <= T)]
    for _, iss in newly_created.iterrows():
        r = risk_cur_all.get(iss["ID"], iss["static_risk"])
        if r >= RISK_THRESHOLD:
            events.append((iss["Creation_Date"], f"🆕 Issue {fmt_int(iss['ID'])} created -- initial risk {r:.0%}"))

    resolved_between = mc_issues[(mc_issues["Resolution_Date"] > T_prev) & (mc_issues["Resolution_Date"] <= T)
                                  & (~mc_issues["ID"].isin(mc_closed_ids))]
    for _, iss in resolved_between.iterrows():
        r = risk_cur_all.get(iss["ID"], iss["static_risk"])
        was_delayed = bool(iss["delayed_v2"])
        predicted_delay = r >= RISK_THRESHOLD
        correct = predicted_delay == was_delayed
        outcome = "delayed" if was_delayed else "on time"
        mark = "✅ correct call" if correct else "❌ missed call"
        icon = "🔴" if was_delayed else "✅"
        events.append((iss["Resolution_Date"],
                       f"{icon} Issue {fmt_int(iss['ID'])} resolved -- {outcome}, predicted {r:.0%} ({mark})"))

    common_open = open_prev_ids & open_cur_ids
    if common_open:
        prev_bands = risk_prev_all.loc[list(common_open)].apply(dd.risk_band)
        cur_bands = risk_cur_all.loc[list(common_open)].apply(dd.risk_band)
        changed = prev_bands[prev_bands != cur_bands]
        rank = {"low": 0, "medium": 1, "high": 2}
        for iid in changed.index:
            old_r, new_r = risk_prev_all.loc[iid], risk_cur_all.loc[iid]
            up = rank[cur_bands[iid]] > rank[prev_bands[iid]]
            direction = "⚠️" if up else "⬇️"
            verb = "crossed into higher risk" if up else "dropped to lower risk"
            events.append((T, f"{direction} Issue {fmt_int(iid)} {verb} ({old_r:.0%} → {new_r:.0%})"))

    events.sort(key=lambda e: e[0], reverse=True)
    if events:
        for _, msg in events[:25]:
            st.write(msg)
        if len(events) > 25:
            st.caption(f"... and {len(events) - 25} more significant events this step.")
    else:
        st.caption("No significant events this step.")

    # --- Blocker cascade among overrunning issues ---
    st.subheader("⛓️ Blocker cascade -- overrunning issues holding up other work")
    overrunning_ids = [i for i in open_cur_ids if cls_cur.loc[i, "status"] == "overrunning"]
    cascade_rows = []
    for oid in overrunning_ids:
        blocked = [b for b in blocking_graph.get(oid, []) if b in open_cur_ids]
        if blocked:
            cascade_rows.append((oid, blocked))
    cascade_rows.sort(key=lambda r: -len(r[1]))

    if cascade_rows:
        overrun_lookup = cls_cur["overrun_multiple"]
        for oid, blocked in cascade_rows[:5]:
            mult = overrun_lookup.get(oid, float("nan"))
            st.markdown(
                f"🔴 **Issue {fmt_int(oid)}** -- *{title_lookup.get(oid, '')[:70]}* "
                f"({mult:.1f}x expected duration) blocks **{len(blocked)}** open issue(s):"
            )
            for b in blocked:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ Issue {fmt_int(b)} -- *{title_lookup.get(b, '')[:60]}*",
                            unsafe_allow_html=True)
    else:
        st.caption("No overrunning issue currently blocks another open issue in this window.")

    # --- Open issues table, filtered, with transparency columns ---
    st.subheader("Open issues, sorted by risk")
    table_ids = list(open_cur_ids)
    active_filter = st.session_state.mc_table_filter
    if active_filter == "past_target":
        table_ids = [i for i in table_ids if vs_target_cur.get(i) == "past_target"]
    elif active_filter:
        table_ids = [i for i in table_ids if cls_cur.loc[i, "status"] == active_filter]

    if table_ids:
        open_df = mc_issues[mc_issues["ID"].isin(table_ids)].drop(columns=["expected_duration_minutes"]).merge(
            cls_cur.reset_index()[["ID", "risk", "source", "status", "overrun_multiple",
                                    "elapsed_minutes", "expected_duration_minutes"]],
            on="ID", how="left",
        ).sort_values("risk", ascending=False)
        open_df["vs_target"] = open_df["ID"].map(vs_target_cur)
        open_df["Open for (days)"] = open_df["elapsed_minutes"] / 1440.0
        open_df["Expected (days)"] = open_df["expected_duration_minutes"] / 1440.0
        open_df["Target (days)"] = open_df["Type_Normalized"].astype(str).map(mc_targets)

        display_mc = open_df[["ID", "Title", "Type", "Priority", "Assignee_ID", "status", "vs_target",
                               "risk", "source", "overrun_multiple", "Open for (days)",
                               "Expected (days)", "Target (days)"]].rename(
            columns={"risk": "predicted_risk", "status": "vs_history"}
        ).reset_index(drop=True)
        display_mc["ID"] = display_mc["ID"].map(fmt_int)
        display_mc["Assignee_ID"] = display_mc["Assignee_ID"].map(fmt_int)
        display_mc["overrun_multiple"] = display_mc["overrun_multiple"].map(
            lambda v: f"{v:.1f}x" if pd.notna(v) else ""
        )
        display_mc["Open for (days)"] = display_mc["Open for (days)"].map(lambda v: fmt_num(v, 1))
        display_mc["Expected (days)"] = display_mc["Expected (days)"].map(
            lambda v: fmt_num(v, 1) if pd.notna(v) else "insufficient history"
        )
        display_mc["Target (days)"] = display_mc["Target (days)"].map(
            lambda v: fmt_num(v, 1) if pd.notna(v) else "—"
        )
        styled_mc = display_mc.style.map(
            lambda v: f"background-color:{dd.risk_color(v)}33;color:{dd.risk_color(v)};font-weight:600;",
            subset=["predicted_risk"],
        ).format({"predicted_risk": "{:.0%}"})
        st.dataframe(styled_mc, width="stretch", height=350)
    else:
        st.caption("No open issues match the current filter.")

    # --- Selected-issue panel: two-signal detail, divergence, close action ---
    st.subheader("Selected issue")
    if table_ids:
        sorted_table_ids = sorted(table_ids, key=lambda i: -cls_cur.loc[i, "risk"])
        sel_labels = {
            i: f"{fmt_int(i)} - {title_lookup.get(i, '')[:50]} (risk={cls_cur.loc[i, 'risk']:.0%})"
            for i in sorted_table_ids
        }
        sel_id = st.selectbox(
            "Select an issue", sorted_table_ids, format_func=lambda i: sel_labels[i],
            key=f"mc_issue_select_{project_id}_{active_filter}",
        )
        vs_hist = cls_cur.loc[sel_id, "status"]
        vs_tgt = vs_target_cur.get(sel_id, "no_target_set")
        elapsed_d = cls_cur.loc[sel_id, "elapsed_minutes"] / 1440.0
        expected_minutes = cls_cur.loc[sel_id, "expected_duration_minutes"]
        expected_d = expected_minutes / 1440.0 if pd.notna(expected_minutes) else None
        type_n = str(type_lookup.get(sel_id))
        target_d = mc_targets.get(type_n)

        hist_label = {"overrunning": "Overrunning", "at_risk": "At risk", "on_track": "On track"}[vs_hist]
        tgt_label = {"past_target": "Past target", "within_target": "Within target",
                     "no_target_set": "No target set"}[vs_tgt]
        bcol1, bcol2 = st.columns(2)
        bcol1.markdown(f"**vs. history:** {hist_label}")
        bcol2.markdown(f"**vs. target:** {tgt_label}")

        flagged_hist = vs_hist in ("at_risk", "overrunning")
        flagged_tgt = vs_tgt == "past_target"
        if vs_tgt != "no_target_set" and flagged_hist != flagged_tgt:
            exp_txt = f"a typical {expected_d:.0f}" if expected_d is not None else "an unknown"
            if flagged_tgt and not flagged_hist:
                st.info(
                    f"On track by historical pace (day {elapsed_d:.0f} of {exp_txt}), but past your "
                    f"{target_d:.0f}-day target."
                )
            else:
                st.info(
                    f"Flagged by historical pace (day {elapsed_d:.0f} of {exp_txt}), but still within "
                    f"your {target_d:.0f}-day target."
                )

        st.write(
            f"**Open for:** {elapsed_d:.1f} days &nbsp;·&nbsp; **Expected:** "
            f"{f'{expected_d:.1f} days' if expected_d is not None else 'insufficient history'} "
            f"&nbsp;·&nbsp; **Target:** {f'{target_d:.1f} days' if target_d is not None else '—'}",
            unsafe_allow_html=True,
        )

        if st.button("🔒 Close issue", key=f"mc_close_{sel_id}"):
            st.session_state.mc_closed_ids.add(int(sel_id))
            st.rerun()
    else:
        st.caption("No issue available to select under the current filter.")

    # --- Closed-issues panel (Step 6/8) ---
    st.subheader("Closed issues (this session)")
    st.caption("Session-only -- closing or reopening here never modifies the underlying data files.")
    if mc_closed_ids:
        closed_df = mc_issues[mc_issues["ID"].isin(mc_closed_ids)][["ID", "Title", "Type"]].copy()
        closed_df["ID"] = closed_df["ID"].map(fmt_int)
        st.dataframe(closed_df, width="stretch")
        reopen_options = sorted(mc_closed_ids)
        reopen_labels = {i: f"{fmt_int(i)} - {title_lookup.get(i, '')[:50]}" for i in reopen_options}
        reopen_id = st.selectbox(
            "Reopen an issue", reopen_options, format_func=lambda i: reopen_labels[i],
            key=f"mc_reopen_select_{project_id}",
        )
        if st.button("Reopen", key=f"mc_reopen_btn_{reopen_id}"):
            st.session_state.mc_closed_ids.discard(reopen_id)
            st.rerun()
    else:
        st.caption("No issues closed this session.")

    if playing and T < project_end:
        time.sleep(0.7)
        st.session_state.mc_cursor = min(project_end, T + step)
        st.rerun()
    elif playing and T >= project_end:
        st.session_state.mc_playing = False
# ---------------------------------------------------------------------
# Tab 2: Issue Risk & Explanation (+ folded-in Reassignment, Step 6)
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
            row["ID"]: f"{fmt_int(row['ID'])} - {row['Title'][:60]} (risk={row['dynamic_risk']:.0%})"
            for _, row in matches.iterrows()
        }
        selected_id = st.selectbox(
            "Select issue", options, format_func=lambda i: labels[i], index=0,
            key=f"tab2_select_{search}",
        )

        row = portfolio[portfolio["ID"] == selected_id].iloc[0]
        proj_avg = portfolio.loc[portfolio["Project_ID"] == row["Project_ID"], "dynamic_risk"].mean()
        below_m3 = row["checkpoint_index"] < CROSSOVER_CHECKPOINT

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted risk (dynamic)", f"{row['dynamic_risk']:.0%}",
                    f"{row['dynamic_risk'] - proj_avg:+.0%} vs. project avg")
        col2.metric("Predicted risk (static)", f"{row['static_risk']:.0%}")
        col3.metric("Checkpoint reached", f"M{fmt_int(row['checkpoint_index'])}")
        st.markdown(risk_badge(row["dynamic_risk"]), unsafe_allow_html=True)
        if below_m3:
            st.caption(
                "⚠️ This issue is below the M3 reliability threshold -- the static (creation-time) "
                "estimate is the more trustworthy number here."
            )

        st.write(f"**Type:** {row['Type']}  |  **Priority:** {row['Priority']}  |  "
                 f"**Assignee:** {fmt_int(row['Assignee_ID'])}  |  **Project:** {fmt_int(row['Project_ID'])}")

        # --- Risk trajectory chart with cohort band ---
        st.subheader("Risk trajectory across checkpoints")
        traj = dd.build_issue_trajectory(selected_id, row, all_checkpoints, dynamic_model)
        cohort = ex.cohort_trajectories(all_checkpoints, feat_full, row["Project_ID"], row["Type_Normalized"], dynamic_model)

        if traj.empty:
            st.info("No checkpoint history available for this issue (resolved before M1).")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.axhspan(0, 0.4, color=GREEN, alpha=0.06)
            ax.axhspan(0.4, 0.7, color=AMBER, alpha=0.06)
            ax.axhspan(0.7, 1.0, color=RED, alpha=0.06)

            ax.fill_between(cohort["checkpoint_index"], cohort["q25"], cohort["q75"],
                             color="#4C78A8", alpha=0.15, label=f"Cohort IQR (Project {fmt_int(row['Project_ID'])}, {row['Type']})")
            ax.plot(cohort["checkpoint_index"], cohort["median"], color="#4C78A8", linewidth=1.5,
                    linestyle="--", alpha=0.7, label="Cohort median")

            max_ck = traj["checkpoint_index"].max()
            if max_ck >= 1:
                ax.axvspan(1, min(2, max_ck), color="grey", alpha=0.15)
                ax.text(1.5, 0.03, "insufficient history --\nstatic preferred", fontsize=11,
                        ha="center", va="bottom", color="#555555")

            ax.plot(traj["checkpoint_index"], traj["risk"], color="#eb6834", linewidth=2.4,
                    marker="o", markersize=7, label="This issue (dynamic)", zorder=5)
            ax.axhline(row["static_risk"], color="#2a78d6", linestyle="--", linewidth=2,
                       label=f"Static prediction: {row['static_risk']:.0%}")
            ax.axhline(RISK_THRESHOLD, color="black", linestyle=":", linewidth=1.3,
                       label=f"Decision threshold ({RISK_THRESHOLD})")
            ax.axvline(CROSSOVER_CHECKPOINT, color="#54A24B", linestyle="-", linewidth=1.6, alpha=0.8)
            ax.annotate("dynamic more reliable\nfrom here", xy=(CROSSOVER_CHECKPOINT, 0.95),
                        fontsize=11, color="#2c6b3f", ha="left", va="top")

            actual_delayed = int(row["delayed_v2"])
            outcome_text = "DELAYED" if actual_delayed else "ON TIME"
            outcome_color = RED if actual_delayed else GREEN
            ax.scatter([max_ck], [traj.iloc[-1]["risk"]], s=140, facecolors="none",
                       edgecolors=outcome_color, linewidths=2.5, zorder=6)
            ax.annotate(f"Actual outcome: {outcome_text}", xy=(max_ck, traj.iloc[-1]["risk"]),
                        xytext=(-10, 20 if not actual_delayed else -30), textcoords="offset points",
                        fontsize=11, fontweight="bold", color=outcome_color, ha="right")

            ax.set_xlabel("Checkpoint index (M1-M10)", fontsize=12)
            ax.set_ylabel("Predicted delay risk", fontsize=12)
            ax.set_xticks(range(1, 11))
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=9, loc="lower right")
            ax.set_title(f"Issue {fmt_int(selected_id)}: risk trajectory vs. cohort", fontsize=13)
            st.pyplot(fig, clear_figure=True)

            last_ck = int(traj.iloc[-1]["checkpoint_index"])
            last_risk = traj.iloc[-1]["risk"]
            outcome_phrase = "It was ultimately delayed." if actual_delayed else "It ultimately finished on time."
            st.markdown(
                f"*At creation this issue was predicted **{row['static_risk']:.0%}** likely to be delayed "
                f"(static). By checkpoint **M{last_ck}** the dynamic model's estimate had moved to "
                f"**{last_risk:.0%}**. {outcome_phrase}*"
            )

        # --- Grouped SHAP + NL explanation ---
        st.subheader("Why the model predicts this")
        tree_explainer = ra._get_explainer(dynamic_model)
        row_features = row[dd.DYN_FEATURE_COLS].to_frame().T
        for col in dd.DYN_FEATURE_COLS:
            if col in dd.CATEGORICAL_COLS + ["pattern_label"]:
                row_features[col] = row_features[col].astype(portfolio[col].dtype)
            else:
                row_features[col] = row_features[col].astype(float)
        shap_values = tree_explainer(row_features[dd.DYN_FEATURE_COLS])
        contrib = pd.Series(shap_values.values[0], index=dd.DYN_FEATURE_COLS)
        grouped = ex.grouped_shap(contrib)

        fig_g, ax_g = plt.subplots(figsize=(8, 3.5))
        order = grouped.index[::-1]
        colors_g = ["#c0392b" if v > 0 else "#1b8a5a" for v in grouped.loc[order]]
        ax_g.barh(order, grouped.loc[order], color=colors_g)
        ax_g.axvline(0, color="grey", linewidth=1)
        ax_g.set_xlabel("SHAP contribution to risk", fontsize=11)
        ax_g.tick_params(labelsize=10)
        st.pyplot(fig_g, clear_figure=True)

        status_row = checkpoint_status[
            (checkpoint_status["Issue_ID"] == selected_id)
            & (checkpoint_status["checkpoint_index"] == row["checkpoint_index"])
        ]
        current_status = status_row["current_status"].iloc[0] if len(status_row) else None
        last_change = status_row["last_status_change_time"].iloc[0] if len(status_row) else None
        checkpoint_time = row["Creation_Date"] + pd.Timedelta(minutes=row["elapsed_minutes"])
        avg_delay_rate = ex.project_avg_delay_rate(feat_full, row["Project_ID"])

        explanation = ex.generate_explanation(
            row, contrib, ref_stats, avg_delay_rate,
            current_status=current_status, last_status_change_time=last_change, checkpoint_time=checkpoint_time,
            predicted_risk=row["dynamic_risk"],
        )
        st.markdown(explanation["summary"])
        if explanation["raising"]:
            st.markdown("⬆️ **Raising the risk estimate**")
            for b in explanation["raising"]:
                st.markdown(f"- {b}")
        if explanation["lowering"]:
            st.markdown("⬇️ **Lowering the risk estimate**")
            for b in explanation["lowering"]:
                st.markdown(f"- {b}")

        # --- Peer comparison ---
        st.subheader("Peer comparison")
        peer_times = ex.peer_resolution_times(feat_full, row["Project_ID"], row["Type_Normalized"], row["Priority_Normalized"])
        if len(peer_times) >= 5:
            elapsed_days = row["elapsed_minutes"] / 1440
            fig_p, ax_p = plt.subplots(figsize=(8, 3))
            ax_p.hist(peer_times, bins=30, color="#4C78A8", alpha=0.7, edgecolor="white")
            ax_p.axvline(elapsed_days, color="#eb6834", linewidth=2.5,
                         label=f"This issue: day {elapsed_days:.1f}")
            pct = (peer_times < elapsed_days).mean()
            ax_p.set_xlabel("Resolution time (days)", fontsize=11)
            ax_p.set_ylabel("Peer issue count", fontsize=11)
            ax_p.legend(fontsize=9)
            st.pyplot(fig_p, clear_figure=True)
            st.caption(
                f"This issue has already taken longer than **{pct:.0%}** of {len(peer_times)} peer issues "
                f"(same project, type, and priority) took to resolve in total."
            )
        else:
            st.caption("Not enough peer issues (same project/type/priority) for a meaningful comparison.")

        # --- What-if controls (Step 7 fix: per-issue widget keys) ---
        st.subheader("What if...")
        st.caption(
            "Note: this model uses only 11 boosting rounds (tuned via validation, not chosen for "
            "smoothness), so for some issues a moderate slider move produces little to no change -- "
            "that reflects the model's actual (low) sensitivity to that feature for this specific "
            "issue, not a bug. Try larger moves or a different issue to see clearer swings."
        )
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            what_if_concurrent = st.slider(
                "Assignee concurrent open issues", 0, 60, int(row["assignee_concurrent_open"]),
                key=f"wi_concurrent_{selected_id}",
            )
        with wcol2:
            default_delay_rate = float(row["assignee_prior_delay_rate"]) if pd.notna(row["assignee_prior_delay_rate"]) else 0.5
            what_if_delay_rate = st.slider(
                "Assignee historical delay rate", 0.0, 1.0, default_delay_rate,
                key=f"wi_delay_rate_{selected_id}",
            )
        with wcol3:
            priority_options = sorted(portfolio["Priority_Normalized"].cat.categories.tolist())
            what_if_priority = st.selectbox(
                "Priority", priority_options, index=priority_options.index(row["Priority_Normalized"]),
                key=f"wi_priority_{selected_id}",
            )

        what_if_row = row_features.copy()
        what_if_row["assignee_concurrent_open"] = float(what_if_concurrent)
        what_if_row["assignee_prior_delay_rate"] = float(what_if_delay_rate)
        what_if_row["Priority_Normalized"] = pd.Categorical([what_if_priority], categories=priority_options)
        what_if_risk = float(dynamic_model.predict_proba(what_if_row[dd.DYN_FEATURE_COLS])[:, 1][0])

        wcol_result1, wcol_result2 = st.columns(2)
        wcol_result1.metric("Original predicted risk", f"{row['dynamic_risk']:.0%}")
        wcol_result2.metric("What-if predicted risk", f"{what_if_risk:.0%}",
                            f"{what_if_risk - row['dynamic_risk']:+.0%}")

        # --- Step 6: Reassignment folded into the issue view ---
        st.subheader("Reassignment")
        if st.button("Check reassignment", key=f"check_reassignment_{selected_id}"):
            if below_m3:
                reassign_model, reassign_cols, model_label = (
                    static_model, dd.STATIC_FEATURE_COLS_V2,
                    "creation-time estimate -- this issue has not yet reached the M3 reliability threshold",
                )
            else:
                reassign_model, reassign_cols, model_label = dynamic_model, dd.DYN_FEATURE_COLS, "dynamic"

            result = ra.suggest_reassignment(
                selected_id, portfolio, feat_full, dynamic_model, top_n=5,
                model=reassign_model, feature_cols=reassign_cols, model_label=model_label,
            )
            if below_m3:
                st.caption(f"ℹ️ Using the **static** model for this suggestion ({model_label}).")

            st.write(f"**Top risk drivers:** " + ", ".join(f"{f} ({v:+.3f})" for f, v in result["top_drivers"]))

            if result["verdict"] == "reassignment_may_help":
                st.success(f"### ✅ Reassignment may help\n{result['explanation']}")
            else:
                st.warning(f"### ⚠️ Reassignment unlikely to help\n{result['explanation']}")

            if result["candidates"]:
                cand_df = pd.DataFrame(result["candidates"])
                bars = pd.concat([
                    pd.DataFrame([{"assignee_id": f"{fmt_int(result['current_assignee_id'])} (current)",
                                    "predicted_risk": result["current_risk"], "is_current": True}]),
                    cand_df.assign(
                        assignee_id=cand_df["assignee_id"].apply(fmt_int), is_current=False
                    )[["assignee_id", "predicted_risk", "is_current"]],
                ]).sort_values("predicted_risk", ascending=True)

                fig_r, ax_r = plt.subplots(figsize=(8, max(2.5, 0.5 * len(bars))))
                colors_r = ["#0b0b0b" if is_cur else dd.risk_color(r)
                            for r, is_cur in zip(bars["predicted_risk"], bars["is_current"])]
                ax_r.barh(bars["assignee_id"], bars["predicted_risk"], color=colors_r)
                ax_r.axvline(RISK_THRESHOLD, color="grey", linestyle="--", linewidth=1)
                ax_r.set_xlabel("Predicted risk", fontsize=11)
                ax_r.set_xlim(0, 1)
                ax_r.tick_params(labelsize=10)
                for spine in ["top", "right"]:
                    ax_r.spines[spine].set_visible(False)
                st.pyplot(fig_r, clear_figure=True)
                st.caption("Black bar = current assignee. Bar colour follows the same risk bands as elsewhere.")

                with st.expander("Show as table"):
                    cand_display = cand_df.copy()
                    cand_display["assignee_id"] = cand_display["assignee_id"].map(fmt_int)
                    cand_display["predicted_risk"] = cand_display["predicted_risk"].map(lambda v: fmt_num(v, 2))
                    cand_display["risk_delta"] = cand_display["risk_delta"].map(lambda v: fmt_num(v, 2))
                    st.dataframe(cand_display, width="stretch")

            if result.get("candidate_pool_threshold"):
                st.caption(f"Candidate pool: developers with >= {result['candidate_pool_threshold']} "
                           f"prior resolved issues in this project.")

# ---------------------------------------------------------------------
# Tab 3: Team (workload table + heatmap)
# ---------------------------------------------------------------------
with tab3:
    st.header("Team")
    projects = sorted(portfolio["Project_ID"].unique())
    project_id_3 = st.selectbox("Project", projects, key="tab3_project")

    proj_issues = feat_full[feat_full["Project_ID"] == project_id_3].dropna(subset=["Assignee_ID"])
    proj_issues = proj_issues.sort_values("Creation_Date")

    st.subheader("Current workload snapshot")
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
    workload_df["Assignee_ID"] = workload_df["Assignee_ID"].map(fmt_int)
    workload_df["concurrent_open_issues"] = workload_df["concurrent_open_issues"].map(fmt_int)
    workload_df["prior_resolved_count"] = workload_df["prior_resolved_count"].map(fmt_int)
    styled_workload = workload_df.style.map(
        lambda v: f"background-color: {dd.risk_color(v)}33; color: {dd.risk_color(v)}; font-weight: 600;",
        subset=["historical_delay_rate"],
    ).format({"historical_delay_rate": "{:.0%}"})
    st.dataframe(styled_workload, width="stretch")

    st.subheader("Team heatmap: average risk by developer and month")
    st.caption(
        "Simplification, stated plainly: this buckets issues by the MONTH THEY WERE CREATED (not a "
        "true day-by-day open-issues average, which Mission Control provides for one project/window) "
        "-- cell = average predicted risk (at each issue's last observed checkpoint) of issues that "
        "developer created in that month."
    )
    proj_scored = portfolio[portfolio["Project_ID"] == project_id_3].copy()
    if len(proj_scored):
        proj_scored["month"] = proj_scored["Creation_Date"].dt.to_period("M").astype(str)
        pivot = proj_scored.pivot_table(index="Assignee_ID", columns="month", values="dynamic_risk", aggfunc="mean")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            fig_h, ax_h = plt.subplots(figsize=(min(14, 1 + 0.6 * pivot.shape[1]), max(2.5, 0.35 * pivot.shape[0])))
            im = ax_h.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
            ax_h.set_xticks(range(pivot.shape[1]))
            ax_h.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
            ax_h.set_yticks(range(pivot.shape[0]))
            ax_h.set_yticklabels([fmt_int(a) for a in pivot.index], fontsize=8)
            fig_h.colorbar(im, ax=ax_h, label="Average predicted risk", shrink=0.8)
            st.pyplot(fig_h, clear_figure=True)
        else:
            st.caption("Not enough data for a heatmap in this project's test-portfolio slice.")
    else:
        st.caption("No portfolio issues for this project.")
