"""
Delay Radar v5 -- Streamlit app for the improved models (Tasks J/K).

Run from the capstone folder:   streamlit run app/app_v5.py
The original app/app.py (v3 models) is untouched.

Two modes, chosen by the project manager in the sidebar:

  FLOW MODE   -- "will this issue take longer than this project normally takes?"
                 Pick a project and slide the date; every open issue is scored with features
                 computed from events before that date, with plain-English reasons.
                 When the date moves, newly opened issues glow at the top of the list for 5 s
                 and closed ones fade out (app/live_list.py). Issue detail has a reassignment
                 check; Team workload shows who holds the late work.
  SCRUM MODE  -- the PM runs sprints, so the deadline is the sprint end.
                 Sprint board (KPIs, burndown, spill risk per committed issue),
                 issue detail, sprint history (predicted vs actual spills).

This is a historical replay on TAWOS data, not a live Jira feed.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import importlib
import app_v5_backend
import recommender
import scrum_backend as scrum_mod

st.set_page_config(page_title="Delay Radar v5", layout="wide", page_icon="📡")
RED, BLUE, GREY = "#d1495b", "#2a78d6", "#898781"
RECO_NOTE = ("Each row re-scores this issue with one thing changed (e.g. status moved to review): the model's "
             "estimate of how its risk would move, not a guaranteed causal effect. Checked against history "
             "(test period, issues still open a week later, matched on risk): every action points the same way "
             "in real outcomes, but status moves (start work / review) are overstated about 2x, reassignment is "
             "about right, and commenting is understated.")


# Backend code lives outside app/, so Streamlit does not hot-reload it. The cache key is the
# backend files' modification times: edit any of them and the next rerun reloads the modules and
# rebuilds the backend (~1 minute) instead of serving an old object that lacks new methods.
BACKEND_FILES = ("app_v5_backend.py", "recommender.py", "scrum_backend.py", "feature_engine.py")


def code_version():
    return tuple(os.path.getmtime(f) for f in BACKEND_FILES if os.path.exists(f))


@st.cache_resource(max_entries=1, show_spinner="Loading change history and models (first run takes ~1 minute)...")
def backend(version):
    for m in (recommender, app_v5_backend):
        importlib.reload(m)
    return app_v5_backend.Backend()


@st.cache_resource(max_entries=1, show_spinner="Preparing Scrum mode (sprint memberships)...")
def scrum_backend(_b, version):
    importlib.reload(scrum_mod)
    return scrum_mod.ScrumBackend(_b.fe, _b.titles)


VERSION = code_version()
b = backend(VERSION)

with st.sidebar:
    st.title("📡 Delay Radar v5")
    mode = st.radio("Mode", ["Flow mode", "Scrum mode"], horizontal=True,
                    help="Flow: deadline = the project's typical duration. "
                         "Scrum: deadline = the end of the sprint the issue is committed to.")


# =====================================================================================
# FLOW MODE
# =====================================================================================
@st.cache_data(max_entries=64, show_spinner=False)
def open_issues_at(pid, date, version=None):
    return b.open_issues(pid, pd.Timestamp(date))


STEPS = {"1 day": pd.DateOffset(days=1), "1 week": pd.DateOffset(weeks=1), "1 month": pd.DateOffset(months=1)}


def _shift_date(key, direction, first, last):
    d = pd.Timestamp(st.session_state[key]) + direction * STEPS[st.session_state.get("flow_step", "1 week")]
    st.session_state[key] = min(max(d.date(), first), last)


def _is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


# ---------------------------------------------------------------- manager targets (Configuration tab)
MIN_HISTORY = 5     # fewer resolved issues of a type than this: no history median / reality check


def targets_for(pid):
    """{issue type: target days} the manager set for this project. Kept outside widget state so it
    survives switching projects and tabs; never fed to the model."""
    return st.session_state.setdefault("targets", {}).setdefault(pid, {})


def with_targets(d, targets):
    if d is None or d.empty:
        return d if d is None else d.assign(target=pd.Series(dtype=float), past_target=pd.Series(dtype=bool))
    t = d["type"].map(targets).astype(float)
    return d.assign(target=t, past_target=(d["age_days"] > t).fillna(False).astype(bool))


@st.cache_data(max_entries=32, show_spinner=False)
def type_history_at(pid, date, version=None):
    return b.type_history(pid, pd.Timestamp(date))


def _toggle_target(pid, t, default):
    store = targets_for(pid)
    if st.session_state.get(f"tgt_on_{pid}_{t}"):
        store[t] = float(st.session_state.get(f"tgt_val_{pid}_{t}", default))
    else:
        store.pop(t, None)


def _set_target(pid, t):
    targets_for(pid)[t] = float(st.session_state[f"tgt_val_{pid}_{t}"])


def _clear_targets(pid):
    targets_for(pid).clear()
    for k in [k for k in st.session_state.keys() if str(k).startswith((f"tgt_on_{pid}_", f"tgt_val_{pid}_"))]:
        del st.session_state[k]


def configuration(pid, date):
    st.subheader("Targets for this project")
    med = b.project_median_days(pid)
    st.caption(f"Set how many days you want each issue type to take. Targets are **your standard**: they add a "
               f"'past target' check to the list, a KPI and a filter, but they never change the model's risk, which "
               f"is learned from history. 'Already overdue' always means older than this project's median age "
               f"(**{med:.0f} days**).")
    hist = type_history_at(pid, date, VERSION)
    open_types = set(open_issues_at(pid, date, VERSION).get("type", pd.Series(dtype=str)))
    g = hist.groupby("type")["days"].agg(median="median", n="size") if len(hist) else pd.DataFrame(columns=["median", "n"])
    types = sorted((set(g.index) | open_types) - {"nan", "None", ""},
                   key=lambda t: -(g["n"].get(t, 0) if len(g) else 0))
    store = targets_for(pid)

    h = st.columns([2, 3, 2, 2, 5])
    for col, lab in zip(h, ["**Type**", "**History up to this date**", "**Set a target**", "**Target (days)**",
                             "**Reality check**"]):
        col.markdown(lab)
    deltas = []
    for t in types:
        n = int(g["n"].get(t, 0)) if len(g) else 0
        m = float(g["median"].get(t, np.nan)) if len(g) else np.nan
        enough = n >= MIN_HISTORY
        default = round(m, 1) if enough else round(med, 1)
        c = st.columns([2, 3, 2, 2, 5])
        c[0].write(t)
        c[1].write(f"median {m:.1f} d · {n} closed" if enough else f"not enough history ({n} closed)")
        on = c[2].checkbox("set", value=t in store, key=f"tgt_on_{pid}_{t}", label_visibility="collapsed",
                           on_change=_toggle_target, args=(pid, t, default))
        if on:
            c[3].number_input("days", min_value=0.5, value=float(store.get(t, default)), step=0.5,
                              key=f"tgt_val_{pid}_{t}", label_visibility="collapsed",
                              on_change=_set_target, args=(pid, t))
            tgt = store.get(t, default)
            if enough:
                missed = float((hist.loc[hist["type"] == t, "days"] > tgt).mean())
                c[4].caption(f"{missed:.0%} of the {n} past {t}s here would have missed {tgt:g} days "
                             f"(history median {m:.1f}).")
                deltas.append((m - tgt) / m if m > 0 else 0)
            else:
                c[4].caption("Too little history to check this target.")
        else:
            c[3].caption("—")
            c[4].caption("No target: only the model's risk and the median-age line apply.")

    if deltas:
        avg = np.mean(deltas) * 100
        st.markdown(f"**Your targets are on average {abs(avg):.0f}% {'tighter' if avg >= 0 else 'looser'} than this "
                    f"project's history up to {date:%d %b %Y}.**")
    if store:
        st.button("Clear all targets", on_click=_clear_targets, args=(pid,))
    st.caption("Targets are kept for this browser session only; nothing is written to the data files.")


def vs_target(row):
    """Two separate signals for one issue: the model/history view and the manager's own target."""
    hist_lbl = {"already overdue": "already overdue", "likely late": "likely late (model)",
                "on track": "on track (model)"}[row["flag"]]
    tgt = row.get("target")
    c1, c2 = st.columns(2)
    c1.markdown(f"**vs history:** {hist_lbl} — day {row['age_days']:.0f} of a typical {row['median_days']:.0f}")
    if tgt is None or pd.isna(tgt):
        c2.markdown(f"**vs your target:** no target set for {row['type']} (Configuration tab)")
        return
    past = row["age_days"] > tgt
    c2.markdown(f"**vs your target:** {'past' if past else 'within'} your {tgt:g}-day target for {row['type']}")
    flagged = row["flag"] in ("already overdue", "likely late")
    if past and not flagged:
        st.info(f"On track by the model and the project's history, but already past your {tgt:g}-day target.")
    elif flagged and not past:
        st.info(f"Flagged by the model/history, but still within your {tgt:g}-day target "
                f"({tgt - row['age_days']:.0f} days left).")


# ---------------------------------------------------------------- parked issues (manager's "not a priority" list)
def parked_for(pid):
    """{issue id: {"reason", "on"}} the manager parked in this project. Session only; never touches the data."""
    return st.session_state.setdefault("parked", {}).setdefault(pid, {})


def _park_ids(pid, ids, date, reason_key=None):
    reason = (st.session_state.get(reason_key, "") or "").strip() if reason_key else ""
    store = parked_for(pid)
    for i in ids:
        store[int(i)] = {"reason": reason, "on": date}
    if reason_key:
        st.session_state[reason_key] = ""


def _park_picked(pid, date):
    _park_ids(pid, st.session_state.get(f"park_pick_{pid}", []), date, f"park_reason_{pid}")
    st.session_state[f"park_pick_{pid}"] = []


def _unpark(pid, ids_in_table, table_key, everything=False):
    store = parked_for(pid)
    if everything:
        store.clear()
        return
    sel = st.session_state.get(table_key)
    rows = list(getattr(getattr(sel, "selection", None), "rows", []) or (sel or {}).get("selection", {}).get("rows", []))
    for r in rows:
        if 0 <= r < len(ids_in_table):
            store.pop(ids_in_table[r], None)


def parked_section(pid, date, full_all, active_all):
    store = parked_for(pid)
    st.markdown(f"#### 🅿️ Parked — not a priority ({len(store)})")
    st.caption("Issues you've decided not to chase. They stay open, but they're left out of the list and counts "
               "above, Issue detail and Team workload. Kept for this browser session only.")
    c1, c2, c3 = st.columns([4, 3, 1], vertical_alignment="bottom")
    opts = active_all["Issue_ID"].astype(int).tolist() if len(active_all) else []
    titles = active_all.set_index("Issue_ID")["Title"] if len(active_all) else pd.Series(dtype=str)
    c1.multiselect("Park issues", opts, key=f"park_pick_{pid}", placeholder="Choose open issues to park",
                   format_func=lambda i: f"{i} — {str(titles.get(i, ''))[:60]}")
    c2.text_input("Reason (optional)", key=f"park_reason_{pid}", placeholder="e.g. nice-to-have, waiting on upstream")
    c3.button("Park", key=f"park_btn_{pid}", width="stretch", on_click=_park_picked, args=(pid, date),
              disabled=not st.session_state.get(f"park_pick_{pid}"))
    if not store:
        st.caption("Nothing parked yet. You can also park from Issue detail (🅿️ Park).")
        return
    ids = list(store)
    now = full_all.set_index("Issue_ID") if len(full_all) else pd.DataFrame()
    out = b.outcomes(ids, date).set_index("Issue_ID")
    rows = []
    for i in ids:
        if len(now) and i in now.index:
            r = now.loc[i]
            rows.append({"Issue": i, "Title": r["Title"], "now": "still open", "age (d)": r["age_days"],
                         "risk": r["risk"], "flag": r["flag"]})
        else:
            done = bool(out.loc[i, "resolved"]) if i in out.index else False
            rows.append({"Issue": i, "Title": str(b.titles.get(i, "")),
                         "now": f"closed {out.loc[i, 'resolved_at']:%d %b %Y}" if done else "not created yet on this date",
                         "age (d)": np.nan, "risk": np.nan, "flag": ""})
        rows[-1].update({"reason": store[i]["reason"] or "—", "parked on": f"{store[i]['on']:%d %b %Y}"})
    tkey = f"parked_tbl_{pid}"
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key=tkey, on_select="rerun",
                 selection_mode="multi-row", column_config={
                     "risk": st.column_config.ProgressColumn("risk", min_value=0.0, max_value=1.0, format="%.2f"),
                     "age (d)": st.column_config.NumberColumn("age (d)", format="%.0f"),
                     "Title": st.column_config.TextColumn("Title", width="large")})
    u1, u2, _ = st.columns([2, 2, 6])
    u1.button("Unpark selected", key=f"unpark_sel_{pid}", width="stretch", on_click=_unpark, args=(pid, ids, tkey))
    u2.button("Unpark all", key=f"unpark_all_{pid}", width="stretch", on_click=_unpark, args=(pid, ids, tkey, True))
    st.caption("Tick rows on the left of the table, then Unpark selected to put them back in the list.")


def flow_mode():
    projects = b.projects()
    projects = projects[projects["n"] >= 200].sort_values("n", ascending=False)
    with st.sidebar:
        pid = st.selectbox("Project", projects.index.tolist(),
                           format_func=lambda p: f"Project {p}  ({projects.loc[p, 'n']:,} issues)")
        first = pd.to_datetime(projects.loc[pid, "first"], unit="s").date()
        last = pd.to_datetime(projects.loc[pid, "last"], unit="s").date()
        skey = f"flow_date_{pid}"
        if skey not in st.session_state:
            st.session_state[skey] = (pd.Timestamp(last) - pd.Timedelta(days=180)).date()
        date = st.slider("As of date", min_value=first, max_value=last, key=skey,
                         step=pd.Timedelta(days=1).to_pytimedelta(), format="DD MMM YYYY")
        st.radio("Step", list(STEPS), index=1, horizontal=True, key="flow_step", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.button("◀ Back", on_click=_shift_date, args=(skey, -1, first, last), width="stretch")
        c2.button("Forward ▶", on_click=_shift_date, args=(skey, +1, first, last), width="stretch")
        st.caption("Everything is computed from events before this date. New issues glow green at the top "
                   "for 5 s; closed ones fade out.")
        show_truth = st.toggle("Show actual outcomes", value=False)
        show_overdue = st.toggle("Include already-overdue issues", value=False,
                                 help="Issues already older than the project's typical duration are late by "
                                      "definition; hidden by default so the list focuses on issues that can still be saved.")

    # ---- what changed since the previous date (the date before the last move) ----
    hkey = f"flow_hist_{pid}"
    hist = st.session_state.get(hkey)
    if hist is None:
        hist = {"cur": date, "prev": None}
        just_moved = False
    elif hist["cur"] != date:
        hist = {"cur": date, "prev": hist["cur"]}
        just_moved = True
    else:
        just_moved = False
    st.session_state[hkey] = hist
    prev_date = hist["prev"]

    targets = targets_for(pid)
    parked = parked_for(pid)
    unparked = (lambda d: d[~d["Issue_ID"].isin(list(parked))].reset_index(drop=True)
                if (d is not None and len(d) and parked) else d)
    full_all = with_targets(open_issues_at(pid, date, VERSION), targets)      # parked included (parked table)
    df_all = unparked(full_all)
    keep = (lambda d: d) if show_overdue else (lambda d: d[~d["overdue"]].reset_index(drop=True) if len(d) else d)
    df = keep(df_all)
    prev_all = unparked(with_targets(open_issues_at(pid, prev_date, VERSION), targets)) if prev_date is not None else None
    prev_df = keep(prev_all) if prev_all is not None else None

    tab_port, tab_issue, tab_team, tab_cfg, tab_card = st.tabs(
        ["Portfolio", "Issue detail", "Team workload", "⚙️ Configuration", "Model card"])

    with tab_port:
        if df_all.empty:
            st.info("No open issues on this date." + (" (Parked issues are below.)" if parked else ""))
        else:
            portfolio(pid, date, prev_date, df_all, df, prev_all, prev_df, just_moved, show_truth)
        parked_section(pid, date, full_all, df_all)

    with tab_issue:
        if df.empty:
            st.info("No open issues on this date.")
        else:
            issue_detail(pid, df, date)

    with tab_team:
        if df_all.empty:
            st.info("No open issues on this date.")
        else:
            team_workload(pid, date, df_all)

    with tab_cfg:
        configuration(pid, date)

    with tab_card:
        card("md files/task-J-overnight-summary.md")
        with st.expander("Further evidence (Task L): external companies, baselines, uncertainty, recommendations"):
            card("md files/task-L-results.md")


def _kpis(d_all, d):
    late = int((d["flag"] == "likely late").sum()) if len(d) else 0
    return {"open": len(d_all), "overdue": int(d_all["overdue"].sum()) if len(d_all) else 0, "late": late,
            "confident": int((d["confidence"] == "confident: late").sum()) if len(d) else 0,
            "age": float(d["age_days"].median()) if len(d) else float("nan"),
            "not_overdue": int((~d["overdue"]).sum()) if len(d) else 0,
            "past_target": int(d_all["past_target"].sum()) if len(d_all) else 0}


def portfolio(pid, date, prev_date, df_all, df, prev_all, prev_df, just_moved, show_truth):
    k = _kpis(df_all, df)
    pk = _kpis(prev_all, prev_df) if prev_all is not None else None
    dl = (lambda key: f"{k[key] - pk[key]:+d} vs {prev_date:%d %b}" if pk else None)
    dc = (lambda key, bad="inverse": "off" if not pk or k[key] == pk[key] else bad)
    has_t = bool(df_all["target"].notna().any())
    cols = st.columns(6 if has_t else 5)
    c1, c2, c3, c4, c5 = cols[:5]
    c1.metric("Open issues", f"{k['open']:,}", dl("open"), delta_color="off")
    c2.metric("Already overdue", f"{k['overdue']:,}", dl("overdue"), delta_color=dc("overdue"))
    c3.metric("Flagged likely late", f"{k['late']:,}", dl("late"), delta_color=dc("late"),
              help=f"{k['late'] / max(k['not_overdue'], 1):.0%} of not-yet-overdue issues.")
    c4.metric("Confident late calls", f"{k['confident']:,}", dl("confident"), delta_color=dc("confident"),
              help="Issues whose 90% conformal prediction set contains only 'late'. On held-out test data "
                   "the sets contain the true outcome ~92% of the time, and a 'confident: late' call on a not-yet-"
                   "overdue issue was right ~97% of the time. 'Uncertain' means the model cannot rule either "
                   "outcome out at that level.")
    c5.metric("Median age, listed (d)", f"{k['age']:.0f}" if k["age"] == k["age"] else "–",
              help=f"This project's issues typically close in {b.project_median_days(pid):.0f} days; "
                   "older than that = already overdue.")
    if has_t:
        cols[5].metric("Past your target", f"{k['past_target']:,}", dl("past_target"), delta_color=dc("past_target"),
                       help="Open issues older than the target you set for their type (Configuration tab). "
                            "Pure arithmetic on age; targets never change the model's risk.")

    # ---- changes since the previous date ----
    new_ids, gone = set(), pd.DataFrame()
    forward = True
    if prev_df is not None:
        forward = date >= prev_date
        cur_ids = set(df["Issue_ID"]) if len(df) else set()
        prev_ids = set(prev_df["Issue_ID"]) if len(prev_df) else set()
        new_ids = cur_ids - prev_ids
        left = prev_ids - cur_ids
        if left:
            gone = prev_df[prev_df["Issue_ID"].isin(left)].copy()
            still_open = set(df_all["Issue_ID"]) if len(df_all) else set()
            out = b.outcomes(gone["Issue_ID"], date).set_index("Issue_ID")
            gone["finished_late"] = gone["Issue_ID"].map(out["finished_late"])
            gone["reason"] = np.where(gone["Issue_ID"].isin(still_open), "overdue",
                                      np.where(gone["Issue_ID"].map(out["resolved"]).fillna(False).astype(bool),
                                               "closed", "not_created"))
        closed = gone[gone["reason"] == "closed"] if len(gone) else gone
        if not forward:
            st.caption(f"Moved back from {prev_date:%d %b %Y}: **{len(new_ids)}** issues are back in the list "
                       f"(still open, not yet overdue on this date), **{len(gone)}** drop out (created later).")
        elif len(closed):
            called = (closed["flag"].isin(["likely late", "already overdue"]) == closed["finished_late"].astype(bool)).mean()
            st.caption(f"Since {prev_date:%d %b %Y}: **{len(new_ids)}** opened, **{len(closed)}** closed "
                       f"({int(closed['finished_late'].sum())} of them late). The model's call on the previous date was "
                       f"right for {called:.0%} of the closed ones.")
        else:
            st.caption(f"Since {prev_date:%d %b %Y}: **{len(new_ids)}** opened, nothing closed.")

    # ---- filters ----
    f1, f2 = st.columns([3, 2])
    opts = ["All", "Likely late", "On track"] + (["Already overdue"] if df["overdue"].any() else []) + \
           (["Past target"] if df["past_target"].any() else [])
    pick = f1.segmented_control("Show", opts, default="All", key=f"flow_filter_{pid}", label_visibility="collapsed")
    q = f2.text_input("Search", placeholder="Search issue ID or title", key=f"flow_q_{pid}",
                      label_visibility="collapsed")
    view = df
    flag_for = {"Likely late": "likely late", "On track": "on track", "Already overdue": "already overdue"}
    if pick in flag_for:
        view = view[view["flag"] == flag_for[pick]]
    elif pick == "Past target":
        view = view[view["past_target"]]
    if q:
        hit = lambda d: d[d["Issue_ID"].astype(str).str.contains(q) | d["Title"].str.contains(q, case=False, na=False)]
        view = hit(view)
        gone = hit(gone) if len(gone) else gone
    if pick not in (None, "All") and len(gone):
        gone = gone.iloc[0:0]

    import live_list
    importlib.reload(live_list)     # small module; picks up edits without a server restart
    show_changes = just_moved and prev_date is not None
    st.html(live_list.render(view, new_ids if show_changes else (), gone if show_changes else None,
                             prev_label=f"{prev_date:%d %b %Y}" if prev_date else "", forward=forward,
                             show_truth=show_truth, dark=_is_dark()))
    n_parked = len(parked_for(pid))
    st.caption(f"{len(view):,} issues listed" + (f" ({n_parked} parked, not counted)" if n_parked else "") +
               f" · sorted by risk · scroll inside the list. **Median age** = how long "
               f"this project's issues typically take to close ({b.project_median_days(pid):.0f} days); older than "
               f"that = already overdue." + (" Targets come from the Configuration tab." if has_t else ""))

    if show_truth:
        d = df.dropna(subset=["actually_late"])
        if d["actually_late"].nunique() == 2:
            from sklearn.metrics import roc_auc_score
            top = d.head(max(len(d) // 10, 1))
            st.caption(f"Check on this snapshot: AUC {roc_auc_score(d['actually_late'], d['risk']):.3f}; "
                       f"{top['actually_late'].mean():.0%} of the top-10% flags really ended up late "
                       f"(base rate {d['actually_late'].mean():.0%}).")


def issue_detail(pid, df, date):
    iid = st.selectbox("Issue", df["Issue_ID"].tolist(),
                       format_func=lambda i: f"{i} — {df.set_index('Issue_ID').loc[i, 'Title'][:80]}")
    row = df.set_index("Issue_ID").loc[iid]
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    h1.subheader(f"Risk {row['risk']:.2f} — {row['flag']}")
    with h2.popover("🅿️ Park", width="stretch", help="Not a priority any more: move it to the Parked table"):
        st.text_input("Reason (optional)", key=f"park1_reason_{pid}_{iid}", placeholder="e.g. nice-to-have")
        st.button("Park this issue", key=f"park1_btn_{pid}_{iid}", type="primary", width="stretch",
                  on_click=_park_ids, args=(pid, [int(iid)], date, f"park1_reason_{pid}_{iid}"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model confidence (90%)", row["confidence"].replace("confident: ", "") if row["confidence"] else "–",
              help="From a 90% conformal prediction set calibrated on the validation period.")
    med = row["median_days"]
    m2.metric("Project median age", f"{med:.0f} days", f"this one is {row['age_days'] / med:.1f}×", delta_color="off", delta_arrow="off",
              help="How long this project's issues typically take to close. Older than this = already overdue.")
    m3.metric("Open for", f"{row['age_days']:.0f} days")
    m4.metric("Assignee", "unassigned" if pd.isna(row["assignee"]) else f"{int(row['assignee'])}")
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
    vs_target(row)
    what_would_help(int(iid), pd.Timestamp(date), row)
    reassignment(int(iid), date, row)


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


def reassignment(iid, date, row):
    st.markdown("#### Reassignment")
    rkey = f"reassign_{iid}_{date}"
    if st.button("🔁 Check reassignment", key=f"reassign_btn_{iid}_{date}"):
        with st.spinner("Scoring active colleagues..."):
            try:
                st.session_state[rkey] = b.reassignment(iid, pd.Timestamp(date))
            except Exception as e:
                st.session_state[rkey] = {"error": str(e)}
    res = st.session_state.get(rkey)
    if res is None:
        st.caption("Re-scores this issue as if each active colleague in the project (5+ issues resolved here, "
                   "active in the last 180 days) owned it, using their track record and workload as of this date.")
        return
    if "error" in res:
        st.caption(f"Reassignment check unavailable ({res['error']}).")
        return
    cand = res["candidates"]
    cur = "unassigned" if res["current"] is None else f"assignee {res['current']}"
    share = f"Who it's assigned to accounts for {res['assignee_share']:.0%} of this issue's score"
    if res["verdict"] == "no_candidates":
        st.warning("No active colleagues with a track record in this project on this date, so no swap can be scored.")
        return
    best = cand.iloc[0]
    if res["verdict"] == "may_help":
        st.success(f"**Reassignment may help.** Moving it from {cur} to assignee {int(best['assignee'])} lowers risk "
                   f"{res['base']:.2f} → {best['risk_if_reassigned']:.2f} ({best['change']:+.2f}). {share}.")
    else:
        why = ("the risk mostly comes from the issue itself (its age, stage, activity, content), not from who owns it"
               if not res["assignee_in_top3"] else "no colleague's record is clearly better for this issue")
        st.warning(f"**Reassignment unlikely to help.** Best option changes risk by only {best['change']:+.2f} "
                   f"(needs −0.05 or more): {why}. {share}.")
    cr = res["current_record"]
    bars = pd.concat([pd.DataFrame([{"who": f"{cur} (current)", "risk": res["base"], "current": True}]),
                      pd.DataFrame({"who": "assignee " + cand["assignee"].astype(str), "risk": cand["risk_if_reassigned"],
                                    "current": False})])
    ch = alt.Chart(bars).mark_bar().encode(
        y=alt.Y("who:N", sort=alt.EncodingSortField("risk", order="ascending"), title=None,
                axis=alt.Axis(labelOverlap=False, labelLimit=260)),
        x=alt.X("risk:Q", scale=alt.Scale(domain=[0, 1]), title="predicted risk"),
        color=alt.condition("datum.current", alt.value(GREY), alt.value(BLUE)),
        tooltip=["who", alt.Tooltip("risk:Q", format=".2f")])
    st.altair_chart(ch.properties(height=max(160, 30 * len(bars))), width="stretch")
    table = pd.concat([pd.DataFrame([{"assignee": f"{cur} (current)", "resolved_here": cr["n"] if cr else None,
                                      "late_rate": cr["late"] if cr else None, "open_now": cr["open_now"] if cr else None,
                                      "risk_if_reassigned": res["base"], "change": 0.0}]),
                       cand.assign(assignee=cand["assignee"].astype(str))], ignore_index=True)
    table["late_rate"] = table["late_rate"].astype(float) * 100
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "resolved_here": st.column_config.NumberColumn("resolved here (before date)", format="%d"),
        "late_rate": st.column_config.NumberColumn("share late", format="%.0f%%"),
        "open_now": st.column_config.NumberColumn("open now (all projects)", format="%d"),
        "risk_if_reassigned": st.column_config.ProgressColumn("risk if they own it", min_value=0.0, max_value=1.0, format="%.2f"),
        "change": st.column_config.NumberColumn("change", format="%+.2f")})
    st.caption("Grey bar = current owner. Model estimate, not a guaranteed effect; checked against history, the "
               "reassignment estimate's size was about right.")


ZOOM_STEP = 1.5
MIN_TRACK_RECORD = 10    # closed issues needed before 'share late' is ranked as a track record


def _zoom(key, step, max_level):
    """step +1 = zoom in, -1 = zoom out, None = fit everything."""
    lvl = 0 if step is None else st.session_state.get(key, 0) + step
    st.session_state[key] = min(max(lvl, 0), max_level)


def team_workload(pid, date, df_all):
    w = b.team_workload(pid, pd.Timestamp(date), df_all)
    people = w[w["assignee"] != "unassigned"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People with open work", len(people))
    c2.metric("Unassigned open issues", int(w.loc[w["assignee"] == "unassigned", "open_here"].sum()))
    c3.metric("Overloaded", int((people["load"] == "overloaded").sum()),
              help="Open issues across all projects at least 2x the team median (and 5+).")
    c4.metric("People with 3+ late/overdue", int((people["likely_late"] + people["overdue"] >= 3).sum()))

    # ---- sort + zoom controls (buttons, not trackpad; the axis always starts at 0) ----
    sort_opts = {"Likely late": ("likely_late", False), "Already overdue": ("overdue", False),
                 "On track": ("on_track", False), "Open issues (this project)": ("open_here", False),
                 "Open across all projects": ("open_all_projects", False), "Share late (history)": ("late_rate", False)}
    n_max = len(w)
    c_sort, c_n, c_out, c_in, c_fit = st.columns([3, 3, 1.3, 1.3, 1], vertical_alignment="bottom")
    sort_by = c_sort.selectbox("Sort by", list(sort_opts), key=f"team_sort_{pid}")
    col, asc = sort_opts[sort_by]
    w = w.sort_values([col, "open_here"], ascending=[asc, False], na_position="last").reset_index(drop=True)
    tiers = [w["assignee"] != "unassigned", w["assignee"] == "unassigned"]
    if col == "late_rate":   # a rate from 1-9 closed issues is mostly luck (0% or 100%): rank real track records first
        enough = w["resolved_here"].fillna(0) >= MIN_TRACK_RECORD
        tiers = [tiers[0] & enough, tiers[0] & ~enough, tiers[1]]
    w = pd.concat([w[t] for t in tiers], ignore_index=True)
    n_show = c_n.slider("People in the chart", min_value=min(5, n_max), max_value=n_max, value=min(15, n_max),
                        key=f"team_n_{pid}") if n_max > 5 else n_max
    top = w.head(n_show)
    totals = top["open_here"].astype(int)
    full = max(int(totals.max()) if len(top) else 1, 1)
    zkey = f"team_zoom_{pid}"
    max_level = max(int(np.floor(np.log(full / 2) / np.log(ZOOM_STEP))), 0) if full > 2 else 0
    st.session_state[zkey] = min(max(st.session_state.get(zkey, 0), 0), max_level)
    level = st.session_state[zkey]
    c_out.button("Zoom out", icon=":material/zoom_out:", key=f"team_zout_{pid}", width="stretch", disabled=level == 0,
                 on_click=_zoom, args=(zkey, -1, max_level))
    c_in.button("Zoom in", icon=":material/zoom_in:", key=f"team_zin_{pid}", width="stretch", disabled=level >= max_level,
                on_click=_zoom, args=(zkey, +1, max_level))
    c_fit.button("Fit", key=f"team_zfit_{pid}", width="stretch", disabled=level == 0, help="Show every bar in full",
                 on_click=_zoom, args=(zkey, None, max_level))
    xmax = full / ZOOM_STEP ** level if level else full

    long = top.melt(id_vars=["assignee"], value_vars=["on_track", "likely_late", "overdue"],
                    var_name="status", value_name="issues")
    long["status"] = long["status"].map({"on_track": "on track", "likely_late": "likely late", "overdue": "already overdue"})
    order = top["assignee"].tolist()
    xs = alt.Scale(domain=[0, xmax], nice=False, zero=True, clamp=True)
    bars = alt.Chart(long).mark_bar(clip=True).encode(
        y=alt.Y("assignee:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("issues:Q", title="open issues in this project", stack="zero", scale=xs,
                axis=alt.Axis(tickMinStep=1, format="d")),
        color=alt.Color("status:N", scale=alt.Scale(domain=["on track", "likely late", "already overdue"],
                                                    range=["#2e9e5b", RED, GREY]), title=None,
                        legend=alt.Legend(orient="top")),
        order=alt.Order("status:N", sort="descending"),
        tooltip=["assignee", "status", "issues"])
    lab = pd.DataFrame({"assignee": top["assignee"], "x": np.minimum(totals, xmax),
                        "label": [f"{t} ›" if t > xmax else f"{t}" for t in totals]})
    text = alt.Chart(lab).mark_text(align="left", dx=4, fontSize=11).encode(
        y=alt.Y("assignee:N", sort=order), x=alt.X("x:Q", scale=xs), text="label:N")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)   # keeps the chart toolbar off the buttons
    st.altair_chart((bars + text).properties(height=max(220, 26 * len(top))), width="stretch")
    zoom_txt = f"zoom {ZOOM_STEP ** level:.1f}×, showing 0–{xmax:.0f} issues; '›' = bar continues past the edge" \
        if level else "zoomed out fully"
    if col == "late_rate":
        st.caption(f"Share late ranks people with {MIN_TRACK_RECORD}+ issues closed here first; people with fewer "
                   "come after them (their rate is mostly luck), then unassigned.")
    st.caption(f"Top {len(top)} of {n_max}, sorted by {sort_by.lower()}, as of the selected date · {zoom_txt}. "
               "Numbers at the bar ends are each person's open issues in this project.")

    st.dataframe(w[["assignee", "load", "open_here", "likely_late", "overdue", "avg_risk", "open_all_projects",
                    "resolved_here", "late_rate"]].assign(late_rate=w["late_rate"] * 100), hide_index=True, width="stretch", column_config={
        "open_here": st.column_config.NumberColumn("open here", format="%d"),
        "likely_late": st.column_config.NumberColumn("likely late", format="%d"),
        "overdue": st.column_config.NumberColumn("already overdue", format="%d"),
        "avg_risk": st.column_config.ProgressColumn("avg risk", min_value=0.0, max_value=1.0, format="%.2f"),
        "open_all_projects": st.column_config.NumberColumn("open (all projects)", format="%d"),
        "resolved_here": st.column_config.NumberColumn("resolved here before date", format="%d"),
        "late_rate": st.column_config.NumberColumn("share late (history)", format="%.0f%%")})

    who = st.selectbox("See one person's open issues", w["assignee"].tolist(), key=f"team_who_{pid}")
    mine = df_all[(df_all["assignee"].isna()) if who == "unassigned" else
                  (df_all["assignee"] == float(who))]
    st.dataframe(mine[["Issue_ID", "Title", "stage", "age_days", "risk", "flag", "why"]], hide_index=True,
                 width="stretch", column_config={
                     "risk": st.column_config.ProgressColumn("risk", min_value=0.0, max_value=1.0, format="%.2f"),
                     "age_days": st.column_config.NumberColumn("age (days)", format="%.0f"),
                     "why": st.column_config.TextColumn("top reasons", width="large")})
    st.caption("To move one of these, open it in Issue detail and use 🔁 Check reassignment.")


# =====================================================================================
# SCRUM MODE
# =====================================================================================
def scrum_mode():
    sb = scrum_backend(b, VERSION)
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
