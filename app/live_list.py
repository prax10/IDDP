"""
live_list.py -- the flow-mode issue list as HTML, so rows can be animated when the date moves.

  render(open_df, new_ids, gone_df, ...)  -> HTML string for st.html

When the as-of date changes:
  * issues that APPEARED are pinned to the top and glow green for 5 s, then fade to normal;
  * issues that LEFT the list (closed, or crossed into "already overdue" while overdue issues are
    hidden) are shown above them, dimmed, for 4 s; then they quietly fade and fold away, so the
    new issues end up at the very top.
On any other rerun (no date change) the list is drawn plainly, sorted by risk.
Pure CSS animations: the element is re-created when its HTML changes, which restarts them.
"""

import html

import numpy as np
import pandas as pd

HOLD_S, FADE_S = 5.0, 2.5            # new rows: glow for 5 s, then fade over 2.5 s
GONE_HOLD_S, GONE_FADE_S = 4.0, 1.5  # leaving rows: shown dimmed for 4 s, then a quiet fade + collapse


def _risk_color(r):
    return "#d1495b" if r >= 0.7 else ("#e8a33d" if r >= 0.4 else "#2e9e5b")


def _flag_pill(flag):
    cls = {"likely late": "late", "on track": "ok", "already overdue": "over"}.get(flag, "ok")
    return f'<span class="dr-pill dr-{cls}">{html.escape(flag)}</span>'


def _css(dark):
    head_bg = "#0e1117" if dark else "#ffffff"
    sec_bg = "rgba(250,250,250,.04)" if dark else "rgba(0,0,0,.03)"
    tot_new = HOLD_S + FADE_S
    tot_gone = GONE_HOLD_S + GONE_FADE_S + 0.4
    p_new = HOLD_S / tot_new * 100
    p_gone_hold = GONE_HOLD_S / tot_gone * 100
    p_gone_faded = (GONE_HOLD_S + GONE_FADE_S) / tot_gone * 100
    return f"""
<style>
.dr-wrap {{ max-height: 600px; overflow-y: auto; border: 1px solid rgba(128,128,128,.28); border-radius: 10px;
           font-size: 14px; line-height: 1.35; }}
.dr-row {{ display: grid; grid-template-columns: 92px minmax(260px, 1fr) 128px 64px 150px 128px 118px;
          gap: 12px; align-items: center; padding: 8px 14px; border-bottom: 1px solid rgba(128,128,128,.14);
          max-height: 140px; overflow: hidden; }}
.dr-t .dr-row {{ grid-template-columns: 92px minmax(260px, 1fr) 128px 64px 150px 128px 118px 70px; }}
.dr-head {{ position: sticky; top: 0; z-index: 3; background: {head_bg}; font-size: 11.5px; font-weight: 600;
           text-transform: uppercase; letter-spacing: .04em; opacity: .9; border-bottom: 1px solid rgba(128,128,128,.35); }}
.dr-sec {{ padding: 6px 14px; font-size: 12.5px; font-weight: 600; background: {sec_bg};
          border-bottom: 1px solid rgba(128,128,128,.14); max-height: 60px; overflow: hidden; }}
.dr-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
.dr-title {{ font-weight: 500; }}
.dr-why {{ font-size: 12px; opacity: .62; margin-top: 2px; }}
.dr-muted {{ font-size: 12px; opacity: .62; }}
.dr-bar {{ height: 7px; border-radius: 4px; background: rgba(128,128,128,.22); overflow: hidden; margin-top: 3px; }}
.dr-bar > div {{ height: 100%; border-radius: 4px; }}
.dr-pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
.dr-late {{ background: rgba(209,73,91,.16); color: #d1495b; }}
.dr-ok   {{ background: rgba(46,158,91,.16); color: #2e9e5b; }}
.dr-over {{ background: rgba(137,135,129,.20); color: #898781; }}
.dr-tag  {{ display: block; width: fit-content; margin-top: 3px; padding: 0 6px; border-radius: 4px; font-size: 10px;
           font-weight: 700; letter-spacing: .04em; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
.dr-row.dr-compact {{ grid-template-columns: 158px minmax(260px, 1fr) 190px 260px; padding: 5px 14px; }}
.dr-compact .dr-title {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.dr-compact .dr-id {{ display: flex; gap: 6px; align-items: center; }}
.dr-compact .dr-tag {{ margin-top: 0; }}
.dr-tag-new  {{ background: #22c55e; color: #fff; }}
.dr-tgt {{ font-size: 12px; font-weight: 600; margin-top: 1px; }}
.dr-tgt-past {{ color: #d1495b; }}
.dr-tgt-ok {{ color: #2e9e5b; }}
.dr-tag-gone {{ background: #898781; color: #fff; }}
.dr-tag-over {{ background: #898781; color: #fff; }}

@keyframes drNew {{
  0%       {{ background: rgba(34,197,94,.30); box-shadow: inset 4px 0 0 #22c55e; }}
  {p_new:.1f}%   {{ background: rgba(34,197,94,.30); box-shadow: inset 4px 0 0 #22c55e; }}
  100%     {{ background: transparent;        box-shadow: inset 4px 0 0 rgba(34,197,94,.35); }}
}}
@keyframes drGone {{
  0%       {{ opacity: .6; }}
  {p_gone_hold:.1f}%   {{ opacity: .6; }}
  {p_gone_faded:.1f}%  {{ opacity: 0; max-height: 60px; padding-top: 5px; padding-bottom: 5px; }}
  100%     {{ opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; border-bottom-width: 0; }}
}}
.dr-new  {{ animation: drNew {tot_new}s ease-out forwards; }}
.dr-gone, .dr-gone-over {{ animation: drGone {tot_gone}s ease forwards; }}
</style>"""


def _median_cell(r):
    """Project median age (the 'overdue' line), how far past it this issue is, and the manager's target if set."""
    med, age = r.get("median_days"), float(r["age_days"])
    if med is None or pd.isna(med) or med <= 0:
        out = "–"
    else:
        x = age / med
        out = f'{med:.0f}<div class="dr-muted">now {x:.1f}×</div>' if x >= 1 else \
              f'{med:.0f}<div class="dr-muted">{med - age:.0f} d to go</div>'
    tgt = r.get("target")
    if tgt is not None and not pd.isna(tgt):
        if age > tgt:
            out += f'<div class="dr-tgt dr-tgt-past">target {tgt:g} · past</div>'
        else:
            out += f'<div class="dr-tgt dr-tgt-ok">target {tgt:g} · {tgt - age:.0f} d left</div>'
    return out


def _open_row(r, cls="", tag="", show_truth=False):
    risk = float(r["risk"])
    why = html.escape(str(r.get("why", "") or ""))
    conf = str(r.get("confidence", "") or "").replace("confident: ", "confident ")
    truth = ""
    if show_truth:
        a = r.get("actually_late")
        truth = f'<div>{"—" if pd.isna(a) else ("late" if a == 1 else "on time")}</div>'
    return (f'<div class="dr-row {cls}">'
            f'<div class="dr-id">{int(r["Issue_ID"])}{tag}</div>'
            f'<div><div class="dr-title">{html.escape(str(r["Title"])[:140])}</div><div class="dr-why">{why}</div></div>'
            f'<div>{html.escape(str(r["stage"]))}</div>'
            f'<div>{float(r["age_days"]):.0f}</div>'
            f'<div>{risk:.2f}<div class="dr-bar"><div style="width:{risk * 100:.0f}%;background:{_risk_color(risk)}"></div></div></div>'
            f'<div>{_flag_pill(r["flag"])}<div class="dr-muted">{html.escape(conf)}</div></div>'
            f'<div>{_median_cell(r)}</div>'
            f'{truth}</div>')


def _gone_row(r, reason, forward, show_truth=False):
    """A compact row for an issue that just left the list. reason: 'closed' | 'overdue' | 'not_created'."""
    risk = float(r["risk"])
    if reason == "overdue":
        cls, tag = "dr-gone-over", '<span class="dr-tag dr-tag-over">OVERDUE</span>'
        status = '<span class="dr-pill dr-over">moved to the overdue list</span>'
    elif reason == "not_created":
        cls, tag = "dr-gone-over", '<span class="dr-tag dr-tag-over">LATER</span>'
        status = '<span class="dr-pill dr-over">not created yet on this date</span>'
    else:
        cls, tag = "dr-gone", '<span class="dr-tag dr-tag-gone">CLOSED</span>'
        late = r.get("finished_late")
        if pd.isna(late):
            status = '<span class="dr-pill dr-over">closed</span>'
        else:
            said_late = r["flag"] in ("likely late", "already overdue")
            right = "✓ model called it" if said_late == bool(late) else "✗ model missed it"
            status = (f'<span class="dr-pill {"dr-late" if late else "dr-ok"}">finished {"late" if late else "on time"}</span>'
                      f' <span class="dr-muted">{right}</span>')
    return (f'<div class="dr-row dr-compact {cls}">'
            f'<div class="dr-id">{int(r["Issue_ID"])}{tag}</div>'
            f'<div class="dr-title" title="{html.escape(str(r["Title"]))}">{html.escape(str(r["Title"]))}</div>'
            f'<div class="dr-muted">was {risk:.2f} · {html.escape(r["flag"])}</div>'
            f'<div>{status}</div></div>')


def render(open_df, new_ids=(), gone_df=None, prev_label="", forward=True, show_truth=False,
           dark=False, max_gone=6):
    """open_df: rows to list (already filtered, sorted by risk). new_ids: ids that just appeared.
    gone_df: rows from the previous date that just left, with column `reason` (+ `finished_late`)."""
    new_ids = set(int(i) for i in new_ids)
    head = ('<div class="dr-row dr-head">'
            '<div>Issue</div><div>Title · top reasons (↑ raises risk, ↓ lowers)</div><div>Stage</div><div>Age (d)</div>'
            '<div>Risk</div><div>Flag</div><div title="How long this project\'s issues typically take to close. Older than this = already overdue.">Median age (d)</div>' + ("<div>Actual</div>" if show_truth else "") + "</div>")
    parts = [_css(dark), f'<div class="dr-wrap{" dr-t" if show_truth else ""}">', head]

    is_new = open_df["Issue_ID"].astype(int).isin(new_ids) if len(open_df) else pd.Series([], dtype=bool)
    fresh, rest = open_df[is_new], open_df[~is_new]
    if gone_df is not None and len(gone_df):
        order = {"closed": 0, "overdue": 1, "not_created": 2}
        g = gone_df.assign(_o=gone_df["reason"].map(order)).sort_values(["_o", "risk"], ascending=[True, False])
        n_closed = int((g["reason"] == "closed").sum())
        n_over = int((g["reason"] == "overdue").sum())
        n_later = int((g["reason"] == "not_created").sum())
        bits = [f"{n_closed} closed"] * bool(n_closed) + [f"{n_over} now overdue"] * bool(n_over) + \
               [f"{n_later} not created yet"] * bool(n_later)
        parts.append(f'<div class="dr-sec dr-gone">↘ {", ".join(bits)} {"since" if forward else "vs"} {html.escape(prev_label)}</div>')
        parts += [_gone_row(r, r["reason"], forward, show_truth) for _, r in g.head(max_gone).iterrows()]
        if len(g) > max_gone:
            parts.append(f'<div class="dr-sec dr-gone dr-muted">… and {len(g) - max_gone} more (counts above the list)</div>')

    if len(fresh):
        txt = (f"🆕 {len(fresh)} opened since {html.escape(prev_label)}" if forward else
               f"↩ {len(fresh)} back in the list (open and not yet overdue on this date, vs {html.escape(prev_label)})")
        parts.append(f'<div class="dr-sec dr-new">{txt}</div>')
        tag = '<span class="dr-tag dr-tag-new">NEW</span>' if forward else '<span class="dr-tag dr-tag-new">BACK</span>'
        parts += [_open_row(r, "dr-new", tag, show_truth) for _, r in fresh.iterrows()]

    if len(fresh) or (gone_df is not None and len(gone_df)):
        parts.append(f'<div class="dr-sec">All other open issues, by risk ({len(rest)})</div>')
    parts += [_open_row(r, "", "", show_truth) for _, r in rest.iterrows()]
    if not len(open_df):
        parts.append('<div class="dr-sec">No open issues match.</div>')
    parts.append("</div>")
    return "\n".join(parts)
