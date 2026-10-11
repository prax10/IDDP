"""
Task L4, step 1: convert the Public Jira Dataset (Montgomery et al., MSR 2022, Zenodo
10.5281/zenodo.5882881, CC BY 4.0) from its mongodump BSON files into the SAME raw
table layout as the TAWOS caches in data/raw/, so feature_engine.FeatureEngine can run
on it unchanged.

No MongoDB server is needed: bson.decode_file_iter streams documents from the .bson file.

  python3 l4_convert_jira_dump.py <path/to/Collection.bson> <out_root> [--max N]
  -> <out_root>/raw/{issue_slim.csv.gz, comment.csv, change_log_status.csv.gz,
                     change_log_people.csv, change_log_sprint.csv, sprint.csv,
                     change_log_other.csv, change_log_fields.csv.gz, issue_text.csv.gz}

Mapping (Jira REST JSON -> TAWOS columns), kept deliberately identical to TAWOS:
  status changelog          -> change_log_status   (only field == "status"; TAWOS mixed in
                                                     resolution rows, which FeatureEngine drops)
  assignee changelog        -> change_log_people   (Field = assignee)
  Sprint changelog          -> change_log_sprint   (sprint names, comma-separated)
  Story Points changelog    -> change_log_other    (Change_Type = STORY_POINT)
  Fix Version, labels, Rank, Link, Component, priority, issuetype, Epic Link,
  timeestimate, timeoriginalestimate               -> change_log_fields (the exact TAWOS set)
  sprint objects found in custom fields            -> sprint.csv
  title / description lengths, desc_has_code       -> same regex as task_i1
Issue, project and person ids are re-numbered to integers (issues from 10,000,000 up so
they can never collide with TAWOS ids).
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import bson

KEEP_FIELDS = {"fix version", "labels", "rank", "link", "component", "priority", "issuetype", "epic link",
               "timeestimate", "timeoriginalestimate"}
SP_FIELDS = {"story points", "story point estimate", "story point"}
CODE_RE = re.compile(r"\{code|\{noformat|```|Exception|at [a-z]+\.[a-z]")
SPRINT_RE = re.compile(r"\[(.*)\]")
ISSUE_OFFSET = 10_000_000


def ts(v):
    """BSON datetime or Jira ISO string -> 'YYYY-MM-DD HH:MM:SS' in UTC (TAWOS format)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return pd.Timestamp(v).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S") if v.tzinfo is None else \
            pd.Timestamp(v).tz_convert("UTC").tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
    try:
        t = pd.Timestamp(str(v))
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


class Ids:
    def __init__(self, start=1):
        self.m, self.n = {}, start

    def __call__(self, k):
        if k is None:
            return None
        if k not in self.m:
            self.m[k] = self.n
            self.n += 1
        return self.m[k]


def person_key(p):
    if not isinstance(p, dict):
        return None
    return p.get("accountId") or p.get("key") or p.get("name") or p.get("displayName")


def parse_sprint(v):
    """custom-field sprint value (greenhopper string or dict) -> dict or None"""
    if isinstance(v, dict) and "name" in v and ("startDate" in v or "state" in v):
        return {k: v.get(k) for k in ("id", "name", "state", "startDate", "endDate", "completeDate", "activatedDate")}
    if isinstance(v, str) and "greenhopper" in v and "name=" in v:
        m = SPRINT_RE.search(v)
        if not m:
            return None
        d = {}
        for part in re.split(r",(?=[a-zA-Z]+=)", m.group(1)):
            if "=" in part:
                k, val = part.split("=", 1)
                d[k] = None if val in ("<null>", "") else val
        return {k: d.get(k) for k in ("id", "name", "state", "startDate", "endDate", "completeDate", "activatedDate")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bson_file")
    ap.add_argument("out_root")
    ap.add_argument("--max", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(f"{a.out_root}/raw", exist_ok=True)

    issue_ids, proj_ids, people = Ids(ISSUE_OFFSET), Ids(1000), Ids(1)
    key_to_id = {}
    I, CM, ST, PE, SC, SP, FL, TX = [], [], [], [], [], [], [], []
    sprints = {}
    n = 0
    with open(a.bson_file, "rb") as fh:
        for doc in bson.decode_file_iter(fh):
            n += 1
            if a.max and n > a.max:
                break
            f = doc.get("fields") or {}
            if not f or not f.get("created"):
                continue
            iid = issue_ids(str(doc.get("id") or doc.get("_id")))
            key_to_id[doc.get("key")] = iid
            proj = f.get("project") or {}
            pid = proj_ids(proj.get("key") or proj.get("id"))
            title = f.get("summary") or ""
            desc = f.get("description") or ""
            if not isinstance(desc, str):          # Atlassian document format (newer Cloud API)
                desc = str(desc)
            sp_now = None
            spr_now = None
            for k, v in f.items():
                if not k.startswith("customfield_") or v is None:
                    continue
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    s = parse_sprint(x)
                    if s and s.get("name"):
                        sprints[(pid, s["name"])] = s
                        spr_now = s.get("id")
            links = f.get("issuelinks") or []
            I.append({"ID": iid, "Project_ID": pid,
                      "Priority": (f.get("priority") or {}).get("name"), "Type": (f.get("issuetype") or {}).get("name"),
                      "Status": (f.get("status") or {}).get("name"), "Resolution": (f.get("resolution") or {}).get("name"),
                      "Story_Point": sp_now, "Assignee_ID": people(person_key(f.get("assignee"))),
                      "Creator_ID": people(person_key(f.get("creator"))), "Reporter_ID": people(person_key(f.get("reporter"))),
                      "Sprint_ID": spr_now, "Creation_Date": ts(f.get("created")), "Resolution_Date": ts(f.get("resolutiondate")),
                      "Estimation_Date": None, "title_len": len(title), "desc_len": len(desc),
                      "desc_has_code": int(bool(CODE_RE.search(desc))), "Key": doc.get("key"),
                      "n_links_final": len(links),
                      "n_blocking_links_final": sum(1 for l in links if "block" in str((l.get("type") or {}).get("name", "")).lower())})
            TX.append({"ID": iid, "Title": title, "Description_Text": desc[:20000]})
            cms = (f.get("comment") or {}).get("comments") or doc.get("comments") or []
            for c in cms:
                CM.append({"Issue_ID": iid, "Creation_Date": ts(c.get("created")), "Author_ID": people(person_key(c.get("author")))})
            hist = (doc.get("changelog") or {}).get("histories") or []
            for h in hist:
                t = ts(h.get("created"))
                for it in h.get("items") or []:
                    fld = str(it.get("field") or "")
                    fl = fld.lower()
                    fr, to = it.get("fromString"), it.get("toString")
                    if fl == "status":
                        ST.append({"Issue_ID": iid, "Change_Type": "STATUS", "From_String": fr, "To_String": to, "Creation_Date": t})
                    elif fl == "assignee":
                        PE.append({"Issue_ID": iid, "Change_Type": "PEOPLE", "Field": "assignee",
                                   "From_String": it.get("from") or fr, "To_String": it.get("to") or to, "Creation_Date": t})
                    elif fl == "sprint":
                        SC.append({"Issue_ID": iid, "Change_Type": "OTHER", "Field": "Sprint", "From_String": fr,
                                   "To_String": to, "Creation_Date": t})
                    elif fl in SP_FIELDS:
                        SP.append({"Issue_ID": iid, "Change_Type": "STORY_POINT", "From_String": fr, "To_String": to,
                                   "Creation_Date": t})
                    elif fl in KEEP_FIELDS:
                        FL.append({"Issue_ID": iid, "Field": fld, "From_String": fr, "To_String": to, "Creation_Date": t})
            if n % 20000 == 0:
                print(f"  {n:,} documents", flush=True)

    out = f"{a.out_root}/raw"
    iss = pd.DataFrame(I)
    # final story points: last story-point changelog value (custom field ids differ per repo)
    sp = pd.DataFrame(SP, columns=["Issue_ID", "Change_Type", "From_String", "To_String", "Creation_Date"])
    if len(sp):
        last_sp = sp.sort_values("Creation_Date").groupby("Issue_ID")["To_String"].last()
        iss["Story_Point"] = pd.to_numeric(iss["ID"].map(last_sp), errors="coerce")
    iss.to_csv(f"{out}/issue_slim.csv.gz", index=False)
    pd.DataFrame(TX).to_csv(f"{out}/issue_text.csv.gz", index=False)
    pd.DataFrame(CM, columns=["Issue_ID", "Creation_Date", "Author_ID"]).to_csv(f"{out}/comment.csv", index=False)
    pd.DataFrame(ST, columns=["Issue_ID", "Change_Type", "From_String", "To_String", "Creation_Date"]).to_csv(
        f"{out}/change_log_status.csv.gz", index=False)
    pd.DataFrame(PE, columns=["Issue_ID", "Change_Type", "Field", "From_String", "To_String", "Creation_Date"]).to_csv(
        f"{out}/change_log_people.csv", index=False)
    pd.DataFrame(SC, columns=["Issue_ID", "Change_Type", "Field", "From_String", "To_String", "Creation_Date"]).to_csv(
        f"{out}/change_log_sprint.csv", index=False)
    sp.to_csv(f"{out}/change_log_other.csv", index=False)
    pd.DataFrame(FL, columns=["Issue_ID", "Field", "From_String", "To_String", "Creation_Date"]).to_csv(
        f"{out}/change_log_fields.csv.gz", index=False)
    spr = pd.DataFrame([{"ID": i + 1, "JiraID": s.get("id"), "Name": s["name"], "State": s.get("state"),
                         "Start_Date": ts(s.get("startDate")), "End_Date": ts(s.get("endDate")),
                         "Activated_Date": ts(s.get("activatedDate")), "Complete_Date": ts(s.get("completeDate")),
                         "Project_ID": pid} for i, ((pid, _), s) in enumerate(sprints.items())],
                       columns=["ID", "JiraID", "Name", "State", "Start_Date", "End_Date", "Activated_Date",
                                "Complete_Date", "Project_ID"])
    spr.to_csv(f"{out}/sprint.csv", index=False)
    pd.Series({k: v for k, v in proj_ids.m.items()}).rename("Project_ID").to_csv(f"{out}/project_map.csv")
    res = iss["Resolution_Date"].notna()
    print(f"{a.bson_file}: {len(iss):,} issues ({res.mean():.0%} resolved), {iss.Project_ID.nunique()} projects, "
          f"{len(CM):,} comments, {len(ST):,} status changes, {len(PE):,} assignee changes, {len(SC):,} sprint changes, "
          f"{len(sp):,} story-point changes, {len(FL):,} other field changes, {len(spr):,} sprints")


if __name__ == "__main__":
    main()
