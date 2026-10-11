"""
Task L4, step 2: rebuild the TAWOS pipeline on an external Jira repository, with the
same definitions, so a model trained on TAWOS can be tested on it.

  python3 l4_build_external.py <ext_root>          (ext_root made by l4_convert_jira_dump.py)

Definitions copied from the TAWOS pipeline:
  population   issues resolved as completed work (Fixed / Done / Complete ..., the TAWOS set;
               Duplicate, Won't Fix, Invalid ... excluded), resolution time > 0, projects with >= 30 resolved
               train+val issues
  split        chronological by creation date within this repository: 70 / 10 / 20
  label        delayed_v2 = resolution time > the project's train+val median
  censoring    'safe' = created at least the project's train+val p90 resolution time before
               the last event in the dump (otherwise only fast issues could be resolved yet)
  checkpoints  M_k = k/10 x expected_duration_proxy, where the proxy is an EXPANDING median of
               issues resolved before this issue was created, first group with >= 5 such issues:
               project+type+priority -> project+type -> project+priority -> project -> global.
               A checkpoint row exists only if the issue was still open at that moment.
  features     feature_engine.FeatureEngine pointed at <ext_root>, i.e. the identical code,
               every feature from events strictly before the checkpoint moment
Writes <ext_root>/features/{phase6_feature_table_v3.csv, phase8_split.csv, checkpoints.csv,
checkpoint_features.parquet}.
"""

import os
import sys
import numpy as np
import pandas as pd
from sortedcontainers import SortedList

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engine import FeatureEngine, secs

# TAWOS resolved population = issues closed as completed work (derived from the TAWOS caches:
# these resolutions are kept 93-100%, every other resolution 0%)
COMPLETED = {"fixed", "done", "complete", "completed", "handled by support", "resolved locally", "implemented",
             "deployed", "resolved"}
TAWOS_TYPES = {"Bug", "Story", "Improvement", "Task", "Suggestion", "Sub-task", "New Feature", "Epic",
               "Enhancement Request", "Technical task", "Question", "Support Request", "Documentation",
               "Test Task", "Build Failure"}


def norm_priority(p):
    s = str(p).lower()
    for words, v in [(("blocker", "highest", "p1", "showstopper", "urgent"), "5"), (("critical", "high", "p2"), "4"),
                     (("major", "medium", "normal", "p3"), "3"), (("minor", "low", "p4"), "2"),
                     (("trivial", "lowest", "p5"), "1")]:
        if any(w in s for w in words):
            return v
    return "Unknown"


def norm_type(t):
    t = str(t)
    if t in TAWOS_TYPES:
        return t
    low = t.lower()
    if low in ("subtask", "sub task"):
        return "Sub-task"
    if low in ("feature", "new feature request"):
        return "New Feature"
    return "Other"


def expected_duration_proxy(df):
    """expanding hierarchical median of resolution minutes of issues resolved before each creation."""
    levels = [("Project_ID", "Type_Normalized", "Priority_Normalized"), ("Project_ID", "Type_Normalized"),
              ("Project_ID", "Priority_Normalized"), ("Project_ID",), ()]
    ev = pd.concat([pd.DataFrame({"t": df["c_s"], "kind": 1, "i": np.arange(len(df))}),       # query at creation
                    pd.DataFrame({"t": df["r_s"], "kind": 2, "i": np.arange(len(df))})])      # add at resolution
    ev = ev.sort_values(["t", "kind"], kind="mergesort").to_numpy()   # queries first on ties: strictly before
    keys = [list(zip(*[df[c].astype(str) for c in lv])) if lv else [()] * len(df) for lv in levels]
    rt = df["Resolution_Time_Minutes"].to_numpy(float)
    stores = [dict() for _ in levels]
    out = np.full(len(df), np.nan)
    for t, kind, i in ev:
        i = int(i)
        if kind == 2:
            for L in range(len(levels)):
                stores[L].setdefault(keys[L][i], SortedList()).add(rt[i])
        else:
            for L in range(len(levels)):
                s = stores[L].get(keys[L][i])
                last = L == len(levels) - 1          # global level: used whatever its count (as in TAWOS)
                if s is not None and (len(s) >= 5 or (last and len(s) > 0)):
                    n = len(s)
                    out[i] = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
                    break
    return out


def main():
    root = sys.argv[1]
    raw, feat = f"{root}/raw", f"{root}/features"
    os.makedirs(feat, exist_ok=True)
    iss = pd.read_csv(f"{raw}/issue_slim.csv.gz")
    iss["c_s"], iss["r_s"] = secs(iss["Creation_Date"]), secs(iss["Resolution_Date"])
    ok = (iss["c_s"] > 0) & (iss["r_s"] > iss["c_s"]) & iss["Resolution"].astype(str).str.lower().str.strip().isin(COMPLETED)
    df = iss[ok].copy()
    df["Resolution_Time_Minutes"] = (df["r_s"] - df["c_s"]) / 60
    df["Priority_Normalized"] = df["Priority"].map(norm_priority)
    df["Type_Normalized"] = df["Type"].map(norm_type)

    cut = df["c_s"].quantile([0.7, 0.8]).to_numpy()
    df["split"] = np.where(df["c_s"] < cut[0], "train", np.where(df["c_s"] < cut[1], "val", "test"))
    tv = df[df["split"].isin(["train", "val"])]
    g = tv.groupby("Project_ID")["Resolution_Time_Minutes"]
    keep = g.size()[g.size() >= 30].index
    df = df[df["Project_ID"].isin(keep)].copy()
    med, p90 = g.median(), g.quantile(0.9)
    df["delayed_v2"] = (df["Resolution_Time_Minutes"] > df["Project_ID"].map(med)).astype(int)
    end = max(iss["c_s"].max(), iss["r_s"].max())
    df["safe"] = (df["c_s"] <= end - df["Project_ID"].map(p90) * 60).astype(int)
    print(f"resolved population {len(df):,} issues in {df.Project_ID.nunique()} projects; split "
          f"{df.split.value_counts().to_dict()}; late rate {df.delayed_v2.mean():.1%}; "
          f"test censoring-safe {df[df.split == 'test'].safe.mean():.0%}", flush=True)

    # FeatureEngine needs these two files (label for known-at-T track records; split for the median)
    df.rename(columns={})[["ID", "Project_ID", "delayed_v2", "Resolution_Time_Minutes", "Priority_Normalized",
                           "Type_Normalized", "Story_Point", "safe"]].to_csv(f"{feat}/phase6_feature_table_v3.csv", index=False)
    df[["ID", "split"]].to_csv(f"{feat}/phase8_split.csv", index=False)

    df = df.sort_values("c_s").reset_index(drop=True)
    df["proxy_min"] = expected_duration_proxy(df)
    df = df[df["proxy_min"].notna() & (df["proxy_min"] > 0)]
    rows = []
    for k in range(1, 11):
        el = k / 10 * df["proxy_min"]
        alive = df["Resolution_Time_Minutes"] > el
        rows.append(pd.DataFrame({"Issue_ID": df.loc[alive, "ID"], "checkpoint_index": k, "elapsed_minutes": el[alive]}))
    ck = pd.concat(rows).merge(df[["ID", "Project_ID", "split", "delayed_v2", "safe", "Priority_Normalized",
                                   "Type_Normalized", "Resolution_Time_Minutes", "c_s"]].rename(columns={"ID": "Issue_ID"}),
                               on="Issue_ID").sort_values(["Issue_ID", "checkpoint_index"]).reset_index(drop=True)
    ck.drop(columns="c_s").to_csv(f"{feat}/checkpoints.csv", index=False)
    print(f"checkpoint rows {len(ck):,} over {ck.Issue_ID.nunique():,} issues "
          f"(mean {len(ck) / ck.Issue_ID.nunique():.2f} per issue)", flush=True)

    fe = FeatureEngine(verbose=True, raw_dir=raw, features_dir=feat)
    T = ck["c_s"].to_numpy(np.int64) + np.floor(ck["elapsed_minutes"].to_numpy() * 60).astype(np.int64)
    parts = []
    for i in range(0, len(ck), 200_000):
        parts.append(fe.at(ck["Issue_ID"].to_numpy()[i:i + 200_000], T[i:i + 200_000]))
    X = pd.concat(parts, ignore_index=True)
    X = pd.concat([ck.drop(columns="c_s").reset_index(drop=True), X.drop(columns=["Issue_ID"])], axis=1)
    X.to_parquet(f"{feat}/checkpoint_features.parquet", index=False)
    print(f"features {X.shape} -> {feat}/checkpoint_features.parquet")
    print(X.groupby("checkpoint_index")["delayed_v2"].agg(["size", "mean"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
