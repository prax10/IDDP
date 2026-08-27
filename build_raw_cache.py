"""
Build a narrow, disk-efficient local cache of TAWOS_DB tables for the full
row population (Issue, Change_Log[STATUS], Issue_Link, Comment).

Read-only against MySQL. Writes CSVs to data/raw/.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE_URL = "mysql+pymysql://root:root%40123@localhost/TAWOS_DB"
OUT_DIR = "data/raw"

ISSUE_CATEGORY_COLS = ["Priority", "Type", "Status", "Resolution"]
CHANGE_LOG_CATEGORY_COLS = ["Change_Type", "From_String", "To_String"]

ISSUE_DATE_COLS = ["Creation_Date", "Resolution_Date", "Estimation_Date"]
CHANGE_LOG_DATE_COLS = ["Creation_Date"]
COMMENT_DATE_COLS = ["Creation_Date"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = create_engine(ENGINE_URL)

    # --- Issue ---
    issue_cols = (
        "ID, Project_ID, Priority, Type, Status, Resolution, "
        "Resolution_Time_Minutes, In_Progress_Minutes, Story_Point, "
        "Total_Effort_Minutes, Assignee_ID, Creator_ID, Reporter_ID, "
        "Sprint_ID, Creation_Date, Resolution_Date, Estimation_Date, "
        "Title, Description_Text"
    )
    print("Extracting Issue ...")
    issue_df = pd.read_sql(
        f"SELECT {issue_cols} FROM Issue", engine, parse_dates=ISSUE_DATE_COLS
    )
    for col in ISSUE_CATEGORY_COLS:
        issue_df[col] = issue_df[col].astype("category")
    issue_path = os.path.join(OUT_DIR, "issue.csv")
    issue_df.to_csv(issue_path, index=False)
    print(f"  -> {issue_path} ({len(issue_df)} rows)")

    # --- Change_Log (STATUS only) ---
    print("Extracting Change_Log (Change_Type = 'STATUS') ...")
    change_log_df = pd.read_sql(
        "SELECT Issue_ID, Change_Type, From_String, To_String, Creation_Date "
        "FROM Change_Log WHERE Change_Type = 'STATUS'",
        engine,
        parse_dates=CHANGE_LOG_DATE_COLS,
    )
    for col in CHANGE_LOG_CATEGORY_COLS:
        change_log_df[col] = change_log_df[col].astype("category")
    change_log_path = os.path.join(OUT_DIR, "change_log_status.csv")
    change_log_df.to_csv(change_log_path, index=False)
    print(f"  -> {change_log_path} ({len(change_log_df)} rows)")

    # --- Issue_Link ---
    print("Extracting Issue_Link ...")
    issue_link_df = pd.read_sql(
        "SELECT Issue_ID, Description, Target_Issue_ID FROM Issue_Link", engine
    )
    issue_link_path = os.path.join(OUT_DIR, "issue_link.csv")
    issue_link_df.to_csv(issue_link_path, index=False)
    print(f"  -> {issue_link_path} ({len(issue_link_df)} rows)")

    # --- Comment ---
    print("Extracting Comment ...")
    comment_df = pd.read_sql(
        "SELECT Issue_ID, Creation_Date, Author_ID FROM Comment",
        engine,
        parse_dates=COMMENT_DATE_COLS,
    )
    comment_path = os.path.join(OUT_DIR, "comment.csv")
    comment_df.to_csv(comment_path, index=False)
    print(f"  -> {comment_path} ({len(comment_df)} rows)")

    # ------------------------------------------------------------------
    # Step 3: verification
    # ------------------------------------------------------------------
    print("\n=== Verification ===")

    with engine.connect() as conn:
        live_issue_count = conn.execute(text("SELECT COUNT(*) FROM Issue")).scalar()
        live_change_log_status_count = conn.execute(
            text("SELECT COUNT(*) FROM Change_Log WHERE Change_Type = 'STATUS'")
        ).scalar()
        live_issue_link_count = conn.execute(
            text("SELECT COUNT(*) FROM Issue_Link")
        ).scalar()
        live_comment_count = conn.execute(text("SELECT COUNT(*) FROM Comment")).scalar()

    checks = []

    # 1. Row counts
    checks.append(("Issue row count", live_issue_count, len(issue_df)))
    checks.append(
        (
            "Change_Log (STATUS) row count",
            live_change_log_status_count,
            len(change_log_df),
        )
    )
    checks.append(("Issue_Link row count", live_issue_link_count, len(issue_link_df)))
    checks.append(("Comment row count", live_comment_count, len(comment_df)))

    print("\n-- Row count checks --")
    all_pass = True
    for name, live, cached in checks:
        ok = live == cached
        all_pass &= ok
        print(f"{name}: live={live} cached={cached} -> {'PASS' if ok else 'FAIL'}")

    # 2. No accidental row filtering
    print("\n-- Filtering sanity checks --")
    nunique_projects = issue_df["Project_ID"].nunique()
    proj_ok = nunique_projects == 39
    print(
        f"Distinct Project_ID count: {nunique_projects} -> {'PASS' if proj_ok else 'FAIL'}"
    )
    all_pass &= proj_ok

    n_null_resolution_date = issue_df["Resolution_Date"].isna().sum()
    unresolved_present = n_null_resolution_date > 0
    print(
        f"Issues with null Resolution_Date (unresolved present): {n_null_resolution_date} "
        f"-> {'PASS' if unresolved_present else 'FAIL'}"
    )
    all_pass &= unresolved_present

    # 3. Column presence and dtypes
    print("\n-- Dtypes --")
    print("Issue:")
    print(issue_df.dtypes)
    print("\nChange_Log:")
    print(change_log_df.dtypes)
    print("\nIssue_Link:")
    print(issue_link_df.dtypes)
    print("\nComment:")
    print(comment_df.dtypes)

    # 4. Null-rate spot check
    print("\n-- Null rates --")
    print("Issue:")
    print(issue_df.isna().mean())
    print("\nChange_Log:")
    print(change_log_df.isna().mean())
    print("\nIssue_Link:")
    print(issue_link_df.isna().mean())
    print("\nComment:")
    print(comment_df.isna().mean())

    change_log_issue_id_null = change_log_df["Issue_ID"].isna().sum()
    change_log_id_ok = change_log_issue_id_null == 0
    print(
        f"\nChange_Log null Issue_ID count: {change_log_issue_id_null} -> "
        f"{'PASS' if change_log_id_ok else 'FAIL'}"
    )
    all_pass &= change_log_id_ok

    # 5. File sizes
    print("\n-- File sizes --")
    for path in [issue_path, change_log_path, issue_link_path, comment_path]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{path}: {size_mb:.2f} MB")

    print(f"\n=== Overall: {'PASS' if all_pass else 'FAIL'} ===")


if __name__ == "__main__":
    main()
