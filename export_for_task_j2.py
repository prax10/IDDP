"""
Second export for Task J. Run:  python export_for_task_j2.py
  1. prints Change_Log's columns + samples of PEOPLE / OTHER rows (so we see field names)
  2. data/raw/change_log_people.csv   - all PEOPLE rows (assignee / reporter changes)
  3. data/raw/change_log_sprint.csv   - OTHER rows that are sprint changes
"""
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://root:root%40123@127.0.0.1:3306/TAWOS_DB")
OUT = "data/raw"

with engine.connect() as conn:
    cols = pd.read_sql(text("SHOW COLUMNS FROM Change_Log"), conn)
    print("Change_Log columns:", list(cols["Field"]))
    colnames = list(cols["Field"])
    field_col = next((c for c in colnames if c.lower() in ("field", "field_name", "fieldname", "item_field")), None)
    print("field-name column:", field_col)

    sel = ", ".join(c for c in colnames if c.lower() not in ("id",))
    print("\nSample PEOPLE rows:")
    print(pd.read_sql(text(f"SELECT {sel} FROM Change_Log WHERE Change_Type='PEOPLE' LIMIT 5"), conn).to_string())
    print("\nSample OTHER rows:")
    print(pd.read_sql(text(f"SELECT {sel} FROM Change_Log WHERE Change_Type='OTHER' LIMIT 8"), conn).to_string())

    if field_col:
        print(f"\nTop {field_col} values inside OTHER and PEOPLE:")
        f = pd.read_sql(text(f"SELECT Change_Type, {field_col} AS f, COUNT(*) AS n FROM Change_Log "
                             f"WHERE Change_Type IN ('OTHER','PEOPLE') GROUP BY Change_Type, {field_col} "
                             f"ORDER BY n DESC LIMIT 40"), conn)
        print(f.to_string(index=False))
        f.to_csv(f"{OUT}/change_log_field_counts.csv", index=False)

    keep = [c for c in ["Issue_ID", "Change_Type", field_col, "From_String", "To_String", "Creation_Date"] if c]
    people = pd.read_sql(text(f"SELECT {', '.join(keep)} FROM Change_Log WHERE Change_Type='PEOPLE'"), conn)
    people.to_csv(f"{OUT}/change_log_people.csv", index=False)
    print(f"\nsaved {len(people)} PEOPLE rows -> {OUT}/change_log_people.csv")

    if field_col:
        sq = f"SELECT {', '.join(keep)} FROM Change_Log WHERE Change_Type='OTHER' AND LOWER({field_col}) LIKE '%sprint%'"
    else:  # no field column: sprint changes usually name a sprint in To/From
        sq = (f"SELECT {', '.join(keep)} FROM Change_Log WHERE Change_Type='OTHER' AND "
              f"(LOWER(To_String) LIKE '%sprint%' OR LOWER(From_String) LIKE '%sprint%')")
    sprint_cl = pd.read_sql(text(sq), conn)
    sprint_cl.to_csv(f"{OUT}/change_log_sprint.csv", index=False)
    print(f"saved {len(sprint_cl)} sprint-change rows -> {OUT}/change_log_sprint.csv")

print("\nDone. Tell Claude: exported")
