"""
One-off export for Task J (Scrum mode + reassignment features).
Run from the capstone folder with the venv active:
    python export_for_task_j.py
Writes into data/raw/:
    change_type_counts.csv   - every Change_Type in Change_Log with its row count
    sprint.csv               - the full Sprint table
    change_log_other.csv     - Change_Log rows for assignee / sprint / story points /
                               priority / fix version / estimate changes
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text

OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)

# Try TCP on 127.0.0.1 and localhost, then the Unix socket (MySQL DMG default /tmp/mysql.sock)
CANDIDATES = [
    "mysql+pymysql://root:root%40123@127.0.0.1:3306/TAWOS_DB",
    "mysql+pymysql://root:root%40123@localhost/TAWOS_DB",
    "mysql+pymysql://root:root%40123@localhost/TAWOS_DB?unix_socket=/tmp/mysql.sock",
]
engine = None
for url in CANDIDATES:
    try:
        e = create_engine(url)
        with e.connect() as c:
            c.execute(text("SELECT 1"))
        engine = e
        print(f"connected via: {url.split('@')[1]}")
        break
    except Exception as ex:
        print(f"  could not connect via {url.split('@')[1]}: {type(ex).__name__}")
if engine is None:
    raise SystemExit("\nMySQL is not accepting connections. Run:  /usr/local/mysql/support-files/mysql.server status\n"
                     "If it says 'not running', start it again and wait ~10 seconds before retrying.")

with engine.connect() as conn:
    print("1/3 Change_Type counts ...")
    counts = pd.read_sql(text("SELECT Change_Type, COUNT(*) AS n FROM Change_Log GROUP BY Change_Type ORDER BY n DESC"), conn)
    counts.to_csv(f"{OUT}/change_type_counts.csv", index=False)
    print(counts.head(30).to_string(index=False))

    print("\n2/3 Sprint table ...")
    sprint = pd.read_sql(text("SELECT * FROM Sprint"), conn)
    sprint.to_csv(f"{OUT}/sprint.csv", index=False)
    print(f"  {len(sprint)} sprints, columns: {list(sprint.columns)}")

    print("\n3/3 Change_Log (assignee / sprint / story points / priority / fix version / estimate) ...")
    q = text("""
        SELECT Issue_ID, Change_Type, From_String, To_String, Creation_Date
        FROM Change_Log
        WHERE LOWER(Change_Type) LIKE '%assign%'
           OR LOWER(Change_Type) LIKE '%sprint%'
           OR LOWER(Change_Type) LIKE '%story%'
           OR LOWER(Change_Type) LIKE '%priority%'
           OR LOWER(Change_Type) LIKE '%fix version%'
           OR LOWER(Change_Type) LIKE '%estimate%'
    """)
    other = pd.read_sql(q, conn)
    other.to_csv(f"{OUT}/change_log_other.csv", index=False)
    print(f"  {len(other)} rows")
    print(other["Change_Type"].value_counts().to_string())

print("\nDone. Files are in data/raw/. Tell Claude: exported")
