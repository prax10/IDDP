"""Third (last) export for Task J. Run:  python export_for_task_j3.py
Saves data/raw/change_log_fields.csv: priority / issuetype / Rank / Link / Fix Version /
Epic Link / timeestimate / timeoriginalestimate / labels / Component changes.
Only timestamps + field name + issue (no long text) to keep the file small."""
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("mysql+pymysql://root:root%40123@127.0.0.1:3306/TAWOS_DB")
FIELDS = ["priority", "issuetype", "Rank", "Link", "Fix Version", "Epic Link",
          "timeestimate", "timeoriginalestimate", "labels", "Component"]
q = text("SELECT Issue_ID, Field, From_String, To_String, Creation_Date FROM Change_Log "
         "WHERE Change_Type='OTHER' AND Field IN :f AND Issue_ID IS NOT NULL").bindparams(
    __import__("sqlalchemy").bindparam("f", expanding=True))
with engine.connect() as conn:
    df = pd.read_sql(q, conn, params={"f": FIELDS})
df["From_String"] = df["From_String"].astype(str).str.slice(0, 80)
df["To_String"] = df["To_String"].astype(str).str.slice(0, 80)
df.to_csv("data/raw/change_log_fields.csv.gz", index=False, compression="gzip")
print(df["Field"].value_counts().to_string())
print(f"\nsaved {len(df)} rows -> data/raw/change_log_fields.csv.gz\nDone. Tell Claude: exported, go")
