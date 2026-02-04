from pathlib import Path
import os
import pandas as pd

from utils.student_assignment import build_assignments

REPO_ROOT = Path(__file__).resolve().parents[1]

secret = os.environ["ASSIGNMENT_SECRET"]
assignments = build_assignments(
    csv_path=REPO_ROOT / "net_IDs.csv",
    column="SIS Login ID",
    secret=secret,
    api_key_mode="derived",
)

rows = [{"netid": a.netid, "api_key": a.api_key} for a in assignments]
df = pd.DataFrame(rows).sort_values("netid")
df.to_csv("api_keys_out.csv", index=False)
print("Wrote api_keys_out.csv")
