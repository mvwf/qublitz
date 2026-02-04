from __future__ import annotations

import sys
from pathlib import Path
import pprint

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.student_assignment import build_assignments, DEFAULT_SECRET


def main():
    assignments = build_assignments(
        csv_path=REPO_ROOT / "net_IDs.csv",
        column="SIS Login ID",
        secret=DEFAULT_SECRET,
        api_key_mode="netid",  # NetID = API key (debug mode)
    )

    print(f"\nLoaded {len(assignments)} assignments\n")

    for a in assignments:
        print("=" * 60)
        print(f"NetID / API key: {a.api_key}")
        print("Parameters:")
        pprint.pprint(a.params)
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
