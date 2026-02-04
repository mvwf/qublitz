from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to Python path so "utils" imports work reliably
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.student_assignment import build_assignments, DEFAULT_SECRET  # noqa: E402


def main():
    csv_path = REPO_ROOT / "net_IDs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Are you in the right repo?")

    assignments = build_assignments(
        csv_path=csv_path,
        column="SIS Login ID",
        secret=DEFAULT_SECRET,
        api_key_mode="netid",  # simplest debug: API key = netid
    )

    print(f"Found {len(assignments)} students.\n")
    print("First 20 valid API keys (NetID mode):")
    for a in assignments[:]:
        print(a.api_key)


if __name__ == "__main__":
    main()
