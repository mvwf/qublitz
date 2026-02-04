from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

NETID_COLUMN_DEFAULT = "SIS Login ID"

# Ranges in GHz and ns (match your app expectations!)
PARAM_RANGES = {
    "omega_q": (4.0, 7.0),         # GHz
    "omega_rabi": (0.02, 0.25),    # GHz
    "T1": (20.0, 200.0),           # ns  (set whatever you want)
}

@dataclass(frozen=True)
class Assignment:
    netid: str
    api_key: str
    params: Dict[str, float]
    version: str = "v1"


def _stable_seed(netid: str, secret: str) -> int:
    msg = (secret + "::" + netid.strip().lower()).encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_api_key(netid: str, secret: str) -> str:
    msg = (secret + "::APIKEY::" + netid.strip().lower()).encode("utf-8")
    return hashlib.sha256(msg).hexdigest()[:24]


def read_netids_csv(csv_path: str | Path, column: str = NETID_COLUMN_DEFAULT) -> List[str]:
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"netIDs CSV not found: {p}")
    df = pd.read_csv(p)
    if column not in df.columns:
        raise ValueError(f"CSV missing column {column!r}. Found columns: {list(df.columns)}")

    netids = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
        .dropna()
        .unique()
        .tolist()
    )
    return netids


def generate_params_for_netid(netid: str, secret: str) -> Dict[str, float]:
    if not secret:
        raise ValueError("ASSIGNMENT_SECRET must be non-empty")

    rng = np.random.default_rng(_stable_seed(netid, secret))
    params: Dict[str, float] = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        params[k] = float(rng.uniform(lo, hi))
    return params


def build_assignments(
    csv_path: str | Path,
    column: str,
    secret: str,
    api_key_mode: str = "derived",  # "derived" or "netid"
) -> List[Assignment]:
    netids = read_netids_csv(csv_path=csv_path, column=column)
    out: List[Assignment] = []
    for netid in netids:
        api_key = netid if api_key_mode == "netid" else derive_api_key(netid, secret)
        params = generate_params_for_netid(netid, secret=secret)
        out.append(Assignment(netid=netid, api_key=api_key, params=params))
    return out


def lookup_assignment_by_api_key(assignments: List[Assignment], api_key: str) -> Optional[Assignment]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    for a in assignments:
        if a.api_key == api_key:
            return a
    return None


def assignment_to_user_data(a: Assignment) -> Dict[str, float | str]:
    payload: Dict[str, float | str] = {"user": a.netid, "version": a.version}
    payload.update(a.params)
    return payload
