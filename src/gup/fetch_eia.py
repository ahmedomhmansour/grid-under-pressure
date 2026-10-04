"""Download hourly demand for PJM's Dominion subregion from EIA-930 (API v2).

Endpoint: electricity/rto/region-sub-ba-data — hourly demand by balancing-
authority subregion. Requires a free key from https://www.eia.gov/opendata/,
read from the EIA_API_KEY environment variable (never hard-code it).

Usage:
    python -m gup.fetch_eia --start 2019-01-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

from .config import RAW, ensure_dirs, load_config

URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
PAGE = 5000  # EIA v2 returns at most 5,000 rows per request


def _get(params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"EIA API error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("EIA API: retries exhausted")


def fetch_dom_demand(start: str, end: str, api_key: str, parent: str = "PJM",
                     subba: str = "DOM") -> pd.DataFrame:
    """Return a DataFrame indexed by UTC hour with column `load_mw`."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[parent][]": parent,
            "facets[subba][]": subba,
            "start": f"{start}T00",
            "end": f"{end}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE,
        }
        payload = _get(params)["response"]
        batch = payload.get("data", [])
        rows.extend(batch)
        total = int(payload.get("total", 0))
        offset += PAGE
        print(f"  fetched {min(offset, total):>7,} / {total:,} rows", end="\r")
        if offset >= total or not batch:
            break
        time.sleep(0.25)  # be polite
    print()
    if not rows:
        raise RuntimeError("EIA returned no rows — check the subba/parent facets and dates.")
    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["time_utc", "load_mw"]].sort_values("time_utc").reset_index(drop=True)


def _key_from_dotenv() -> str | None:
    """Read EIA_API_KEY from a .env file at the repo root (git-ignored)."""
    from .config import ROOT
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("EIA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg["study"]["history_start"])
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()

    key = os.environ.get("EIA_API_KEY") or _key_from_dotenv()
    if not key:
        raise SystemExit("No EIA key found. Put EIA_API_KEY=... in a .env file at the repo root "
                         "(see README → Setup).")
    ensure_dirs()
    df = fetch_dom_demand(args.start, args.end, key,
                          cfg["data"]["eia_parent"], cfg["data"]["eia_subba"])
    out = RAW / "eia930_dom_hourly.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
