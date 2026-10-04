"""Download hourly temperature and solar irradiance from the Open-Meteo
historical archive (ERA5 reanalysis; free, no API key).

Two stations (Richmond, Dulles) are averaged with the weights in
config/assumptions.yaml to approximate weather over Dominion's load centers.

Usage:
    python -m gup.fetch_weather --start 2019-01-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from .config import RAW, ensure_dirs, load_config

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = "temperature_2m,shortwave_radiation"


def fetch_station(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": VARS,
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    for attempt in range(4):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            h = r.json()["hourly"]
            return pd.DataFrame({
                "time_utc": pd.to_datetime(h["time"], utc=True),
                "temp_f": h["temperature_2m"],
                "ghi_wm2": h["shortwave_radiation"],
            })
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo error {r.status_code}: {r.text[:300]}")


def fetch_weighted(stations: dict, start: str, end: str) -> pd.DataFrame:
    frames = []
    for name, s in stations.items():
        print(f"  {name} ({s['lat']}, {s['lon']})")
        df = fetch_station(s["lat"], s["lon"], start, end).set_index("time_utc")
        frames.append(df * s["weight"])
    total_w = sum(s["weight"] for s in stations.values())
    out = sum(frames) / total_w
    return out.reset_index()


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=cfg["study"]["history_start"])
    ap.add_argument("--end", default=(pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    ensure_dirs()
    df = fetch_weighted(cfg["data"]["weather_stations"], args.start, args.end)
    out = RAW / "weather_hourly.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
