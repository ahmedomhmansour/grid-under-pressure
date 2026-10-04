"""Clean and merge raw load + weather into one hourly modeling table.

Cleaning steps (each is logged so the README can report them):
  1. Reindex to a complete hourly UTC grid (exposes missing hours).
  2. Flag physically implausible values (<= 0 or > 2x the series median of
     daily maxima) and isolated spikes (|z| > 6 vs. a 24h rolling median).
  3. Fill gaps of <= 6 hours by time interpolation; longer gaps stay NaN and
     are dropped from model training rather than invented.
  4. Join weather and add local (America/New_York) time, which is what
     drives human behavior, while keeping UTC as the key (no DST duplicates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROCESSED, RAW, ensure_dirs

LOCAL_TZ = "America/New_York"


def clean_load(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = df.set_index("time_utc")["load_mw"].astype(float)
    s = s[~s.index.duplicated(keep="first")]
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    s = s.reindex(grid)
    log = {"hours_total": len(s), "missing_raw": int(s.isna().sum())}

    daily_max_med = s.resample("D").max().median()
    bad = (s <= 0) | (s > 2 * daily_max_med)
    roll = s.rolling(25, center=True, min_periods=5).median()
    resid = s - roll
    mad = resid.abs().rolling(24 * 14, center=True, min_periods=24).median() * 1.4826
    spikes = (resid.abs() > 6 * mad) & mad.gt(0)
    flagged = bad | spikes
    log["flagged_outliers"] = int(flagged.sum())
    s[flagged] = np.nan

    # interpolate, then un-fill any gap longer than 6h (pandas' `limit` would
    # partially fill long gaps, inventing data at their edges)
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    long_gap = isna & (run_len > 6)
    s = s.interpolate(method="time", limit_area="inside")
    s[long_gap] = np.nan
    log["missing_after_fill"] = int(s.isna().sum())
    out = s.rename("load_mw").to_frame()
    out.index.name = "time_utc"
    return out, log


def build_table(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.set_index("time_utc")[["temp_f", "ghi_wm2"]]
    w = w[~w.index.duplicated()]
    df = load.join(w, how="left")
    df["temp_f"] = df["temp_f"].interpolate(limit=3)
    df["ghi_wm2"] = df["ghi_wm2"].interpolate(limit=3).clip(lower=0)
    df["time_local"] = df.index.tz_convert(LOCAL_TZ)
    return df


def main() -> None:
    ensure_dirs()
    load_raw = pd.read_parquet(RAW / "eia930_dom_hourly.parquet")
    weather = pd.read_parquet(RAW / "weather_hourly.parquet")
    load, log = clean_load(load_raw)
    df = build_table(load, weather)
    df.to_parquet(PROCESSED / "hourly.parquet")
    print("Cleaning log:", log)
    print(f"Saved {len(df):,} hours → {PROCESSED / 'hourly.parquet'}")


if __name__ == "__main__":
    main()
