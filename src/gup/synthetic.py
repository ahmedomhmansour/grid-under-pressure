"""Synthetic stand-in data for tests and offline demos.

NOT real data. It only exists so the pipeline, tests and app run without an
API key (CI, a fresh clone). Every output produced from it is stamped
"DEMO DATA". Real results must come from `gup.fetch_eia` + `gup.fetch_weather`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_weather(start: str = "2019-01-01", end: str = "2025-12-31", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, f"{end} 23:00", freq="h", tz="UTC")
    local = t.tz_convert("America/New_York")
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()
    n = len(t)
    # temperature: seasonal + diurnal + persistent weather anomalies (AR(1))
    seasonal = 58 - 20 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    diurnal = -8 * np.cos(2 * np.pi * (hour - 3) / 24)
    anom = np.zeros(n)
    eps = rng.normal(0, 0.9, n)
    for i in range(1, n):
        anom[i] = 0.985 * anom[i - 1] + eps[i]
    temp = seasonal + diurnal + anom
    # irradiance: clear-sky by day length x random daily cloudiness
    decl = 23.45 * np.sin(2 * np.pi * (284 + doy) / 365)
    lat = np.radians(38)
    ha = np.radians(15 * (hour + 0.5 - 12))
    sin_elev = (np.sin(lat) * np.sin(np.radians(decl))
                + np.cos(lat) * np.cos(np.radians(decl)) * np.cos(ha))
    clear = 1000 * np.clip(sin_elev, 0, None) ** 1.15
    days = local.normalize()
    cloud_by_day = pd.Series(rng.beta(2.2, 1.6, len(days.unique())), index=days.unique())
    cloud = cloud_by_day.reindex(days).to_numpy()
    ghi = clear * (0.25 + 0.75 * cloud)
    return pd.DataFrame({"time_utc": t, "temp_f": temp, "ghi_wm2": ghi})


def make_load(weather: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = weather["time_utc"]
    local = pd.DatetimeIndex(t).tz_convert("America/New_York")
    temp = weather["temp_f"].to_numpy()
    years = (local - local[0]).total_seconds().to_numpy() / (365.25 * 86400)
    base = 9800 * (1.045 ** years)                      # data-center-driven trend
    cool = 330 * np.clip(temp - 66, 0, None)
    heat = 290 * np.clip(56 - temp, 0, None)
    hour = local.hour.to_numpy()
    shape = 1 + 0.13 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.05 * np.sin(4 * np.pi * (hour - 6) / 24)
    weekend = np.where(local.dayofweek.to_numpy() >= 5, 0.94, 1.0)
    load = (base * shape * weekend) + cool + heat + rng.normal(0, 180, len(t))
    return pd.DataFrame({"time_utc": t, "load_mw": load})


def make_hourly_table(start: str = "2019-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Return a table shaped exactly like data/processed/hourly.parquet."""
    from .clean import build_table, clean_load
    w = make_weather(start, end)
    load, _ = clean_load(make_load(w))
    return build_table(load, w)
