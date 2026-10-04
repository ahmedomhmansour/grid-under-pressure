"""Turn the base-year weather-year ensemble into target-year load scenarios.

Method:
  * weather-normal base peak = median annual peak across weather years
  * target peak               = base peak x (1 + g) ^ (target_year - base_year)
  * the increment is added as flat, round-the-clock load (data centers run at
    ~80-90%+ load factors, so a flat block is a reasonable first-order model;
    see README "Limitations")
"""
from __future__ import annotations

import pandas as pd


def base_peak(ens: pd.DataFrame) -> float:
    return float(ens.groupby("weather_year")["load_mw"].max().median())


def increment_mw(ens: pd.DataFrame, growth: float, years: int) -> float:
    bp = base_peak(ens)
    return bp * ((1 + growth) ** years - 1)


def target_load(ens: pd.DataFrame, growth: float, years: int) -> pd.DataFrame:
    """Return the ensemble with `load_mw` lifted to the target year."""
    out = ens.copy()
    out["load_mw"] = out["load_mw"] + increment_mw(ens, growth, years)
    return out


def summary(ens: pd.DataFrame, growth: float, years: int) -> dict:
    tl = target_load(ens, growth, years)
    peaks = tl.groupby("weather_year")["load_mw"].max()
    return {
        "growth": growth,
        "increment_mw": round(increment_mw(ens, growth, years), 0),
        "peak_median_mw": round(float(peaks.median()), 0),
        "peak_p90_mw": round(float(peaks.quantile(0.9)), 0),
        "energy_twh": round(float(tl.groupby("weather_year")["load_mw"].sum().mean() / 1e6), 2),
    }
