"""Monte Carlo resource-adequacy check for a capacity portfolio.

Each draw is one simulated year:
  * a historical weather year (load shape + solar output) chosen at random,
  * a random error on the annual load level (forecast uncertainty),
  * random forced outages of thermal units, re-sampled every day,
  * extra correlated outages in very cold hours (the Winter Storm Elliott lesson),
  * battery dispatched greedily (discharge into shortfalls, recharge from surplus).

Metrics
  LOLE  loss-of-load expectation, days/yr with any unserved energy (target 0.1)
  LOLH  hours/yr with unserved energy
  EUE   expected unserved energy, MWh/yr
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(cfg: dict, ens_target: pd.DataFrame, new_mw: dict, draws: int | None = None,
             seed: int = 11) -> dict:
    rc, sysc, tech = cfg["reliability"], cfg["system"], cfg["technologies"]
    N = draws or rc["draws"]
    rng = np.random.default_rng(seed)

    years = sorted(ens_target["weather_year"].unique())
    load_by_year = {y: g["load_mw"].to_numpy() for y, g in ens_target.groupby("weather_year")}
    cf_by_year = {y: g["solar_cf"].to_numpy() for y, g in ens_target.groupby("weather_year")}
    temp_by_year = {y: g["temp_f"].to_numpy() for y, g in ens_target.groupby("weather_year")}
    pick = rng.choice(years, size=N)
    load = np.stack([load_by_year[y] for y in pick])           # [N, 8760]
    cf = np.stack([cf_by_year[y] for y in pick])
    load *= (1 + rng.normal(0, rc["load_forecast_sigma"], N))[:, None]
    H = load.shape[1]
    days = H // 24

    def fleet(total_mw: float, unit: float, f_or: float) -> np.ndarray:
        n = int(round(total_mw / unit))
        if n == 0:
            return np.zeros((N, H))
        up = rng.binomial(n, 1 - f_or, size=(N, days)) * (total_mw / n)
        return np.repeat(up, 24, axis=1)

    # Existing capacity is given in accredited (UCAP) MW; convert back to installed MW
    # with each tranche's ELCC rating, then apply random independent outages here and
    # correlated cold-weather outages below.
    existing_icap = sum(t["mw"] / t.get("elcc", 1.0) for t in sysc["existing_fleet"].values())
    thermal = (fleet(existing_icap, rc["existing_unit_size_mw"], rc["existing_forced_outage_rate"])
               + fleet(new_mw.get("gas_cc", 0), tech["gas_cc"]["unit_size_mw"], tech["gas_cc"]["forced_outage_rate"])
               + fleet(new_mw.get("gas_ct", 0), tech["gas_ct"]["unit_size_mw"], tech["gas_ct"]["forced_outage_rate"]))
    cw = rc.get("cold_weather")
    if cw:
        temp = np.stack([temp_by_year[y] for y in pick])
        extra = np.clip((cw["threshold_f"] - temp) * cw["extra_outage_per_degf"], 0, cw["max_extra_outage"])
        thermal = thermal * (1 - extra)
    solar = cf * (sysc["existing_solar_mw"] + new_mw.get("solar", 0))
    other = sysc["import_limit_mw"] * rc.get("import_availability", 1.0) + new_mw.get("demand_response", 0)
    margin = thermal + solar + other - load                     # [N, H]

    P = new_mw.get("battery_4h", 0.0)
    E = P * tech["battery_4h"]["duration_h"]
    eta = tech["battery_4h"]["round_trip_eff"]
    short = np.zeros_like(margin)
    soc = np.full(N, E)
    for h in range(H):
        m = margin[:, h]
        if P > 0:
            dis = np.minimum(np.minimum(np.clip(-m, 0, None), P), soc)
            chg = np.minimum(np.minimum(np.clip(m, 0, None), P), (E - soc) / eta)
            soc = soc - dis + eta * chg
            m = m + dis
        short[:, h] = np.clip(-m, 0, None)

    lol_hours = (short > 0).sum(axis=1)
    lol_days = (short.reshape(N, days, 24).max(axis=2) > 0).sum(axis=1)
    eue = short.sum(axis=1)
    return {
        "draws": N,
        "lole_days_per_yr": round(float(lol_days.mean()), 3),
        "lolh_hours_per_yr": round(float(lol_hours.mean()), 2),
        "eue_mwh_per_yr": round(float(eue.mean()), 0),
        "p_any_shortfall": round(float((lol_hours > 0).mean()), 3),
        "meets_target": bool(lol_days.mean() <= rc["lole_target_days"]),
    }
