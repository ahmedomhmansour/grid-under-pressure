"""Least-cost capacity expansion (linear program, PuLP + HiGHS).

Decision variables
  cap[k]          new capacity (MW), k in {gas_cc, gas_ct, solar, battery_4h, demand_response}
  g[r,d,h]        hourly dispatch of every resource r on representative day d, hour h
  imp, ens        imports and energy-not-served
  ch, dis, soc    battery charge, discharge, state of charge

Objective
  min  sum_k capex_k * cap_k  +  sum_d w_d * sum_h (fuel + O&M + carbon + VOLL * ens)

Constraints
  energy balance every hour;
  dispatch <= available capacity (solar <= CF x installed; existing tranches fixed);
  battery: soc_t = soc_{t-1} + eta*ch - dis, soc <= duration*cap, day-cyclic;
  reliability requirement (PJM-style, in accredited/UCAP MW):
      sum_k ELCC_k * capacity_k + accredited imports >= FPR * peak.

The reserve constraint is what makes this a *planning* problem: it values
capacity by how much it can be counted on at peak (accreditation), not just
by the energy it produces. Its dual is the marginal value of capacity.
"""
from __future__ import annotations

import numpy as np
import pulp

from .config import annualized_capex

NEW_TECHS = ["gas_cc", "gas_ct", "solar", "battery_4h", "demand_response"]
THERMAL_NEW = ["gas_cc", "gas_ct"]


def _solver():
    if pulp.HiGHS().available():
        return pulp.HiGHS(msg=False)
    if pulp.PULP_CBC_CMD().available():
        return pulp.PULP_CBC_CMD(msg=False)
    raise RuntimeError("No LP solver found: `pip install highspy`.")


def existing_firm_mw(cfg: dict) -> float:
    return float(sum(t["mw"] for t in cfg["system"]["existing_fleet"].values()))


def _unit(spec: dict, gas_price: float, gas_co2: float) -> tuple[float, float]:
    """(fuel+O&M $/MWh, t CO2/MWh) for a resource spec with either heat_rate or var_cost."""
    if "heat_rate" in spec:
        return gas_price * spec["heat_rate"] + spec.get("vom", 0.0), spec["heat_rate"] * gas_co2
    return spec.get("var_cost", spec.get("vom", 0.0)), spec.get("co2_t_per_mwh", 0.0)


def resource_costs(cfg: dict, gas_price: float, carbon_price: float) -> dict:
    """Return {resource: (variable $/MWh incl. carbon, t CO2/MWh)} for every dispatchable resource."""
    gco2 = cfg["gas_co2_t_per_mmbtu"]
    sysc, tech = cfg["system"], cfg["technologies"]
    out = {}
    for name, spec in sysc["existing_fleet"].items():
        c, e = _unit(spec, gas_price, gco2)
        out[f"existing_{name}"] = (c + carbon_price * e, e)
    for k in THERMAL_NEW:
        c, e = _unit(tech[k], gas_price, gco2)
        out[k] = (c + carbon_price * e, e)
    imp_c = sysc["import_price_adder"] + sysc["import_heat_rate"] * gas_price
    imp_e = sysc["import_co2_t_per_mwh"]
    out["imports"] = (imp_c + carbon_price * imp_e, imp_e)
    out["demand_response"] = (tech["demand_response"]["vom"], 0.0)
    return out


def solve(cfg: dict, load: np.ndarray, solar_cf: np.ndarray, weights: np.ndarray,
          peak_mw: float, gas_price: float | None = None, carbon_price: float | None = None,
          allowed: list[str] | None = None, fpr: float | None = None) -> dict:
    """Solve the LP. `load`, `solar_cf` are [days, 24]; `weights` = days/yr each rep day stands for."""
    gas_price = cfg["gas_price"] if gas_price is None else gas_price
    carbon_price = cfg.get("carbon_price", 0.0) if carbon_price is None else carbon_price
    fpr = cfg["system"]["reliability_requirement_factor"] if fpr is None else fpr
    allowed = NEW_TECHS if allowed is None else list(allowed)
    sysc, tech = cfg["system"], cfg["technologies"]
    costs = resource_costs(cfg, gas_price, carbon_price)
    D, H = load.shape
    DH = [(d, h) for d in range(D) for h in range(H)]

    p = pulp.LpProblem("capacity_expansion", pulp.LpMinimize)
    cap = {k: pulp.LpVariable(f"cap_{k}", lowBound=0,
                              upBound=(tech[k].get("max_mw") if k in allowed else 0))
           for k in NEW_TECHS}
    lv = lambda name: pulp.LpVariable.dicts(name, DH, lowBound=0)  # noqa: E731
    disp = {r: lv(f"g_{r}") for r in costs}             # every dispatchable incl. imports & DR
    g_pv, ens, ch, dis, soc = lv("g_pv"), lv("ens"), lv("ch"), lv("dis"), lv("soc")

    eta = tech["battery_4h"]["round_trip_eff"]
    dur = tech["battery_4h"]["duration_h"]
    pv_total = sysc["existing_solar_mw"] + cap["solar"]
    limit = {f"existing_{n}": s["mw"] for n, s in sysc["existing_fleet"].items()}
    limit.update({"gas_cc": cap["gas_cc"], "gas_ct": cap["gas_ct"],
                  "imports": sysc["import_limit_mw"], "demand_response": cap["demand_response"]})

    fixed = pulp.lpSum(annualized_capex(cfg, k) * 1000 * cap[k] for k in NEW_TECHS)
    variable = pulp.lpSum(
        weights[d] * (pulp.lpSum(costs[r][0] * disp[r][d, h] for r in costs)
                      + tech["battery_4h"]["vom"] * dis[d, h] + sysc["voll"] * ens[d, h])
        for d, h in DH)
    p += fixed + variable

    for d, h in DH:
        p += (pulp.lpSum(disp[r][d, h] for r in costs) + g_pv[d, h] + dis[d, h] - ch[d, h]
              + ens[d, h] == float(load[d, h]))
        for r in costs:
            p += disp[r][d, h] <= limit[r]
        p += g_pv[d, h] <= float(solar_cf[d, h]) * pv_total
        p += ch[d, h] <= cap["battery_4h"]
        p += dis[d, h] <= cap["battery_4h"]
        p += soc[d, h] <= dur * cap["battery_4h"]
        p += soc[d, h] == soc[d, (h - 1) % H] + eta * ch[d, h] - dis[d, h]  # day-cyclic

    accredited = (existing_firm_mw(cfg)
                  + tech["solar"]["capacity_credit"] * pv_total
                  + sysc["import_capacity_credit"] * sysc["import_limit_mw"]
                  + pulp.lpSum(tech[k]["capacity_credit"] * cap[k] for k in NEW_TECHS if k != "solar"))
    p += accredited >= fpr * peak_mw, "planning_reserve"

    status = p.solve(_solver())
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[status]}")

    val = lambda x: float(pulp.value(x) or 0.0)  # noqa: E731
    wsum = lambda var: float(sum(weights[d] * val(var[d, h]) for d, h in DH))  # noqa: E731
    energy = {r: wsum(disp[r]) for r in costs}
    energy.update({"solar": wsum(g_pv), "battery_discharge": wsum(dis), "unserved": wsum(ens)})
    co2 = sum(energy[r] * costs[r][1] for r in costs)
    demand = float(sum(weights[d] * load[d].sum() for d in range(D)))
    total = val(p.objective)
    return {
        "new_capacity_mw": {k: round(val(cap[k]), 1) for k in NEW_TECHS},
        "annual_cost_musd": round(total / 1e6, 1),
        "fixed_cost_musd": round(val(fixed) / 1e6, 1),
        "variable_cost_musd": round(val(variable) / 1e6, 1),
        "energy_twh": {k: round(v / 1e6, 3) for k, v in energy.items()},
        "demand_twh": round(demand / 1e6, 2),
        "avg_cost_usd_per_mwh": round(total / demand, 2),
        "co2_mt": round(co2 / 1e6, 2),
        "co2_t_per_mwh": round(co2 / demand, 3),
        # dual of the reserve constraint = marginal cost of one more MW of accredited capacity
        "reserve_shadow_price_usd_per_kw_yr": round(
            abs(p.constraints["planning_reserve"].pi or 0.0) / 1000, 2),
        "gas_price": gas_price,
        "carbon_price": carbon_price,
        "fpr": fpr,
        "accredited_requirement_mw": round(fpr * peak_mw, 0),
        "allowed": allowed,
    }
