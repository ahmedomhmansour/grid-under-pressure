"""Unit and integration tests. Run with `pytest -q` (no API key or network needed)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from gup import forecast, optimize, reliability, repdays, scenarios
from gup.clean import clean_load
from gup.config import load_config
from gup.synthetic import make_hourly_table


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def table():
    return make_hourly_table("2019-01-01", "2023-01-01")


@pytest.fixture(scope="module")
def ensemble(table):
    ens, _ = forecast.weather_year_ensemble(table, base_year=2022)
    return ens


# ---------- cleaning ----------------------------------------------------------
def test_clean_fills_short_gaps_only():
    t = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    load = 10000 + 1000 * np.sin(np.arange(len(t)) * 2 * np.pi / 24)
    df = pd.DataFrame({"time_utc": t, "load_mw": load})
    df = df.drop(index=range(100, 104))              # 4-hour gap -> filled
    df = df.drop(index=range(300, 320))              # 20-hour gap -> left missing
    out, log = clean_load(df)
    assert len(out) == len(t)
    assert out["load_mw"].iloc[100:104].notna().all()
    assert out["load_mw"].iloc[300:320].isna().all()
    assert log["missing_raw"] == 24


def test_clean_flags_spike():
    t = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    load = 10000 + 1000 * np.sin(np.arange(len(t)) * 2 * np.pi / 24)
    load[200] = 60000
    out, log = clean_load(pd.DataFrame({"time_utc": t, "load_mw": load}))
    assert log["flagged_outliers"] >= 1
    assert out["load_mw"].iloc[200] < 12000


# ---------- forecasting -------------------------------------------------------
def test_backtest_beats_linear_benchmark(table):
    bt = forecast.backtest(table, test_year=2022)
    assert bt["gbm"]["mape_pct"] < 5
    assert bt["gbm"]["mape_pct"] < bt["linear_benchmark"]["mape_pct"]


def test_ensemble_shape(ensemble):
    assert ensemble.groupby("weather_year").size().eq(8760).all()
    assert ensemble["solar_cf"].between(0, 1).all()


# ---------- scenarios ---------------------------------------------------------
def test_increment_matches_growth_formula(ensemble):
    bp = scenarios.base_peak(ensemble)
    inc = scenarios.increment_mw(ensemble, 0.05, 5)
    assert inc == pytest.approx(bp * (1.05 ** 5 - 1))
    tl = scenarios.target_load(ensemble, 0.05, 5)
    assert (tl["load_mw"] - ensemble["load_mw"]).round(6).eq(round(inc, 6)).all()


# ---------- optimization ------------------------------------------------------
def _tiny_case(peak=36000.0):
    hours = np.arange(24)
    load = (31000 + 5000 * np.sin((hours - 9) * np.pi / 12)).reshape(1, 24)
    solar = np.clip(np.sin((hours - 6) * np.pi / 12), 0, None).reshape(1, 24) * 0.8
    return load, solar, np.array([365.0]), peak


def test_lp_meets_reserve_and_balance(cfg):
    load, solar, w, peak = _tiny_case()
    sol = optimize.solve(cfg, load, solar, w, peak)
    tech = cfg["technologies"]
    accredited = (optimize.existing_firm_mw(cfg)
                  + cfg["system"]["import_capacity_credit"] * cfg["system"]["import_limit_mw"]
                  + tech["solar"]["capacity_credit"] * (cfg["system"]["existing_solar_mw"] + sol["new_capacity_mw"]["solar"])
                  + sum(tech[k]["capacity_credit"] * sol["new_capacity_mw"][k]
                        for k in optimize.NEW_TECHS if k != "solar"))
    assert accredited >= cfg["system"]["reliability_requirement_factor"] * peak - 1
    assert sol["energy_twh"]["unserved"] == pytest.approx(0, abs=1e-6)
    supplied = sum(v for k, v in sol["energy_twh"].items() if k != "unserved")
    assert supplied >= sol["demand_twh"] - 0.01           # battery losses mean supply >= demand
    assert sol["new_capacity_mw"]["demand_response"] <= tech["demand_response"]["max_mw"] + 1e-6


def test_lp_respects_disallowed_techs(cfg):
    load, solar, w, peak = _tiny_case()
    sol = optimize.solve(cfg, load, solar, w, peak, allowed=["gas_ct"])
    assert all(sol["new_capacity_mw"][k] == 0 for k in optimize.NEW_TECHS if k != "gas_ct")


def test_lp_builds_solar_when_gas_and_carbon_are_expensive(cfg):
    c = copy.deepcopy(cfg)
    c["technologies"]["solar"]["overnight_cost"] = 300
    load, solar, w, peak = _tiny_case()
    sol = optimize.solve(c, load, solar, w, peak, gas_price=8, carbon_price=80)
    assert sol["new_capacity_mw"]["solar"] > 0


def test_optimized_never_costs_more_than_gas_only(cfg, ensemble):
    tgt = scenarios.target_load(ensemble, 0.054, 5)
    L, S, W = repdays.select(tgt, 8)
    peak = scenarios.summary(ensemble, 0.054, 5)["peak_median_mw"]
    full = optimize.solve(cfg, L, S, W, peak)
    gas = optimize.solve(cfg, L, S, W, peak, allowed=["gas_cc", "gas_ct"])
    assert full["annual_cost_musd"] <= gas["annual_cost_musd"] + 0.1


# ---------- representative days -----------------------------------------------
def test_repday_weights_sum_to_a_year(ensemble):
    _, _, w = repdays.select(ensemble, 10)
    assert w.sum() == pytest.approx(365, abs=0.5)


# ---------- finance -----------------------------------------------------------
def test_crf_known_value():
    from gup.config import crf
    assert crf(0.075, 30) == pytest.approx(0.08467, abs=1e-4)


# ---------- reliability -------------------------------------------------------
def test_reliability_extremes(cfg, ensemble):
    big = reliability.simulate(cfg, scenarios.target_load(ensemble, 0.0, 1),
                               {"gas_cc": 30000}, draws=50)
    assert big["lole_days_per_yr"] == 0
    short = reliability.simulate(cfg, scenarios.target_load(ensemble, 0.10, 6), {}, draws=50)
    assert short["lole_days_per_yr"] > 1
    assert not short["meets_target"]
