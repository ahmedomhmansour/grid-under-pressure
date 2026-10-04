"""Grid Under Pressure — interactive planner.

    streamlit run app/streamlit_app.py

Re-solves the capacity-expansion LP and the Monte Carlo reliability check
live as you move the sliders. Needs outputs/ensemble_base_year.parquet,
produced by `python -m gup.pipeline` (or `--demo`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gup import optimize, reliability, repdays, scenarios  # noqa: E402
from gup.config import FIGURES, OUTPUTS, load_config  # noqa: E402
from gup.figures import TECH_COLORS, TECH_LABELS  # noqa: E402

st.set_page_config(page_title="Grid Under Pressure", page_icon="⚡", layout="wide")
cfg = load_config()


@st.cache_data
def load_outputs():
    ens = pd.read_parquet(OUTPUTS / "ensemble_base_year.parquet")
    with open(OUTPUTS / "results.json", encoding="utf-8") as f:
        res = json.load(f)
    return ens, res


@st.cache_data(show_spinner=False)
def plan(growth, gas, carbon, rm, allowed, draws):
    ens, _ = load_outputs()
    yrs = cfg["study"]["target_year"] - cfg["study"]["base_year"]
    tgt = scenarios.target_load(ens, growth, yrs)
    summ = scenarios.summary(ens, growth, yrs)
    L, S, W = repdays.select(tgt, cfg["optimization"]["rep_days"], cfg["optimization"]["seed"])
    sol = optimize.solve(cfg, L, S, W, summ["peak_median_mw"], gas_price=gas, carbon_price=carbon,
                         allowed=list(allowed), fpr=rm)
    gas_only = optimize.solve(cfg, L, S, W, summ["peak_median_mw"], gas_price=gas, carbon_price=carbon,
                              allowed=["gas_cc", "gas_ct"], fpr=rm)
    rel = reliability.simulate(cfg, tgt, sol["new_capacity_mw"], draws)
    return summ, sol, gas_only, rel


try:
    ens, res = load_outputs()
except FileNotFoundError:
    st.error("No model outputs found. Run `python -m gup.pipeline` (or `--demo`) first.")
    st.stop()

st.title("⚡ Grid Under Pressure")
st.caption(f"How should Dominion's PJM zone meet data-center load growth by {cfg['study']['target_year']}? "
           "Forecast → least-cost capacity plan (LP) → Monte Carlo reliability check.")
if res.get("demo_data"):
    st.warning("DEMO DATA — this deployment is running on synthetic data. Numbers are illustrative only.")

with st.sidebar:
    st.header("Scenario levers")
    growth = st.slider("Peak-load growth (%/yr)", 0.0, 10.0,
                       cfg["scenarios"]["base"]["peak_growth"] * 100, 0.1,
                       help="PJM's 2026 forecast: ~3.9%/yr summer, ~4.7%/yr winter (10-yr, DOM zone).") / 100
    gas = st.slider("Gas price ($/MMBtu)", 2.0, 8.0, float(cfg["gas_price"]), 0.25)
    carbon = st.slider("Carbon price ($/t CO₂)", 0, 100, int(cfg.get("carbon_price", 0)), 5)
    rm = st.slider("Reliability requirement (accredited MW ÷ peak)", 0.80, 1.10,
                   float(cfg["system"]["reliability_requirement_factor"]), 0.01,
                   help="PJM's forecast pool requirement (FPR) for 2028/29 is 0.9401: accredited "
                        "capacity (after ELCC derating) must cover ~94% of forecast peak.")
    st.header("Allowed new resources")
    allowed = tuple(k for k in optimize.NEW_TECHS if st.checkbox(TECH_LABELS[k], value=True))
    draws = st.select_slider("Monte Carlo years", [200, 500, 1000], value=1000)

with st.spinner("Solving LP and running Monte Carlo…"):
    summ, sol, gas_only, rel = plan(growth, gas, carbon, rm, allowed, draws)

new_total = sum(sol["new_capacity_mw"].values())
saving = (gas_only["annual_cost_musd"] - sol["annual_cost_musd"]) / gas_only["annual_cost_musd"] * 100
c = st.columns(5)
c[0].metric(f"{cfg['study']['target_year']} peak (weather-normal)", f"{summ['peak_median_mw']/1000:.1f} GW",
            f"+{summ['increment_mw']/1000:.1f} GW vs base year", delta_color="off")
c[1].metric("New capacity needed", f"{new_total/1000:.1f} GW")
c[2].metric("Annual system cost", f"${sol['annual_cost_musd']/1000:.2f}B",
            f"{saving:.1f}% vs gas-only", delta_color="normal" if saving >= 0 else "inverse")
c[3].metric("Loss-of-load expectation", f"{rel['lole_days_per_yr']:.3f} d/yr",
            "meets 0.1 standard" if rel["meets_target"] else "misses 0.1 standard",
            delta_color="normal" if rel["meets_target"] else "inverse")
c[4].metric("CO₂", f"{sol['co2_mt']:.1f} Mt/yr", f"{sol['co2_t_per_mwh']:.2f} t/MWh", delta_color="off")

left, right = st.columns(2)
with left:
    st.subheader("Least-cost new build vs. gas-only")
    fig = go.Figure()
    for k in optimize.NEW_TECHS:
        fig.add_bar(name=TECH_LABELS[k], y=["Optimized", "Gas-only"],
                    x=[sol["new_capacity_mw"][k] / 1000, gas_only["new_capacity_mw"][k] / 1000],
                    orientation="h", marker_color=TECH_COLORS[k],
                    marker_line=dict(color="white", width=2),
                    hovertemplate="%{y}: %{x:.2f} GW<extra>" + TECH_LABELS[k] + "</extra>")
    fig.update_layout(barmode="stack", height=300, xaxis_title="GW", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Where the energy comes from")
    e = sol["energy_twh"]
    keys = [k for k in e if e[k] > 0.001]
    fig2 = go.Figure(go.Bar(x=[e[k] for k in keys], y=[k.replace("_", " ").title() for k in keys],
                            orientation="h", marker_color="#2a78d6",
                            hovertemplate="%{y}: %{x:.1f} TWh<extra></extra>"))
    fig2.update_layout(height=300, xaxis_title="TWh / yr", margin=dict(l=0, r=0, t=10, b=0),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, width="stretch")

st.subheader("Plan details")
st.dataframe(pd.DataFrame({
    "Optimized": [sol["annual_cost_musd"], sol["avg_cost_usd_per_mwh"], sol["reserve_shadow_price_usd_per_kw_yr"],
                  sol["co2_mt"], round(new_total)],
    "Gas-only": [gas_only["annual_cost_musd"], gas_only["avg_cost_usd_per_mwh"],
                 gas_only["reserve_shadow_price_usd_per_kw_yr"], gas_only["co2_mt"],
                 round(sum(gas_only["new_capacity_mw"].values()))],
}, index=["Annual cost ($M)", "Average cost ($/MWh)", "Capacity value ($/kW-yr, LP dual)",
          "CO₂ (Mt/yr)", "New capacity (MW)"]), width="stretch")
st.caption(f"Reliability: {rel['draws']} simulated years · LOLH {rel['lolh_hours_per_yr']} h/yr · "
           f"EUE {rel['eue_mwh_per_yr']:,.0f} MWh/yr. Weather years: {res['ensemble']['weather_years']}.")

with st.expander("Model backtest and methodology"):
    st.write("Load-shape model (gradient boosting) vs. linear HDD/CDD benchmark, out-of-sample:")
    st.dataframe(pd.DataFrame([{"Test year": b["test_year"], "GBM MAPE %": b["gbm"]["mape_pct"],
                                "Linear MAPE %": b["linear_benchmark"]["mape_pct"],
                                "GBM daily-peak MAPE %": b["gbm"]["daily_peak_mape_pct"]}
                               for b in res["backtest"]]), hide_index=True)
    for f in ("peak_scenarios.png", "reserve_tradeoff.png"):
        if (FIGURES / f).exists():
            st.image(str(FIGURES / f))
    st.markdown("Full method, assumptions and limitations: see the project README.")
