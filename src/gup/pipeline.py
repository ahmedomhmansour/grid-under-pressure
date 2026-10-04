"""End-to-end run: backtest -> weather-year ensemble -> scenarios -> LP -> Monte Carlo.

    python -m gup.pipeline            # real data in data/processed/hourly.parquet
    python -m gup.pipeline --demo     # synthetic data, outputs stamped DEMO DATA
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import forecast, optimize, reliability, repdays, scenarios
from .config import OUTPUTS, PROCESSED, ensure_dirs, load_config


def historical_peaks(df: pd.DataFrame) -> pd.DataFrame:
    local = pd.DatetimeIndex(df["time_local"])
    s = pd.Series(df["load_mw"].to_numpy(), index=local)
    season = pd.Series(["summer" if m in (6, 7, 8, 9) else "winter" if m in (12, 1, 2) else "shoulder"
                        for m in local.month], index=local)
    # winter season labeled by the year it starts (Dec 2025 + Jan-Feb 2026 = winter 2025-26)
    wyear = pd.Series(local.year - (local.month <= 2).astype(int), index=local)
    out = []
    counts = pd.Series(local.year).value_counts()
    for yr in sorted(y for y in counts.index if counts[y] >= 24 * 200):  # skip stub years
        sm = s[(season == "summer") & (local.year == yr)]
        wm = s[(season == "winter") & (wyear == yr)]
        out.append({"year": yr,
                    "summer_peak_mw": round(float(sm.max()), 0) if sm.notna().sum() > 24 * 60 else None,
                    "winter_peak_mw": round(float(wm.max()), 0) if wm.notna().sum() > 24 * 60 else None,
                    "annual_peak_mw": round(float(s[local.year == yr].max()), 0)})
    out_df = pd.DataFrame(out)
    return out_df.astype(object).where(out_df.notna(), None)


def reliability_sensitivity(cfg: dict, ens: pd.DataFrame, years: int, draws: int | None,
                            seed: int) -> list[dict]:
    """Re-test the base-case plan while changing one judgment assumption at a time.
    Shows which assumptions the reliability conclusion actually depends on."""
    import copy
    g = cfg["scenarios"]["base"]["peak_growth"]
    tgt = scenarios.target_load(ens, g, years)
    peak = scenarios.summary(ens, g, years)["peak_median_mw"]
    L, S, W = repdays.select(tgt, cfg["optimization"]["rep_days"], seed)
    plan = optimize.solve(cfg, L, S, W, peak)["new_capacity_mw"]
    cases = [("Baseline assumptions", lambda c: None),
             ("No cold-weather outages", lambda c: c["reliability"].update(cold_weather=None)),
             ("Cold-weather effect halved", lambda c: c["reliability"]["cold_weather"].update(
                 extra_outage_per_degf=c["reliability"]["cold_weather"]["extra_outage_per_degf"] / 2)),
             ("Imports 100% deliverable", lambda c: c["reliability"].update(import_availability=1.0)),
             ("Existing outage rate 4% (vs 6%)", lambda c: c["reliability"].update(existing_forced_outage_rate=0.04))]
    rows = []
    for label, change in cases:
        c = copy.deepcopy(cfg)
        change(c)
        rel = reliability.simulate(c, tgt, plan, draws)
        rows.append({"case": label, "lole_days_per_yr": rel["lole_days_per_yr"],
                     "eue_mwh_per_yr": rel["eue_mwh_per_yr"], "meets_target": rel["meets_target"]})
    return rows


def reserve_sweep(cfg: dict, ens: pd.DataFrame, years: int, draws: int | None, seed: int,
                  factors=(0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00)) -> list[dict]:
    """Cost vs. reliability: re-plan the base scenario at several reliability
    requirements (accredited MW as a multiple of peak) and stress-test each plan
    with Monte Carlo. Shows what each increment of required capacity buys."""
    g = cfg["scenarios"]["base"]["peak_growth"]
    tgt = scenarios.target_load(ens, g, years)
    peak = scenarios.summary(ens, g, years)["peak_median_mw"]
    L, S, W = repdays.select(tgt, cfg["optimization"]["rep_days"], seed)
    rows = []
    for f in factors:
        sol = optimize.solve(cfg, L, S, W, peak, fpr=f)
        rel = reliability.simulate(cfg, tgt, sol["new_capacity_mw"], draws)
        rows.append({"fpr": f, "annual_cost_musd": sol["annual_cost_musd"],
                     "new_mw_total": round(sum(sol["new_capacity_mw"].values()), 0),
                     "lole_days_per_yr": rel["lole_days_per_yr"], "eue_mwh_per_yr": rel["eue_mwh_per_yr"]})
    return rows


def run(df: pd.DataFrame, cfg: dict, demo: bool = False, draws: int | None = None) -> dict:
    t0 = time.time()
    st = cfg["study"]
    base, target = st["base_year"], st["target_year"]
    years = target - base
    seed = cfg["optimization"]["seed"]

    print("1/5 backtesting load model…")
    bt = [forecast.backtest(df, y, seed) for y in (base - 1, base)]

    print("2/5 building weather-year ensemble…")
    ens, meta = forecast.weather_year_ensemble(df, base, seed)
    ens.to_parquet(OUTPUTS / "ensemble_base_year.parquet", index=False)

    results = {"demo_data": demo, "study": st, "ensemble": meta, "backtest": bt,
               "history": historical_peaks(df).to_dict(orient="records"), "scenarios": {}}
    for i, (name, sc) in enumerate(cfg["scenarios"].items(), start=1):
        print(f"3/5 scenario {name} ({i}/{len(cfg['scenarios'])}): LP + Monte Carlo…")
        g = sc["peak_growth"]
        tgt = scenarios.target_load(ens, g, years)
        summ = scenarios.summary(ens, g, years)
        L, S, W = repdays.select(tgt, cfg["optimization"]["rep_days"], seed)
        peak = summ["peak_median_mw"]
        opt = optimize.solve(cfg, L, S, W, peak)
        gas = optimize.solve(cfg, L, S, W, peak, allowed=["gas_cc", "gas_ct"])
        rel_opt = reliability.simulate(cfg, tgt, opt["new_capacity_mw"], draws)
        rel_gas = reliability.simulate(cfg, tgt, gas["new_capacity_mw"], draws)
        results["scenarios"][name] = {
            "label": sc["label"], "load": summ,
            "optimized": {**opt, "reliability": rel_opt},
            "gas_only": {**gas, "reliability": rel_gas},
            "savings_vs_gas_only_pct": round(
                (gas["annual_cost_musd"] - opt["annual_cost_musd"]) / gas["annual_cost_musd"] * 100, 1),
        }

    print("4/5 reserve-margin sweep (base scenario)…")
    results["reserve_sweep"] = reserve_sweep(cfg, ens, years, draws, seed)
    results["reliability_sensitivity"] = reliability_sensitivity(cfg, ens, years, draws, seed)

    print("    writing results…")
    results["runtime_s"] = round(time.time() - t0, 1)
    with open(OUTPUTS / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("5/5 done in", results["runtime_s"], "s →", OUTPUTS / "results.json")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no API key needed)")
    ap.add_argument("--draws", type=int, default=None, help="Monte Carlo draws override")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    if args.demo:
        from .synthetic import make_hourly_table
        df = make_hourly_table("2019-01-01", f"{cfg['study']['base_year'] + 1}-01-01")
    else:
        path = PROCESSED / "hourly.parquet"
        if not path.exists():
            raise SystemExit("No processed data. Run fetch_eia, fetch_weather and clean first "
                             "(or `python -m gup.pipeline --demo`).")
        df = pd.read_parquet(path)
    run(df, cfg, demo=args.demo, draws=args.draws)


if __name__ == "__main__":
    main()
