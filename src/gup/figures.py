"""Static README figures from outputs/results.json.

    python -m gup.figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIGURES, OUTPUTS, ensure_dirs  # noqa: E402

# Fixed entity -> color mapping (categorical slots in fixed order; never by rank)
TECH_COLORS = {
    "gas_cc": "#2a78d6", "gas_ct": "#eb6834", "solar": "#1baf7a",
    "battery_4h": "#eda100", "demand_response": "#e87ba4",
}
TECH_LABELS = {
    "gas_cc": "Gas CC", "gas_ct": "Gas CT", "solar": "Solar",
    "battery_4h": "Battery 4h", "demand_response": "Demand response",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SCEN_SHADES = {"low": "#a8c8f0", "base": "#2a78d6", "high": "#17407a"}  # sequential, one hue


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def _stamp(fig, demo: bool) -> None:
    if demo:
        fig.text(0.99, 0.01, "DEMO DATA — synthetic, not real results", ha="right",
                 fontsize=9, color="#e34948", fontweight="bold")


def peak_fan(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    hist = [h for h in r["history"] if h["annual_peak_mw"]]
    x = [h["year"] for h in hist]
    y = [h["annual_peak_mw"] / 1000 for h in hist]
    ax.plot(x, y, color=INK, linewidth=2, marker="o", markersize=6, label="Observed annual peak")
    base_year, target_year = r["study"]["base_year"], r["study"]["target_year"]
    for name, sc in r["scenarios"].items():
        bp = sc["load"]["peak_median_mw"] - sc["load"]["increment_mw"]
        g = sc["load"]["growth"]
        yrs = list(range(base_year, target_year + 1))
        vals = [bp * (1 + g) ** (t - base_year) / 1000 for t in yrs]
        ax.plot(yrs, vals, color=SCEN_SHADES[name], linewidth=2, linestyle="--")
        ax.annotate(f"{name} {g:.1%}/yr: {vals[-1]:.1f} GW", (yrs[-1], vals[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axvline(base_year + 0.5, color=GRID, linewidth=1)
    if hist and hist[-1]["year"] > base_year:
        ax.annotate(f"{hist[-1]['year']}: year to date", (x[-1], y[-1]), xytext=(8, -14),
                    textcoords="offset points", ha="left", fontsize=8, color=INK2)
    _style(ax, "Dominion zone peak load: observed and weather-normal scenarios", "Peak load (GW)")
    ax.set_xlim(min(x) - 0.5, target_year + 2.6)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    _stamp(fig, r["demo_data"])
    fig.tight_layout()
    fig.savefig(FIGURES / "peak_scenarios.png", dpi=160)
    plt.close(fig)


def capacity_mix(r: dict) -> None:
    names = list(r["scenarios"])
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    for i, n in enumerate(names):
        left = 0.0
        for k, col in TECH_COLORS.items():
            mw = r["scenarios"][n]["optimized"]["new_capacity_mw"][k] / 1000
            if mw <= 0:
                continue
            ax.barh(i, mw, left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.55,
                    label=TECH_LABELS[k])
            if mw > 1.2:
                ax.text(left + mw / 2, i, f"{mw:.1f}", ha="center", va="center", fontsize=9, color="white")
            left += mw
        ax.text(left + 0.15, i, f"{left:.1f} GW  ·  ${r['scenarios'][n]['optimized']['annual_cost_musd']/1000:.1f}B/yr",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(names)), [r["scenarios"][n]["label"] for n in names])
    _style(ax, f"Least-cost new capacity by {r['study']['target_year']}", "")
    ax.title.set_position((0, 1.02))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("New capacity (GW)", color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="upper right", ncol=len(uniq),
              labelcolor=INK2, fontsize=9)
    ax.set_ylim(len(names) - 0.5, -1.0)   # top-down order, with headroom for the legend
    ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
    _stamp(fig, r["demo_data"])
    fig.tight_layout()
    fig.savefig(FIGURES / "capacity_mix.png", dpi=160)
    plt.close(fig)


def reserve_tradeoff(r: dict) -> None:
    sw = r["reserve_sweep"]
    rm = [s["fpr"] for s in sw]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    a1.plot(rm, [s["annual_cost_musd"] / 1000 for s in sw], color="#2a78d6", linewidth=2, marker="o", markersize=7)
    _style(a1, "Annual system cost", "$ billion / yr")
    a1.set_xlabel("Required accredited capacity (× peak)", color=INK2)
    a2.plot(rm, [s["lole_days_per_yr"] for s in sw], color="#2a78d6", linewidth=2, marker="o", markersize=7)
    a2.axhline(0.1, color="#e34948", linewidth=1.2, linestyle=":")
    a2.annotate("1-day-in-10-years standard", (rm[0], 0.1), xytext=(0, 5), textcoords="offset points",
                ha="left", fontsize=9, color=INK2)
    for ax in (a1, a2):
        ax.axvline(0.9401, color=GRID, linewidth=1.5)
        ax.annotate("PJM FPR 0.94", (0.9401, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=INK2)
    _style(a2, "Simulated loss-of-load expectation", "days / yr")
    a2.set_xlabel("Required accredited capacity (× peak)", color=INK2)
    _stamp(fig, r["demo_data"])
    fig.tight_layout()
    fig.savefig(FIGURES / "reserve_tradeoff.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    with open(OUTPUTS / "results.json", encoding="utf-8") as f:
        r = json.load(f)
    peak_fan(r)
    capacity_mix(r)
    reserve_tradeoff(r)
    print("Figures →", FIGURES)


if __name__ == "__main__":
    main()
