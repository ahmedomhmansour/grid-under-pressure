"""Paths and assumption loading."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, OUTPUTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)


def crf(rate: float, years: int) -> float:
    """Capital recovery factor: turns an up-front cost into an equal annual payment."""
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def annualized_capex(cfg: dict, tech: str) -> float:
    """Annualized fixed cost of new capacity, $/kW-yr (overnight x CRF + fixed O&M)."""
    t = cfg["technologies"][tech]
    if "annualized_fixed" in t:
        return float(t["annualized_fixed"])
    return t["overnight_cost"] * crf(cfg["finance"]["wacc"], t["life_years"]) + t.get("fixed_om", 0.0)
