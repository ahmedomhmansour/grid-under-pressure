"""Weather-driven load-shape model.

Idea: separate *how much* load the zone has (a slow-moving level driven by
data-center build-out) from *how load moves with weather and the calendar*
(the shape). We model

    load_t = level_t x shape(weather_t, calendar_t)

where level_t is the trailing 365-day mean load and `shape` is learned by a
gradient-boosted tree model. Trees cannot extrapolate a trend, so keeping
the trend out of the learning problem is what lets us push the model to
future years: future load = (scenario level) x (learned shape).

The fitted shape model is then replayed over every historical weather year
to give a *weather-year ensemble* of base-year load — the raw material for
both the scenarios and the Monte Carlo reliability analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

HOURS_PER_YEAR = 8760


def add_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["level"] = out["load_mw"].rolling(HOURS_PER_YEAR, min_periods=int(0.9 * HOURS_PER_YEAR)).mean()
    out["shape"] = out["load_mw"] / out["level"]
    return out


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    local = pd.DatetimeIndex(df["time_local"])
    years = range(local.year.min(), local.year.max() + 1)
    hol = holidays.US(years=years, subdiv="VA")
    temp = df["temp_f"]
    X = pd.DataFrame(index=df.index)
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["month"] = local.month
    X["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    X["holiday"] = np.array([d in hol for d in local.date], dtype=int)
    X["temp_f"] = temp.to_numpy()
    X["cdd"] = np.clip(temp - 65, 0, None).to_numpy()
    X["hdd"] = np.clip(65 - temp, 0, None).to_numpy()
    # buildings respond to accumulated heat/cold, not just the current hour
    X["temp_24h"] = temp.rolling(24, min_periods=1).mean().to_numpy()
    X["temp_72h"] = temp.rolling(72, min_periods=1).mean().to_numpy()
    X["ghi_wm2"] = df["ghi_wm2"].to_numpy()
    return X


@dataclass
class ShapeModel:
    model: HistGradientBoostingRegressor

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, seed: int = 7) -> "ShapeModel":
        m = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
            min_samples_leaf=40, l2_regularization=1.0,
            categorical_features=["hour", "dow", "month"], random_state=seed,
        )
        m.fit(X, y)
        return cls(m)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def _metrics(actual: np.ndarray, pred: np.ndarray, local: pd.DatetimeIndex) -> dict:
    mape = float(np.mean(np.abs(pred - actual) / actual) * 100)
    a = pd.Series(actual, index=local).resample("D").max()
    p = pd.Series(pred, index=local).resample("D").max()
    return {
        "mape_pct": round(mape, 2),
        "daily_peak_mape_pct": round(float((np.abs(p - a) / a).mean() * 100), 2),
        "annual_peak_error_pct": round(float((p.max() - a.max()) / a.max() * 100), 2),
    }


def backtest(df: pd.DataFrame, test_year: int, seed: int = 7) -> dict:
    """Train on years < test_year, score on test_year.

    The annual level is taken as known (it is a scenario input, not a
    forecast target), so this isolates how well the model captures
    weather- and calendar-driven variation. A linear HDD/CDD regression is
    reported alongside as a benchmark.
    """
    d = add_level(df).dropna(subset=["level", "shape", "temp_f"])
    local = pd.DatetimeIndex(d["time_local"])
    train = local.year < test_year
    test = local.year == test_year
    if test.sum() < 24 * 300:
        raise ValueError(f"Not enough data in {test_year} to backtest.")
    X = make_features(d)
    gbm = ShapeModel.fit(X[train], d.loc[train, "shape"], seed)
    lin_cols = ["cdd", "hdd", "temp_24h"]
    lin_X = pd.get_dummies(X[lin_cols + ["hour", "dow"]], columns=["hour", "dow"], dtype=float)
    lin = LinearRegression().fit(lin_X[train], d.loc[train, "shape"])
    actual = d.loc[test, "load_mw"].to_numpy()
    lvl = d.loc[test, "level"].to_numpy()
    lt = local[test]
    return {
        "test_year": test_year,
        "gbm": _metrics(actual, gbm.predict(X[test]) * lvl, lt),
        "linear_benchmark": _metrics(actual, lin.predict(lin_X[test]) * lvl, lt),
    }


def solar_cf_from_ghi(ghi: np.ndarray, gain: float = 1.15, cap: float = 0.90) -> np.ndarray:
    """Crude AC capacity-factor proxy for single-axis-tracking solar from horizontal irradiance.

    `gain` lumps together tracker plane-of-array gain and inverter loading
    (DC/AC ratio) net of losses; `cap` is inverter clipping. Gives ~20-23% annual
    CF in Virginia. A better version would run pvlib on hourly weather.
    """
    return np.clip(ghi / 1000.0 * gain, 0, cap)


def weather_year_ensemble(df: pd.DataFrame, base_year: int, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Fit on all data, then replay each complete weather year at the base-year level.

    Returns a long table: weather_year, hour_of_year (0..8759), load_mw
    (base-year level), solar_cf, temp_f, plus metadata.
    """
    d = add_level(df).dropna(subset=["level", "shape", "temp_f"])
    X = make_features(d)
    model = ShapeModel.fit(X, d["shape"], seed)

    local = pd.DatetimeIndex(d["time_local"])
    base_mask = local.year == base_year
    if base_mask.sum() < 24 * 300:
        raise ValueError(f"base_year {base_year} is not (nearly) complete in the data.")
    base_level = float(d.loc[base_mask, "level"].iloc[-1])

    # every complete local-calendar weather year (needs no load, only weather)
    full = df.dropna(subset=["temp_f", "ghi_wm2"])
    fl = pd.DatetimeIndex(full["time_local"])
    Xf = make_features(full)
    shape_all = model.predict(Xf)
    rows = []
    for yr in sorted(set(fl.year)):
        m = (fl.year == yr) & ~((fl.month == 2) & (fl.day == 29))
        if m.sum() < HOURS_PER_YEAR - 24:
            continue
        idx = np.flatnonzero(m)[:HOURS_PER_YEAR]
        if len(idx) < HOURS_PER_YEAR:
            continue
        ghi = full["ghi_wm2"].to_numpy()[idx]
        rows.append(pd.DataFrame({
            "weather_year": yr,
            "hour_of_year": np.arange(HOURS_PER_YEAR),
            "load_mw": shape_all[idx] * base_level,
            "solar_cf": solar_cf_from_ghi(ghi),
            "temp_f": full["temp_f"].to_numpy()[idx],
        }))
    ens = pd.concat(rows, ignore_index=True)
    meta = {"base_level_mw": round(base_level, 1), "weather_years": sorted(ens.weather_year.unique().tolist())}
    return ens, meta
