"""Derived macro signal numbers — pure math, no LLM, no network, no I/O.

Callers (data/snapshots.py) load raw series from the store and pass DataFrames
in; this module only computes derived numbers from already-loaded data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

UTC = timezone.utc


@dataclass(frozen=True)
class MacroSnapshot:
    """Derived macro signals as of a point in time — the LLM's raw material."""
    as_of: datetime
    fed_funds: float
    dgs10: float
    dgs2: float
    yield_curve_spread: float      # dgs10 - dgs2, percentage points
    curve_inverted: bool
    cpi_yoy: float                 # trailing 12m CPI % change (inflation rate)
    real_fed_funds: float          # fed_funds - cpi_yoy
    unemployment: float
    unemployment_change_1y: float  # percentage-point change vs ~1 year ago


def latest_value(series: pd.DataFrame) -> float:
    """Return the value at the most recent date in a (date, value) series.

    Raises ValueError if the series is empty — callers must supply data as
    of a date where the series has observations.
    """
    if series.empty:
        raise ValueError("Cannot compute latest_value on an empty series")
    row = series.sort_values("date").iloc[-1]
    return float(row["value"])


def _value_near(series: pd.DataFrame, target_date: pd.Timestamp, tolerance_days: int = 45) -> float | None:
    """Return the value observed closest to target_date, within tolerance_days."""
    if series.empty:
        return None
    diffs = (series["date"] - target_date).abs()
    idx = diffs.idxmin()
    if diffs.loc[idx] > pd.Timedelta(days=tolerance_days):
        return None
    return float(series.loc[idx, "value"])


def yoy_change(series: pd.DataFrame, *, mode: str = "pct") -> float:
    """Trailing 12-month change in a series, evaluated at its latest date.

    mode="pct": percentage change (e.g. CPI inflation rate).
    mode="diff": raw difference (e.g. unemployment percentage-point change).
    Raises ValueError if there's no observation ~1 year before the latest date.
    """
    if series.empty:
        raise ValueError("Cannot compute yoy_change on an empty series")
    ordered = series.sort_values("date")
    latest_date = ordered["date"].iloc[-1]
    latest = float(ordered["value"].iloc[-1])
    year_ago_target = latest_date - pd.Timedelta(days=365)
    year_ago = _value_near(ordered, year_ago_target)
    if year_ago is None:
        raise ValueError("No observation ~1 year before the latest date")

    if mode == "pct":
        if year_ago == 0:
            raise ValueError("Cannot compute percent change from a zero base")
        return (latest - year_ago) / year_ago
    if mode == "diff":
        return latest - year_ago
    raise ValueError(f"Unknown mode: {mode!r}")


def build_macro_snapshot(series: dict[str, pd.DataFrame], as_of: datetime) -> MacroSnapshot:
    """Assemble a MacroSnapshot from raw FRED series DataFrames.

    `series` keys are FRED series IDs (FEDFUNDS, DGS10, DGS2, CPIAUCSL, UNRATE);
    each DataFrame must already be sliced to observations <= as_of by the caller.
    """
    fed_funds = latest_value(series["FEDFUNDS"])
    dgs10 = latest_value(series["DGS10"])
    dgs2 = latest_value(series["DGS2"])
    spread = dgs10 - dgs2
    cpi_yoy = yoy_change(series["CPIAUCSL"], mode="pct")
    unemployment = latest_value(series["UNRATE"])
    unemployment_change_1y = yoy_change(series["UNRATE"], mode="diff")

    return MacroSnapshot(
        as_of=as_of,
        fed_funds=fed_funds,
        dgs10=dgs10,
        dgs2=dgs2,
        yield_curve_spread=spread,
        curve_inverted=spread < 0,
        cpi_yoy=cpi_yoy,
        real_fed_funds=fed_funds - cpi_yoy * 100,  # cpi_yoy is a fraction; fed_funds is in percentage points
        unemployment=unemployment,
        unemployment_change_1y=unemployment_change_1y,
    )
