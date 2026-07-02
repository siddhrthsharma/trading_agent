"""Tool functions the macro agent can call on demand (Phase 6).

Thin adapters only: math delegates to engine/signals.py, I/O to data/. An
allowlist bounds which FRED series an agent may ever request.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from utils.llm import ToolSpec

UTC = timezone.utc

FRED_ALLOWLIST: set[str] = {
    "FEDFUNDS",   # Federal funds rate
    "CPIAUCSL",   # CPI (inflation)
    "UNRATE",     # Unemployment rate
    "DGS10",      # 10-Year Treasury yield
    "DGS2",       # 2-Year Treasury yield
    "T10Y2Y",     # 10Y-2Y spread, precomputed by FRED directly
    "T10YIE",     # 10-Year breakeven inflation rate
}


def fetch_fred_series(series_id: str, lookback_years: int = 5) -> dict:
    """Fetch a FRED series (local store first, network fallback) and summarize recent history."""
    if series_id not in FRED_ALLOWLIST:
        return {"error": f"{series_id!r} is not in the allowed FRED series list: {sorted(FRED_ALLOWLIST)}"}

    from data import store
    from data.fetch_macro import fetch_macro

    df = store.load_macro(series_id)
    if df.empty:
        fetch_macro({series_id: series_id})
        df = store.load_macro(series_id)
    if df.empty:
        return {"error": f"No data available for {series_id} even after fetching"}

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * lookback_years)
    recent = df[df["date"] >= cutoff]
    if recent.empty:
        recent = df  # fall back to whatever history exists

    return {
        "series_id": series_id,
        "latest_value": float(recent["value"].iloc[-1]),
        "latest_date": str(recent["date"].iloc[-1].date()),
        "observations_in_window": len(recent),
    }


def compute_yield_curve() -> dict:
    """10Y-2Y Treasury spread from the latest locally stored data."""
    from data import store
    from engine.signals import latest_value

    dgs10 = store.load_macro("DGS10")
    dgs2 = store.load_macro("DGS2")
    if dgs10.empty or dgs2.empty:
        return {"error": "DGS10/DGS2 not available locally — run data/fetch_macro.py first"}

    spread = latest_value(dgs10) - latest_value(dgs2)
    return {"spread": spread, "inverted": spread < 0}


def compute_real_rate() -> dict:
    """Fed funds rate minus trailing CPI inflation, from the latest locally stored data."""
    from data import store
    from engine.signals import latest_value, yoy_change

    fed_funds = store.load_macro("FEDFUNDS")
    cpi = store.load_macro("CPIAUCSL")
    if fed_funds.empty or cpi.empty:
        return {"error": "FEDFUNDS/CPIAUCSL not available locally — run data/fetch_macro.py first"}

    cpi_yoy = yoy_change(cpi, mode="pct")
    real_rate = latest_value(fed_funds) - cpi_yoy * 100
    return {"real_fed_funds": real_rate, "cpi_yoy": cpi_yoy}


def get_macro_snapshot(as_of: str | None = None) -> dict:
    """Full MacroSnapshot (as a dict) as of a given ISO date, or the latest available."""
    from data.snapshots import macro_snapshot

    as_of_dt = datetime.fromisoformat(as_of).replace(tzinfo=UTC) if as_of else None
    snap = macro_snapshot(as_of=as_of_dt)
    data = asdict(snap)
    data["as_of"] = snap.as_of.isoformat()
    return data


MACRO_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="fetch_fred_series",
        description="Fetch a FRED macro series (e.g. FEDFUNDS, DGS10, DGS2, CPIAUCSL, UNRATE, T10Y2Y, T10YIE) and summarize recent history.",
        parameters={
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED series ID"},
                "lookback_years": {"type": "integer", "description": "How many years of history to summarize (default 5)"},
            },
            "required": ["series_id"],
        },
        fn=fetch_fred_series,
    ),
    ToolSpec(
        name="compute_yield_curve",
        description="Compute the current 10Y-2Y Treasury yield curve spread and whether it's inverted.",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=compute_yield_curve,
    ),
    ToolSpec(
        name="compute_real_rate",
        description="Compute the current real (inflation-adjusted) fed funds rate.",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=compute_real_rate,
    ),
    ToolSpec(
        name="get_macro_snapshot",
        description="Get the full macro snapshot (fed funds, yield curve, CPI, unemployment) as of a given ISO date, or latest if omitted.",
        parameters={
            "type": "object",
            "properties": {
                "as_of": {"type": "string", "description": "ISO date, e.g. '2022-09-30'; omit for the latest available"},
            },
            "required": [],
        },
        fn=get_macro_snapshot,
    ),
]
