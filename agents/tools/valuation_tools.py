"""Tool functions the valuation agent can call on demand (Phase 6).

Thin adapters only: I/O goes through data/; no math beyond simple aggregation.
"""
from __future__ import annotations

import pandas as pd

from utils.llm import ToolSpec


def fetch_etf_fundamentals(ticker: str) -> dict:
    """Current P/E, dividend yield, and expense ratio for one ticker in the allowed universe."""
    from data import store
    from data.fetch_fundamentals import fetch_fundamentals
    from utils.config import BENCHMARKS, EXPENSE_RATIOS, FULL_UNIVERSE

    allowed = set(FULL_UNIVERSE) | set(BENCHMARKS)
    if ticker not in allowed:
        return {"error": f"{ticker!r} is not in the allowed universe"}

    df = store.load_fundamentals(ticker)
    if df.empty:
        fetch_fundamentals([ticker])
        df = store.load_fundamentals(ticker)
    if df.empty:
        return {"error": f"No fundamentals available for {ticker} even after fetching"}

    latest = df.sort_values("as_of").iloc[-1]
    return {
        "ticker": ticker,
        "pe": float(latest["pe"]) if pd.notna(latest["pe"]) else None,
        "dividend_yield": float(latest["dividend_yield"]) if pd.notna(latest["dividend_yield"]) else None,
        "expense_ratio": EXPENSE_RATIOS.get(ticker),
    }


def compare_historical_pe(ticker: str) -> dict:
    """Compare current P/E against locally accumulated snapshots — honest about thin history."""
    from data import store

    df = store.load_fundamentals(ticker)
    if df.empty:
        return {"error": f"No fundamentals history for {ticker}"}

    pe_series = df.dropna(subset=["pe"])
    if pe_series.empty:
        return {"ticker": ticker, "note": "No P/E history available (common for bond ETFs)"}

    result = {
        "ticker": ticker,
        "current_pe": float(pe_series["pe"].iloc[-1]),
        "historical_min": float(pe_series["pe"].min()),
        "historical_max": float(pe_series["pe"].max()),
        "n_snapshots": int(len(pe_series)),
    }
    if len(pe_series) < 30:
        result["note"] = "Thin history — these are local daily snapshots, not a long-run P/E series"
    return result


VALUATION_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="fetch_etf_fundamentals",
        description="Fetch current P/E, dividend yield, and expense ratio for one ETF ticker.",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "ETF ticker symbol, e.g. VTI"}},
            "required": ["ticker"],
        },
        fn=fetch_etf_fundamentals,
    ),
    ToolSpec(
        name="compare_historical_pe",
        description="Compare a ticker's current P/E against locally accumulated historical snapshots.",
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "ETF ticker symbol, e.g. VTI"}},
            "required": ["ticker"],
        },
        fn=compare_historical_pe,
    ),
]
