"""I/O glue between the store and engine/ — the only place that reads Parquet
and hands pure numbers to engine/ functions and agents. No LLM calls here."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

from data import store
from engine.signals import MacroSnapshot, build_macro_snapshot
from utils.config import EXPENSE_RATIOS, get_settings

UTC = timezone.utc

_MACRO_SERIES = ["FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "UNRATE"]


@dataclass(frozen=True)
class ValuationSnapshot:
    """Per-fund fundamentals as of a point in time."""
    as_of: datetime
    funds: dict[str, dict[str, float | None]] = field(default_factory=dict)


def macro_snapshot(as_of: datetime | None = None) -> MacroSnapshot:
    """Load FRED series from the store (up to as_of) and build a MacroSnapshot.

    Raises ValueError if any required series has no data as of the given date.
    """
    as_of = as_of or datetime.now(UTC)
    series: dict[str, pd.DataFrame] = {}
    for series_id in _MACRO_SERIES:
        df = store.load_macro(series_id, until=as_of)
        if df.empty:
            raise ValueError(f"No {series_id} data available as of {as_of}")
        series[series_id] = df
    return build_macro_snapshot(series, as_of)


def valuation_snapshot(as_of: datetime | None = None, tickers: list[str] | None = None) -> ValuationSnapshot:
    """Load the latest fundamentals snapshot per ticker (up to as_of) from the store.

    Missing P/E or dividend yield fields are passed through as None — the
    valuation agent must say "data unavailable" rather than guessing.
    """
    as_of = as_of or datetime.now(UTC)
    tickers = tickers or get_settings().tickers

    funds: dict[str, dict[str, float | None]] = {}
    for ticker in tickers:
        df = store.load_fundamentals(ticker, until=as_of)
        expense_ratio = EXPENSE_RATIOS.get(ticker)
        if df.empty:
            funds[ticker] = {"pe": None, "dividend_yield": None, "expense_ratio": expense_ratio}
            continue
        latest = df.sort_values("as_of").iloc[-1]
        pe = latest["pe"]
        div_yield = latest["dividend_yield"]
        funds[ticker] = {
            "pe": float(pe) if pd.notna(pe) else None,
            "dividend_yield": float(div_yield) if pd.notna(div_yield) else None,
            "expense_ratio": expense_ratio,
        }

    return ValuationSnapshot(as_of=as_of, funds=funds)


def load_close_matrix(
    tickers: list[str], *, start: date | None = None, until: datetime | None = None
) -> pd.DataFrame:
    """Wide adjusted-close matrix (date index, ticker columns), inner-joined on common dates.

    Loads each ticker through store.load_prices(until=until), which already
    enforces no-lookahead — this function never sees data after `until`.
    """
    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = store.load_prices(ticker, until=until)
        if df.empty:
            raise ValueError(f"No price data available for {ticker}")
        columns[ticker] = df.set_index("date")["close"]

    close = pd.DataFrame(columns).dropna(how="any")
    if start is not None:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        close = close[close.index >= start_ts]
    return close.sort_index()
