from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data import store
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, OSError)),
    reraise=True,
)
def _download_one(ticker: str, days: int) -> pd.DataFrame:
    """Download and normalize a single ticker's OHLCV into the canonical schema."""
    period = f"{days}d"
    raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)

    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {ticker}: {missing}")

    df = df[list(_REQUIRED_COLS)].copy()

    # Normalize index → UTC column
    df.index.name = "date"
    df = df.reset_index()
    if df["date"].dtype == "object":
        df["date"] = pd.to_datetime(df["date"], utc=True)
    elif hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    else:
        df["date"] = df["date"].dt.tz_convert("UTC")

    df["ticker"] = ticker
    df["fetched_at"] = pd.Timestamp(datetime.now(UTC))

    return df[["date", "open", "high", "low", "close", "volume", "ticker", "fetched_at"]]


def fetch_prices(tickers: list[str] | None = None) -> dict[str, int]:
    """Fetch OHLCV for each ticker, save via store, return {ticker: rows_written}."""
    settings = get_settings()
    tickers = tickers or settings.tickers
    days = settings.price_history_days
    results: dict[str, int] = {}

    for ticker in tickers:
        try:
            df = _download_one(ticker, days)
            if df.empty:
                logger.warning("No price data returned for %s — skipping", ticker)
                results[ticker] = 0
                continue
            store.save_prices(df, ticker)
            results[ticker] = len(df)
            logger.info("Fetched %d price rows for %s", len(df), ticker)
        except Exception:
            logger.exception("Failed to fetch prices for %s", ticker)
            results[ticker] = 0

        time.sleep(0.5)  # avoid unofficial yfinance rate limiting

    return results


if __name__ == "__main__":
    fetch_prices()
