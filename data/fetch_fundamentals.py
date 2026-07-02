from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, OSError)),
    reraise=True,
)
def _fetch_one(ticker: str) -> dict:
    """Fetch current P/E and dividend yield for one ticker via yfinance.

    yfinance frequently returns None for ETF P/E and yield fields — pass
    those through as None rather than guessing; callers must tolerate it.
    """
    info = yf.Ticker(ticker).info
    now = datetime.now(UTC)
    return {
        "ticker": ticker,
        "as_of": now,
        "pe": info.get("trailingPE"),
        "dividend_yield": info.get("yield") or info.get("dividendYield"),
        "fetched_at": now,
    }


def fetch_fundamentals(tickers: list[str] | None = None) -> dict[str, int]:
    """Fetch current fundamentals for each ticker, save via store, return {ticker: 1 or 0}."""
    settings = get_settings()
    tickers = tickers or settings.tickers
    results: dict[str, int] = {}

    for ticker in tickers:
        try:
            record = _fetch_one(ticker)
            store.save_fundamentals([record])
            results[ticker] = 1
            logger.info(
                "Fetched fundamentals for %s: PE=%s yield=%s",
                ticker, record["pe"], record["dividend_yield"],
            )
        except Exception:
            logger.exception("Failed to fetch fundamentals for %s", ticker)
            results[ticker] = 0

    return results


if __name__ == "__main__":
    fetch_fundamentals()
