from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone

import pandas as pd
from fredapi import Fred
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
def _fetch_series(series_id: str) -> pd.DataFrame:
    """Fetch one FRED series into schema (series_id, date, value, fetched_at)."""
    settings = get_settings()
    fred = Fred(api_key=settings.fred_api_key)
    raw = fred.get_series(series_id)  # pd.Series indexed by date

    df = raw.reset_index()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = df["value"].astype(float)
    df["series_id"] = series_id
    df["fetched_at"] = pd.Timestamp(datetime.now(UTC))

    # Drop NaN observations (FRED uses NaN for missing/revised-away data)
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    return df[["series_id", "date", "value", "fetched_at"]]


def fetch_macro(series: dict[str, str] | None = None) -> dict[str, int]:
    """Fetch each configured FRED series, save via store, return {series_id: rows_written}."""
    settings = get_settings()
    series = series or settings.fred_series
    results: dict[str, int] = {}

    for series_id in series:
        try:
            df = _fetch_series(series_id)
            store.save_macro(df, series_id)
            results[series_id] = len(df)
            logger.info("Fetched %d rows for FRED series %s", len(df), series_id)
        except Exception:
            logger.exception("Failed to fetch FRED series %s", series_id)
            results[series_id] = 0

    return results


if __name__ == "__main__":
    fetch_macro()
