from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

_PRICE_COLS = ["date", "open", "high", "low", "close", "volume", "ticker", "fetched_at"]
_NEWS_COLS = ["published_at", "fetched_at", "ticker", "headline", "summary", "source", "url"]
_MACRO_COLS = ["series_id", "date", "value", "fetched_at"]
_FILING_COLS = ["ticker", "form", "accession_no", "filed_at", "fetched_at", "path"]


def _data_dir() -> Path:
    """Return the configured data/raw directory, creating it if absent."""
    return get_settings().data_dir


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_parquet(path: Path, empty_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=empty_cols)
    return pd.read_parquet(path, engine="pyarrow")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    df.to_parquet(path, engine="pyarrow", index=False)


def _to_utc(series: pd.Series) -> pd.Series:
    """Normalize a datetime series to UTC timezone-aware."""
    if series.empty:
        return series
    if series.dtype == "object":
        series = pd.to_datetime(series, utc=True)
    elif hasattr(series.dtype, "tz") and series.dtype.tz is None:
        series = series.dt.tz_localize("UTC")
    elif hasattr(series.dtype, "tz") and series.dtype.tz is not None:
        series = series.dt.tz_convert("UTC")
    return series


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def save_prices(df: pd.DataFrame, ticker: str) -> Path:
    """Upsert OHLCV rows for one ticker to data/raw/prices/<ticker>.parquet."""
    path = _data_dir() / "prices" / f"{ticker}.parquet"
    df = df.copy()
    df["date"] = _to_utc(df["date"])
    df["fetched_at"] = _to_utc(df["fetched_at"])

    existing = _read_parquet(path, _PRICE_COLS)
    if not existing.empty:
        existing["date"] = _to_utc(existing["date"])
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = (
        combined
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    _write_parquet(combined, path)
    logger.debug("Saved %d price rows for %s to %s", len(combined), ticker, path)
    return path


def load_prices(ticker: str, *, until: datetime | None = None) -> pd.DataFrame:
    """Load OHLCV for a ticker; if `until` set, return only rows with date <= until."""
    path = _data_dir() / "prices" / f"{ticker}.parquet"
    df = _read_parquet(path, _PRICE_COLS)
    if df.empty:
        return df
    df["date"] = _to_utc(df["date"])
    if until is not None:
        ts = pd.Timestamp(until)
        cutoff = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        df = df[df["date"] <= cutoff].copy()
        # Guard: assert no lookahead leak
        assert df["date"].max() <= cutoff if not df.empty else True, "Lookahead detected in load_prices"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def save_news(records: list[dict], ticker: str) -> Path:
    """Append+dedup news records to data/raw/news/<ticker>.parquet (dedup key = url)."""
    path = _data_dir() / "news" / f"{ticker}.parquet"
    incoming = pd.DataFrame(records, columns=_NEWS_COLS) if records else pd.DataFrame(columns=_NEWS_COLS)
    if not incoming.empty:
        incoming["published_at"] = _to_utc(incoming["published_at"])
        incoming["fetched_at"] = _to_utc(incoming["fetched_at"])

    existing = _read_parquet(path, _NEWS_COLS)
    if not existing.empty:
        existing["published_at"] = _to_utc(existing["published_at"])
        existing["fetched_at"] = _to_utc(existing["fetched_at"])
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = (
        combined
        .drop_duplicates(subset=["url"], keep="last")
        .sort_values("published_at")
        .reset_index(drop=True)
    )
    _write_parquet(combined, path)
    logger.debug("Saved %d news rows for %s to %s", len(combined), ticker, path)
    return path


def load_news(ticker: str, *, until: datetime | None = None) -> pd.DataFrame:
    """Load news for a ticker; if `until` set, return only rows with published_at <= until."""
    path = _data_dir() / "news" / f"{ticker}.parquet"
    df = _read_parquet(path, _NEWS_COLS)
    if df.empty:
        return df
    df["published_at"] = _to_utc(df["published_at"])
    if until is not None:
        ts = pd.Timestamp(until)
        cutoff = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        # Filter on published_at, NEVER fetched_at — published_at is the true event time
        df = df[df["published_at"] <= cutoff].copy()
        assert df["published_at"].max() <= cutoff if not df.empty else True, "Lookahead detected in load_news"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------

def save_macro(df: pd.DataFrame, series_id: str) -> Path:
    """Upsert a FRED series to data/raw/macro/<series_id>.parquet."""
    path = _data_dir() / "macro" / f"{series_id}.parquet"
    df = df.copy()
    df["date"] = _to_utc(df["date"])
    df["fetched_at"] = _to_utc(df["fetched_at"])

    existing = _read_parquet(path, _MACRO_COLS)
    if not existing.empty:
        existing["date"] = _to_utc(existing["date"])
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = (
        combined
        .drop_duplicates(subset=["series_id", "date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    _write_parquet(combined, path)
    logger.debug("Saved %d macro rows for %s to %s", len(combined), series_id, path)
    return path


def load_macro(series_id: str, *, until: datetime | None = None) -> pd.DataFrame:
    """Load a macro series; if `until` set, return only observations dated <= until."""
    path = _data_dir() / "macro" / f"{series_id}.parquet"
    df = _read_parquet(path, _MACRO_COLS)
    if df.empty:
        return df
    df["date"] = _to_utc(df["date"])
    if until is not None:
        ts = pd.Timestamp(until)
        cutoff = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        # NOTE: FRED observation date ≠ release date; CPI for month M publishes mid-M+1.
        # Phase 3 backtester should account for release lag via ALFRED vintage data.
        # For now we filter on observation date — may introduce slight lookahead for macro.
        df = df[df["date"] <= cutoff].copy()
        assert df["date"].max() <= cutoff if not df.empty else True, "Lookahead detected in load_macro"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------

def save_filing_index(records: list[dict]) -> Path:
    """Append+dedup filing metadata to data/raw/filings_index.parquet (dedup key = accession_no)."""
    path = _data_dir() / "filings_index.parquet"
    incoming = pd.DataFrame(records, columns=_FILING_COLS) if records else pd.DataFrame(columns=_FILING_COLS)
    if not incoming.empty:
        incoming["filed_at"] = _to_utc(incoming["filed_at"])
        incoming["fetched_at"] = _to_utc(incoming["fetched_at"])

    existing = _read_parquet(path, _FILING_COLS)
    if not existing.empty:
        existing["filed_at"] = _to_utc(existing["filed_at"])
        existing["fetched_at"] = _to_utc(existing["fetched_at"])
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = (
        combined
        .drop_duplicates(subset=["accession_no"], keep="last")
        .sort_values("filed_at")
        .reset_index(drop=True)
    )
    _write_parquet(combined, path)
    logger.debug("Saved %d filing index rows to %s", len(combined), path)
    return path


def load_filings(ticker: str, *, until: datetime | None = None) -> pd.DataFrame:
    """Load filing metadata for a ticker; if `until` set, only filings with filed_at <= until."""
    path = _data_dir() / "filings_index.parquet"
    df = _read_parquet(path, _FILING_COLS)
    if df.empty:
        return df
    df["filed_at"] = _to_utc(df["filed_at"])
    df = df[df["ticker"] == ticker]
    if until is not None:
        ts = pd.Timestamp(until)
        cutoff = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        df = df[df["filed_at"] <= cutoff].copy()
        assert df["filed_at"].max() <= cutoff if not df.empty else True, "Lookahead detected in load_filings"
    return df.reset_index(drop=True)
