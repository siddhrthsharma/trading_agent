"""Phase 1 data layer tests — all offline, no real API calls."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(ticker: str = "AAPL", days: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * days,
        "high": [110.0] * days,
        "low": [90.0] * days,
        "close": [105.0] * days,
        "volume": [1_000_000] * days,
        "ticker": [ticker] * days,
        "fetched_at": [pd.Timestamp("2026-01-04", tz="UTC")] * days,
    })


def _make_news_records(ticker: str = "AAPL", n: int = 3) -> list[dict]:
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return [
        {
            "published_at": d.to_pydatetime(),
            "fetched_at": datetime(2026, 1, 4, tzinfo=UTC),
            "ticker": ticker,
            "headline": f"Headline {i}",
            "summary": f"Summary {i}",
            "source": "TestSource",
            "url": f"https://example.com/article/{i}",
        }
        for i, d in enumerate(dates)
    ]


def _make_macro_df(series_id: str = "FEDFUNDS", n: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="ME", tz="UTC")
    return pd.DataFrame({
        "series_id": [series_id] * n,
        "date": dates,
        "value": [5.0, 5.25, 5.5],
        "fetched_at": [pd.Timestamp("2026-01-04", tz="UTC")] * n,
    })


def _make_fundamentals_record(ticker: str = "VTI", as_of: str = "2026-01-01", pe: float = 26.0) -> dict:
    return {
        "ticker": ticker,
        "as_of": pd.Timestamp(as_of, tz="UTC"),
        "pe": pe,
        "dividend_yield": 0.01,
        "fetched_at": pd.Timestamp(as_of, tz="UTC"),
    }


# ---------------------------------------------------------------------------
# test_imports
# ---------------------------------------------------------------------------

def test_imports():
    """Every data-layer module must be importable (catches missing deps)."""
    import utils.logger  # noqa: F401
    import utils.config  # noqa: F401
    import data.store  # noqa: F401
    import data.fetch_prices  # noqa: F401
    import data.fetch_macro  # noqa: F401
    import data.fetch_news  # noqa: F401
    import data.fetch_filings  # noqa: F401
    import data.pipeline  # noqa: F401
    import main  # noqa: F401


# ---------------------------------------------------------------------------
# test_logger_singleton
# ---------------------------------------------------------------------------

def test_logger_singleton():
    """Two get_logger() calls must not grow the root handler count."""
    from utils.logger import get_logger
    before = len(logging.getLogger().handlers)
    get_logger("test.a")
    get_logger("test.b")
    after = len(logging.getLogger().handlers)
    assert after == before, f"Handler count grew from {before} to {after}"


# ---------------------------------------------------------------------------
# test_settings_loads
# ---------------------------------------------------------------------------

def test_settings_loads(monkeypatch, tmp_path):
    """Settings must load with required keys present and resolve data_dir."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("NEWS_API_KEY", "test-news")
    monkeypatch.setenv("FRED_API_KEY", "test-fred")

    from utils.config import get_settings
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.groq_api_key == "test-groq"
    assert settings.news_api_key == "test-news"
    assert settings.fred_api_key == "test-fred"
    assert "SPY" in settings.tickers
    assert settings.data_dir.name == "raw"
    assert isinstance(settings.price_history_days, int) and settings.price_history_days > 0

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# test_store_roundtrip
# ---------------------------------------------------------------------------

def test_store_roundtrip(tmp_path, monkeypatch):
    """save_prices → load_prices must return identical data."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    df = _make_price_df()
    store.save_prices(df, "AAPL")
    loaded = store.load_prices("AAPL")

    assert set(loaded.columns) >= {"date", "open", "high", "low", "close", "volume", "ticker"}
    assert len(loaded) == len(df)
    assert (loaded["close"] == df["close"].values).all()


def test_store_roundtrip_news(tmp_path, monkeypatch):
    """save_news → load_news must return all records."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    records = _make_news_records()
    store.save_news(records, "AAPL")
    loaded = store.load_news("AAPL")

    assert len(loaded) == len(records)
    assert set(loaded.columns) >= {"published_at", "headline", "url"}


def test_store_roundtrip_macro(tmp_path, monkeypatch):
    """save_macro → load_macro must return identical data."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    df = _make_macro_df()
    store.save_macro(df, "FEDFUNDS")
    loaded = store.load_macro("FEDFUNDS")

    assert len(loaded) == len(df)
    assert (loaded["value"] == df["value"].values).all()


# ---------------------------------------------------------------------------
# test_store_dedup
# ---------------------------------------------------------------------------

def test_store_dedup(tmp_path, monkeypatch):
    """Double-saving the same price data must not produce duplicate rows."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    df = _make_price_df()
    store.save_prices(df, "AAPL")
    store.save_prices(df, "AAPL")  # second save — same data
    loaded = store.load_prices("AAPL")

    assert len(loaded) == len(df), f"Expected {len(df)} rows but got {len(loaded)}"


def test_store_dedup_news(tmp_path, monkeypatch):
    """Saving the same news articles twice must not produce duplicate rows."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    records = _make_news_records()
    store.save_news(records, "AAPL")
    store.save_news(records, "AAPL")
    loaded = store.load_news("AAPL")

    assert len(loaded) == len(records)


# ---------------------------------------------------------------------------
# test_store_until_no_lookahead  (KEYSTONE TEST)
# ---------------------------------------------------------------------------

def test_store_until_no_lookahead_prices(tmp_path, monkeypatch):
    """load_prices(until=mid) must return ONLY rows on or before mid — no future data."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    df = _make_price_df(days=5)  # 2026-01-01 through 2026-01-05
    store.save_prices(df, "AAPL")

    cutoff = datetime(2026, 1, 3, tzinfo=UTC)
    loaded = store.load_prices("AAPL", until=cutoff)

    assert not loaded.empty, "Should return some rows"
    assert loaded["date"].max() <= pd.Timestamp(cutoff), "Got rows after cutoff (lookahead!)"
    # Rows after cutoff must not appear
    assert len(loaded) == 3, f"Expected 3 rows (Jan 1-3), got {len(loaded)}"


def test_store_until_no_lookahead_news(tmp_path, monkeypatch):
    """load_news(until=mid) must filter on published_at, not fetched_at."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    records = _make_news_records(n=5)  # published 2026-01-01 through 2026-01-05
    # fetched_at is 2026-01-04 for all — filter must NOT use this
    store.save_news(records, "AAPL")

    cutoff = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    loaded = store.load_news("AAPL", until=cutoff)

    assert loaded["published_at"].max() <= pd.Timestamp(cutoff)
    # Articles from Jan 3-5 must not appear even though they were "fetched" on Jan 4
    assert len(loaded) == 2, f"Expected 2 rows (Jan 1-2), got {len(loaded)}"


def test_store_until_no_lookahead_macro(tmp_path, monkeypatch):
    """load_macro(until=mid) must return only observations on or before mid."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    df = _make_macro_df(n=3)  # Jan, Feb, Mar 2026 month-end
    store.save_macro(df, "FEDFUNDS")

    cutoff = datetime(2026, 2, 1, tzinfo=UTC)
    loaded = store.load_macro("FEDFUNDS", until=cutoff)

    assert not loaded.empty
    assert loaded["date"].max() <= pd.Timestamp(cutoff)


# ---------------------------------------------------------------------------
# test_prices_schema
# ---------------------------------------------------------------------------

def test_prices_schema(tmp_path, monkeypatch):
    """fetch_prices must produce a frame with the exact canonical columns and UTC fetched_at."""
    _patch_data_dir(monkeypatch, tmp_path)

    fixture = _make_price_df()
    with patch("data.fetch_prices._download_one", return_value=fixture):
        from data.fetch_prices import fetch_prices
        results = fetch_prices(["AAPL"])

    assert results["AAPL"] == len(fixture)
    from data import store
    loaded = store.load_prices("AAPL")
    expected_cols = {"date", "open", "high", "low", "close", "volume", "ticker", "fetched_at"}
    assert expected_cols <= set(loaded.columns)
    assert loaded["fetched_at"].dt.tz is not None  # UTC-aware


# ---------------------------------------------------------------------------
# test_macro_schema
# ---------------------------------------------------------------------------

def test_macro_schema(tmp_path, monkeypatch):
    """fetch_macro must persist canonical columns with UTC timestamps."""
    _patch_data_dir(monkeypatch, tmp_path)

    fixture = _make_macro_df()
    with patch("data.fetch_macro._fetch_series", return_value=fixture):
        from data.fetch_macro import fetch_macro
        results = fetch_macro({"FEDFUNDS": "Fed funds rate"})

    assert results["FEDFUNDS"] == len(fixture)
    from data import store
    loaded = store.load_macro("FEDFUNDS")
    expected_cols = {"series_id", "date", "value", "fetched_at"}
    assert expected_cols <= set(loaded.columns)
    assert loaded["date"].dt.tz is not None


# ---------------------------------------------------------------------------
# test_news_normalize
# ---------------------------------------------------------------------------

def test_news_normalize():
    """_normalize must map raw fields to canonical schema with UTC published_at."""
    from data.fetch_news import _normalize

    raw = {
        "title": "AAPL hits new high",
        "description": "Apple stock rises.",
        "url": "https://example.com/story1",
        "publishedAt": "2026-01-15T10:30:00Z",
    }
    result = _normalize(raw, "TestSource", "AAPL")

    assert result is not None
    assert result["headline"] == "AAPL hits new high"
    assert result["ticker"] == "AAPL"
    assert result["url"] == "https://example.com/story1"
    assert result["published_at"].tzinfo is not None  # UTC-aware
    assert result["published_at"].year == 2026


# ---------------------------------------------------------------------------
# test_news_drops_undated
# ---------------------------------------------------------------------------

def test_news_drops_undated():
    """_normalize must return None (not now-stamp) when published_at is missing or unparseable."""
    from data.fetch_news import _normalize

    # Missing date
    raw_no_date = {
        "title": "Some article",
        "url": "https://example.com/no-date",
        "publishedAt": None,
    }
    assert _normalize(raw_no_date, "TestSource", "AAPL") is None

    # Unparseable date
    raw_bad_date = {
        "title": "Another article",
        "url": "https://example.com/bad-date",
        "publishedAt": "not-a-date",
    }
    assert _normalize(raw_bad_date, "TestSource", "AAPL") is None


# ---------------------------------------------------------------------------
# test_filings_index_schema
# ---------------------------------------------------------------------------

def test_filings_index_schema(tmp_path):
    """_index_downloaded must produce records with required fields and UTC filed_at."""
    from data.fetch_filings import _index_downloaded

    # Create a minimal directory structure mimicking sec-edgar-downloader output
    ticker_dir = tmp_path / "AAPL" / "10-K"
    accession_dir = ticker_dir / "0000320193-26-000001"
    accession_dir.mkdir(parents=True)
    (accession_dir / "primary-document.htm").write_text("<html>filing</html>")

    records = _index_downloaded("AAPL", "10-K", tmp_path)

    assert len(records) == 1
    r = records[0]
    assert r["ticker"] == "AAPL"
    assert r["form"] == "10-K"
    assert r["accession_no"] == "0000320193-26-000001"
    assert r["filed_at"].tzinfo is not None  # UTC-aware
    assert Path(r["path"]).exists()


# ---------------------------------------------------------------------------
# test_fundamentals_store_roundtrip
# ---------------------------------------------------------------------------

def test_fundamentals_store_roundtrip(tmp_path, monkeypatch):
    """save_fundamentals → load_fundamentals must return identical data."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    record = _make_fundamentals_record()
    store.save_fundamentals([record])
    loaded = store.load_fundamentals("VTI")

    assert len(loaded) == 1
    assert loaded.iloc[0]["pe"] == 26.0


def test_fundamentals_accumulate_across_days(tmp_path, monkeypatch):
    """Fundamentals snapshots from different days must both be kept (history accumulates)."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    store.save_fundamentals([_make_fundamentals_record(as_of="2026-01-01", pe=26.0)])
    store.save_fundamentals([_make_fundamentals_record(as_of="2026-02-01", pe=27.0)])
    loaded = store.load_fundamentals("VTI")

    assert len(loaded) == 2
    assert sorted(loaded["pe"].tolist()) == [26.0, 27.0]


def test_fundamentals_same_day_upserts(tmp_path, monkeypatch):
    """Saving twice for the same (ticker, as_of) must upsert, not duplicate."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    store.save_fundamentals([_make_fundamentals_record(as_of="2026-01-01", pe=26.0)])
    store.save_fundamentals([_make_fundamentals_record(as_of="2026-01-01", pe=99.0)])
    loaded = store.load_fundamentals("VTI")

    assert len(loaded) == 1
    assert loaded.iloc[0]["pe"] == 99.0


def test_fundamentals_no_lookahead(tmp_path, monkeypatch):
    """load_fundamentals(until=mid) must exclude snapshots after the cutoff."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    store.save_fundamentals([_make_fundamentals_record(as_of="2026-01-01", pe=26.0)])
    store.save_fundamentals([_make_fundamentals_record(as_of="2026-06-01", pe=30.0)])

    loaded = store.load_fundamentals("VTI", until=datetime(2026, 3, 1, tzinfo=UTC))
    assert len(loaded) == 1
    assert loaded.iloc[0]["pe"] == 26.0


def test_fetch_fundamentals_tolerates_missing_fields(tmp_path, monkeypatch):
    """fetch_fundamentals must pass through None PE/yield rather than crashing (bond ETFs have no PE)."""
    _patch_data_dir(monkeypatch, tmp_path)

    class FakeTicker:
        def __init__(self, ticker):
            self.info = {"yield": 0.04}  # no trailingPE, no dividendYield — bond-ETF-like

    with patch("data.fetch_fundamentals.yf.Ticker", FakeTicker):
        from data.fetch_fundamentals import fetch_fundamentals
        results = fetch_fundamentals(["BND"])

    assert results["BND"] == 1
    from data import store
    loaded = store.load_fundamentals("BND")
    assert pd.isna(loaded.iloc[0]["pe"])
    assert loaded.iloc[0]["dividend_yield"] == 0.04


# ---------------------------------------------------------------------------
# test_snapshots
# ---------------------------------------------------------------------------

def test_macro_snapshot_builds_from_store(tmp_path, monkeypatch):
    """data.snapshots.macro_snapshot must load each FRED series and build a MacroSnapshot."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    for series_id, values in {
        "FEDFUNDS": [5.0, 5.0],
        "DGS10": [4.0, 4.5],
        "DGS2": [4.0, 4.0],
        "CPIAUCSL": [300.0, 309.0],
        "UNRATE": [4.0, 4.2],
    }.items():
        dates = pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True)
        df = pd.DataFrame({
            "series_id": [series_id] * 2,
            "date": dates,
            "value": values,
            "fetched_at": [pd.Timestamp("2026-01-01", tz="UTC")] * 2,
        })
        store.save_macro(df, series_id)

    from data.snapshots import macro_snapshot
    snap = macro_snapshot(as_of=datetime(2026, 1, 1, tzinfo=UTC))

    assert snap.fed_funds == 5.0
    assert snap.yield_curve_spread == pytest.approx(0.5)


def test_macro_snapshot_missing_series_raises(tmp_path, monkeypatch):
    """macro_snapshot must raise clearly if a required FRED series has no data."""
    _patch_data_dir(monkeypatch, tmp_path)

    from data.snapshots import macro_snapshot
    with pytest.raises(ValueError):
        macro_snapshot(as_of=datetime(2026, 1, 1, tzinfo=UTC))


def test_valuation_snapshot_handles_missing_and_present_data(tmp_path, monkeypatch):
    """valuation_snapshot must report None for tickers with no fundamentals and real values otherwise."""
    _patch_data_dir(monkeypatch, tmp_path)
    from data import store

    store.save_fundamentals([_make_fundamentals_record(ticker="VTI", as_of="2026-01-01", pe=26.0)])

    from data.snapshots import valuation_snapshot
    snap = valuation_snapshot(as_of=datetime(2026, 1, 1, tzinfo=UTC), tickers=["VTI", "BND"])

    assert snap.funds["VTI"]["pe"] == 26.0
    assert snap.funds["VTI"]["expense_ratio"] is not None
    assert snap.funds["BND"]["pe"] is None
    assert snap.funds["BND"]["expense_ratio"] is not None


# ---------------------------------------------------------------------------
# test_pipeline_smoke
# ---------------------------------------------------------------------------

def test_pipeline_smoke(tmp_path, monkeypatch):
    """run_pipeline must call each fetcher and survive even if one raises."""
    # Clear cache before _patch_data_dir replaces get_settings with a plain lambda
    from utils.config import get_settings
    get_settings.cache_clear()

    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    _patch_data_dir(monkeypatch, tmp_path)

    call_log: list[str] = []

    def fake_prices(tickers):
        call_log.append("prices")
        return {"AAPL": 10}

    def fake_macro(series=None):
        call_log.append("macro")
        raise RuntimeError("Macro API down!")

    def fake_filings(tickers, limit=2):
        call_log.append("filings")
        return {"AAPL": 2}

    def fake_news(tickers):
        call_log.append("news")
        return {"AAPL": 5}

    with (
        patch("data.fetch_prices.fetch_prices", fake_prices),
        patch("data.fetch_macro.fetch_macro", fake_macro),
        patch("data.fetch_filings.fetch_filings", fake_filings),
        patch("data.fetch_news.fetch_news", fake_news),
    ):
        from data.pipeline import run_pipeline
        run_pipeline(["AAPL"])  # must not raise

    assert "prices" in call_log
    assert "news" in call_log
    # monkeypatch auto-restores get_settings after the test; cache_clear on restore is handled by _patch_data_dir teardown


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _patch_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Redirect all store writes to a temp directory."""
    from utils.config import get_settings, Settings

    get_settings.cache_clear()
    original = get_settings()
    patched = Settings(
        groq_api_key=original.groq_api_key,
        news_api_key=original.news_api_key,
        fred_api_key=original.fred_api_key,
        tickers=original.tickers,
        data_dir=tmp_path / "raw",
        price_history_days=original.price_history_days,
        fred_series=original.fred_series,
        rss_feeds=original.rss_feeds,
        sec_user_agent=original.sec_user_agent,
    )
    monkeypatch.setattr("utils.config.get_settings", lambda: patched)
    monkeypatch.setattr("data.store.get_settings", lambda: patched)
