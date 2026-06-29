from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import requests
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

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_NEWSAPI_LOOKBACK_DAYS = 7


def _parse_utc(date_str: str | None) -> datetime | None:
    """Parse an ISO-8601 or RFC-2822 datetime string to UTC; return None if unparseable."""
    if not date_str:
        return None
    try:
        # ISO-8601 (NewsAPI)
        import dateutil.parser
        dt = dateutil.parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        try:
            # RFC-2822 (RSS)
            dt = parsedate_to_datetime(date_str)
            return dt.astimezone(UTC)
        except Exception:
            return None


def _normalize(raw: dict, source: str, ticker: str) -> dict | None:
    """Map a raw article to the canonical news record schema; return None if published_at is missing."""
    published_at = _parse_utc(raw.get("publishedAt") or raw.get("published"))
    if published_at is None:
        # LOOKAHEAD GUARD: drop records with no parseable published_at.
        # Never fall back to fetched_at=now — that injects a future-dated row into history.
        return None

    url = (raw.get("url") or raw.get("link") or "").strip()
    if not url:
        return None

    return {
        "published_at": published_at,
        "fetched_at": datetime.now(UTC),
        "ticker": ticker,
        "headline": (raw.get("title") or "").strip(),
        "summary": (raw.get("description") or raw.get("summary") or "").strip(),
        "source": source,
        "url": url,
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, OSError)),
    reraise=True,
)
def _fetch_newsapi(ticker: str) -> list[dict]:
    """Query NewsAPI 'everything' endpoint for a ticker; return normalized records."""
    settings = get_settings()
    from datetime import timedelta
    from_date = (datetime.now(UTC) - timedelta(days=_NEWSAPI_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    resp = requests.get(
        _NEWSAPI_URL,
        params={
            "q": ticker,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 100,
            "apiKey": settings.news_api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        logger.warning("NewsAPI returned non-ok status for %s: %s", ticker, data.get("message"))
        return []

    records = []
    for article in data.get("articles", []):
        source_name = (article.get("source") or {}).get("name") or "NewsAPI"
        normalized = _normalize(article, source_name, ticker)
        if normalized:
            records.append(normalized)

    return records


def _fetch_rss(feed_url: str, ticker: str) -> list[dict]:
    """Parse one RSS feed and return records matching the ticker (keyword filter)."""
    try:
        parsed = feedparser.parse(feed_url)
    except Exception:
        logger.exception("RSS parse failed for %s", feed_url)
        return []

    records = []
    ticker_lower = ticker.lower()

    for entry in parsed.entries:
        title = (entry.get("title") or "").lower()
        summary = (entry.get("summary") or "").lower()

        # Per-ticker feeds (URL contains ticker) need no keyword filter
        is_per_ticker_feed = ticker_lower in feed_url.lower()
        if not is_per_ticker_feed and ticker_lower not in title and ticker_lower not in summary:
            continue

        raw = {
            "title": entry.get("title", ""),
            "description": entry.get("summary", ""),
            "url": entry.get("link", ""),
            "publishedAt": entry.get("published") or entry.get("updated"),
            "published": entry.get("published") or entry.get("updated"),
        }
        source_name = parsed.feed.get("title") or "RSS"
        normalized = _normalize(raw, source_name, ticker)
        if normalized:
            records.append(normalized)

    return records


def fetch_news(tickers: list[str] | None = None) -> dict[str, int]:
    """Fetch NewsAPI + RSS news per ticker, save via store, return {ticker: rows_written}."""
    settings = get_settings()
    tickers = tickers or settings.tickers
    has_news_api = bool(settings.news_api_key)

    if not has_news_api:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI; RSS feeds will still run")

    if len(tickers) > 90:
        logger.warning(
            "NewsAPI free tier allows ~100 requests/day; fetching %d tickers may exhaust the quota",
            len(tickers),
        )

    results: dict[str, int] = {}

    for i, ticker in enumerate(tickers):
        records: list[dict] = []

        # NewsAPI (one request per ticker per day — do NOT paginate)
        if has_news_api:
            try:
                api_records = _fetch_newsapi(ticker)
                records.extend(api_records)
                logger.info("NewsAPI: %d articles for %s (request %d/%d)", len(api_records), ticker, i + 1, len(tickers))
            except Exception:
                logger.exception("NewsAPI fetch failed for %s", ticker)

        # RSS feeds (no quota; run regardless of NEWS_API_KEY)
        for feed_template in settings.rss_feeds:
            feed_url = feed_template.format(ticker=ticker)
            try:
                rss_records = _fetch_rss(feed_url, ticker)
                records.extend(rss_records)
                logger.debug("RSS %s: %d articles for %s", feed_url, len(rss_records), ticker)
            except Exception:
                logger.exception("RSS fetch failed for %s / %s", ticker, feed_url)

        # Drop records without a URL (dedup key)
        records = [r for r in records if r.get("url")]

        if records:
            store.save_news(records, ticker)

        results[ticker] = len(records)
        logger.info("Total news: %d records saved for %s", len(records), ticker)

    return results


if __name__ == "__main__":
    fetch_news()
