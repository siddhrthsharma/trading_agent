from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(tickers: list[str] | None = None) -> None:
    """Run all data fetchers sequentially for the given tickers and log results."""
    from utils.config import get_settings

    settings = get_settings()
    tickers = tickers or settings.tickers
    logger.info("Starting pipeline for tickers: %s", tickers)

    summary: dict[str, dict[str, int]] = {}

    # 1. Prices — no API key required
    try:
        from data.fetch_prices import fetch_prices
        price_results = fetch_prices(tickers)
        summary["prices"] = price_results
        logger.info("Prices done: %s", price_results)
    except Exception:
        logger.exception("Prices fetcher failed")

    # 2. Macro — requires FRED_API_KEY
    if settings.fred_api_key:
        try:
            from data.fetch_macro import fetch_macro
            macro_results = fetch_macro()
            summary["macro"] = macro_results
            logger.info("Macro done: %s", macro_results)
        except Exception:
            logger.exception("Macro fetcher failed")
    else:
        logger.warning("FRED_API_KEY not set — skipping macro fetcher")

    # 3. Filings — no API key, but requires SEC user-agent
    try:
        from data.fetch_filings import fetch_filings
        filing_results = fetch_filings(tickers)
        summary["filings"] = filing_results
        logger.info("Filings done: %s", filing_results)
    except Exception:
        logger.exception("Filings fetcher failed")

    # 4. News — runs last to preserve NewsAPI daily quota
    try:
        from data.fetch_news import fetch_news
        news_results = fetch_news(tickers)
        summary["news"] = news_results
        logger.info("News done: %s", news_results)
    except Exception:
        logger.exception("News fetcher failed")

    _log_summary(summary, tickers)


def _log_summary(summary: dict[str, dict[str, int]], tickers: list[str]) -> None:
    """Log a final table of rows written per source per ticker."""
    logger.info("=" * 60)
    logger.info("Pipeline summary")
    logger.info("=" * 60)
    sources = list(summary.keys())
    header = f"{'Ticker':<10}" + "".join(f"{s:<12}" for s in sources)
    logger.info(header)
    for ticker in tickers:
        row = f"{ticker:<10}" + "".join(f"{summary[s].get(ticker, '-'):<12}" for s in sources)
        logger.info(row)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
