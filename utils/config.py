import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).parent.parent

# --- Asset universe (low-cost, liquid, broad — from CURSOR_PROJECT_SPEC.md) ---
# Only instruments the optimizer is allowed to *recommend holding* live in the
# universe lists. Benchmarks are kept separate so they never leak into an
# allocation proposal (e.g. VOO/IVV/AGG are ~near-duplicates of holdings).

# Three-fund baseline — what you'd actually hold.
CORE_UNIVERSE: list[str] = ["VTI", "VXUS", "BND"]

# Full allocation universe — every instrument the optimizer may propose.
FULL_UNIVERSE: list[str] = [
    "VTI", "QQQ",                        # US equity
    "VEA", "VXUS",                       # international developed
    "VWO",                               # emerging markets
    "BND", "TLT", "BIL", "SHY", "SCHP",  # fixed income
    "VNQ", "GLD",                        # alternatives
]

# Price history / comparison only — never proposed as a holding.
# EFA (developed-markets ETF, inception 2001) stands in for VXUS pre-2011 in backtests.
BENCHMARKS: list[str] = ["SPY", "VOO", "IVV", "AGG", "EFA"]

# Static expense ratios (annual %, as decimals) — yfinance does not reliably report these.
# Source: fund provider fact sheets (CURSOR_PROJECT_SPEC.md §Asset Universe).
EXPENSE_RATIOS: dict[str, float] = {
    "VTI": 0.0003, "QQQ": 0.0020,
    "VEA": 0.0005, "VXUS": 0.0007, "VWO": 0.0008,
    "BND": 0.0003, "TLT": 0.0015, "BIL": 0.001356, "SHY": 0.0015, "SCHP": 0.0003,
    "VNQ": 0.0012, "GLD": 0.0040,
    "SPY": 0.0009, "VOO": 0.0003, "IVV": 0.0003, "AGG": 0.0003, "EFA": 0.0032,
}

# Same index exposure, different wrapper (mutual fund share class).
MUTUAL_FUND_EQUIVALENTS: dict[str, str] = {
    "VOO": "VFIAX",
    "VTI": "VTSAX",
    "BND": "VBTLX",
}

# The three-fund portfolio — the sacred baseline.
THREE_FUND: dict[str, str] = {
    "US_EQUITY": "VTI",
    "INTL_EQUITY": "VXUS",
    "BONDS": "BND",
}

# Default tickers the data pipeline fetches: everything we may hold + benchmarks.
_DEFAULT_TICKERS: list[str] = list(dict.fromkeys(FULL_UNIVERSE + BENCHMARKS))


@dataclass(frozen=True)
class Settings:
    """Immutable runtime config loaded from .env + defaults."""

    groq_api_key: str
    news_api_key: str
    fred_api_key: str
    tickers: list[str]
    data_dir: Path
    price_history_days: int
    fred_series: dict[str, str]
    rss_feeds: list[str]
    sec_user_agent: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the Settings singleton from environment + module defaults."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    news_api_key = os.getenv("NEWS_API_KEY", "")
    fred_api_key = os.getenv("FRED_API_KEY", "")

    tickers_env = os.getenv("TICKERS", "")
    tickers: list[str] = (
        [t.strip() for t in tickers_env.split(",") if t.strip()]
        if tickers_env
        else list(_DEFAULT_TICKERS)
    )

    sec_user_agent = os.getenv("SEC_USER_AGENT", "TradingAgent siddharthsharmaa25@gmail.com")

    return Settings(
        groq_api_key=groq_api_key,
        news_api_key=news_api_key,
        fred_api_key=fred_api_key,
        tickers=tickers,
        data_dir=_PROJECT_ROOT / "data" / "raw",
        price_history_days=9125,  # ~25 years — long enough to cover the 2008 GFC regime
        fred_series={
            "FEDFUNDS": "Federal Funds Rate",
            "CPIAUCSL": "Consumer Price Index",
            "UNRATE": "Unemployment Rate",
            "DGS10": "10-Year Treasury Yield",
            "DGS2": "2-Year Treasury Yield",
        },
        rss_feeds=[
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        ],
        sec_user_agent=sec_user_agent,
    )


# Convenience re-export
TICKERS: list[str] = get_settings().tickers
