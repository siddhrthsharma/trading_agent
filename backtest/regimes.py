"""Named historical stress-test windows and the holdout period — no math, just constants."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Regime:
    """A named historical window to backtest against."""
    name: str
    start: date
    end: date
    description: str


REGIMES: dict[str, Regime] = {
    "gfc_2008": Regime(
        name="gfc_2008",
        start=date(2007, 10, 1),
        end=date(2013, 3, 31),
        description="Global Financial Crisis through recovery — the S&P 500 didn't reclaim its Oct 2007 high until March 2013.",
    ),
    "covid_2020": Regime(
        name="covid_2020",
        start=date(2020, 1, 1),
        end=date(2021, 12, 31),
        description="COVID crash (fastest 30% drawdown in history) and the sharp recovery that followed.",
    ),
    "rate_shock_2022": Regime(
        name="rate_shock_2022",
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
        description="Fastest Fed hiking cycle in decades — both stocks and bonds fell together.",
    ),
    "bull_2010s": Regime(
        name="bull_2010s",
        start=date(2012, 1, 1),
        end=date(2019, 12, 31),
        description="Long, low-volatility bull market — the regime it's easiest to accidentally overfit to.",
    ),
}

# Backtest-only ticker substitution: some FULL_UNIVERSE ETFs postdate the 2008
# regime's start (VXUS launched 2011-01-28). EFA (developed ex-US, since 2001)
# stands in so the GFC window can still be tested; substitution — and that it
# happened — is reported explicitly in backtest output, never silent.
BACKTEST_PROXIES: dict[str, str] = {
    "VXUS": "EFA",
}

# Last 12 months — never tune allocations or prompts against this window.
HOLDOUT_START: date = date(2025, 7, 1)
