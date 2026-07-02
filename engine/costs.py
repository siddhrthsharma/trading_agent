"""Expense-ratio and tax drag — pure math, no LLM, no network."""
from __future__ import annotations

import pandas as pd

_TRADING_DAYS = 252


def portfolio_expense_ratio(allocation: dict[str, float], expense_ratios: dict[str, float]) -> float:
    """Weighted-average annual expense ratio for an allocation.

    Raises ValueError if any allocated ticker has no known expense ratio —
    silently defaulting to 0 would understate cost drag.
    """
    missing = [t for t, w in allocation.items() if w > 0 and t not in expense_ratios]
    if missing:
        raise ValueError(f"No expense ratio known for: {sorted(missing)}")
    return sum(weight * expense_ratios[ticker] for ticker, weight in allocation.items())


def apply_annual_drag(daily_returns: pd.Series, annual_drag: float) -> pd.Series:
    """Subtract a compounding-consistent daily cost drag from a daily-returns series.

    `annual_drag` is a positive fraction (e.g. 0.003 for 0.3%/year expense ratio).
    After exactly 252 trading days of otherwise-flat returns, cumulative value is
    reduced by exactly `annual_drag`.
    """
    daily_factor = (1.0 - annual_drag) ** (1.0 / _TRADING_DAYS)
    return (1.0 + daily_returns) * daily_factor - 1.0
