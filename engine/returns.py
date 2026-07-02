"""Returns, volatility, covariance — pure math over price data, no LLM, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide close-price matrix (date index, ticker columns) to simple daily returns."""
    return close.pct_change().dropna(how="all")


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return from a series of daily simple returns."""
    if returns.empty:
        raise ValueError("Cannot annualize an empty returns series")
    n = len(returns)
    cumulative = (1.0 + returns).prod()
    return float(cumulative ** (_TRADING_DAYS / n) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility (std dev) from a series of daily simple returns."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(_TRADING_DAYS))


def covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix from a wide daily-returns DataFrame."""
    return returns.cov() * _TRADING_DAYS


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted daily return series for a fixed-weight (no intra-period rebalancing) portfolio.

    Rebalancing schedules are the backtester's concern (backtest/engine.py);
    this is the building block it composes over each rebalance sub-period.
    """
    missing = [t for t in weights if t not in returns.columns]
    if missing:
        raise ValueError(f"Tickers missing from returns data: {missing}")
    weighted = sum(returns[ticker] * weight for ticker, weight in weights.items())
    return weighted.rename("portfolio")
