"""Performance metrics — Sharpe, Sortino, CAGR, max drawdown, recovery time.

Pure math over an equity curve, no LLM, no network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    """Full risk/return picture for a backtested equity curve."""
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float        # negative fraction, e.g. -0.51
    worst_year: float          # worst calendar-year return; nan if <1 full year-over-year span
    recovery_days: int | None  # None if the max drawdown never recovered within the window


def _annualized_sharpe_like(daily_excess: pd.Series, downside_only: bool) -> float:
    """Shared Sharpe/Sortino computation: annualized mean excess return / annualized deviation."""
    if downside_only:
        deviation_input = daily_excess[daily_excess < 0]
    else:
        deviation_input = daily_excess
    if len(deviation_input) < 2:
        return float("nan")
    std = deviation_input.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((daily_excess.mean() / std) * np.sqrt(_TRADING_DAYS))


def _max_drawdown_and_recovery(equity_curve: pd.Series) -> tuple[float, int | None]:
    """Max drawdown (negative fraction) and days to recover the prior peak, if it recovered."""
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    max_dd = float(drawdown.min())

    trough_date = drawdown.idxmin()
    peak_value = running_peak.loc[trough_date]
    after_trough = equity_curve.loc[trough_date:]
    recovered = after_trough[(after_trough.index > trough_date) & (after_trough >= peak_value)]
    recovery_days = int((recovered.index[0] - trough_date).days) if not recovered.empty else None

    return max_dd, recovery_days


def _worst_year(equity_curve: pd.Series) -> float:
    """Worst calendar-year return; nan if there isn't at least one full year-end-to-year-end span."""
    yearly = equity_curve.resample("YE").last()
    yearly_returns = yearly.pct_change().dropna()
    if yearly_returns.empty:
        return float("nan")
    return float(yearly_returns.min())


def compute_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> PerformanceMetrics:
    """Compute the full performance-metrics suite from a portfolio equity curve.

    `equity_curve` must be indexed by date (ascending) with the portfolio's
    dollar value (or normalized value) at each observation.
    """
    if len(equity_curve) < 2:
        raise ValueError("Need at least 2 observations to compute metrics")

    daily_ret = equity_curve.pct_change().dropna()
    daily_rf = risk_free_rate / _TRADING_DAYS
    daily_excess = daily_ret - daily_rf

    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = total_days / 365.25
    if years <= 0:
        raise ValueError("equity_curve must span a positive amount of time")
    cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0)

    volatility = float(daily_ret.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if len(daily_ret) > 1 else float("nan")
    sharpe = _annualized_sharpe_like(daily_excess, downside_only=False)
    sortino = _annualized_sharpe_like(daily_excess, downside_only=True)
    max_dd, recovery_days = _max_drawdown_and_recovery(equity_curve)
    worst_year = _worst_year(equity_curve)

    return PerformanceMetrics(
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        worst_year=worst_year,
        recovery_days=recovery_days,
    )
