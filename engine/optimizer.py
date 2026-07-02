"""Mean-variance / risk-parity optimizer — pure math over price data, no LLM, no network.

The optimizer's output is never handed to the allocator as a final answer —
it's additional context alongside the core/-computed baseline (see
agents/allocator_agent.py and CLAUDE.md's math/LLM separation rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from engine.returns import annualized_return, annualized_volatility, daily_returns, portfolio_returns

_OBJECTIVES = {"max_sharpe", "min_volatility"}


@dataclass(frozen=True)
class OptimizationResult:
    """Weights and expected performance from a single optimization run."""
    weights: dict[str, float]
    expected_return: float
    expected_vol: float
    sharpe: float
    method: str   # "max_sharpe" | "min_volatility" | "inverse_vol"


def mean_variance(
    close: pd.DataFrame,
    *,
    bounds: tuple[float, float] = (0.0, 0.60),
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
) -> OptimizationResult:
    """Mean-variance optimization (PyPortfolioOpt) over a wide close-price matrix."""
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")

    mu = expected_returns.mean_historical_return(close)
    cov = risk_models.sample_cov(close)
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)

    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    else:
        ef.min_volatility()

    weights = dict(ef.clean_weights())
    expected_return, expected_vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)

    return OptimizationResult(
        weights=weights,
        expected_return=float(expected_return),
        expected_vol=float(expected_vol),
        sharpe=float(sharpe),
        method=objective,
    )


def inverse_volatility(close: pd.DataFrame, risk_free_rate: float = 0.0) -> OptimizationResult:
    """Simple risk-parity proxy: weight each asset inversely to its own volatility."""
    rets = daily_returns(close)
    vols = {ticker: annualized_volatility(rets[ticker]) for ticker in rets.columns}

    zero_vol = [t for t, v in vols.items() if v == 0 or pd.isna(v)]
    if zero_vol:
        raise ValueError(f"Cannot compute inverse volatility for zero/undefined vol: {zero_vol}")

    inv = {ticker: 1.0 / vol for ticker, vol in vols.items()}
    total = sum(inv.values())
    weights = {ticker: w / total for ticker, w in inv.items()}

    port_rets = portfolio_returns(rets, weights)
    expected_return = annualized_return(port_rets)
    expected_vol = annualized_volatility(port_rets)
    sharpe = (
        (expected_return - risk_free_rate) / expected_vol
        if expected_vol and not pd.isna(expected_vol)
        else float("nan")
    )

    return OptimizationResult(
        weights=weights,
        expected_return=expected_return,
        expected_vol=expected_vol,
        sharpe=sharpe,
        method="inverse_vol",
    )
