"""Multi-regime, no-lookahead backtester. No LLM, no invented numbers.

Costs are embedded continuously into each fund's own price series (matching
how expense ratios actually work — they're baked into daily NAV), then a
share-based simulation applies rebalancing and contributions on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from backtest.baselines import BASELINES
from backtest.regimes import REGIMES, BACKTEST_PROXIES
from data import store
from data.snapshots import load_close_matrix
from engine.costs import apply_annual_drag, portfolio_expense_ratio
from engine.metrics import PerformanceMetrics, compute_metrics
from engine.returns import daily_returns
from utils.config import EXPENSE_RATIOS
from utils.logger import get_logger

logger = get_logger(__name__)
UTC = timezone.utc


@dataclass
class BacktestResult:
    """Full result of simulating one allocation over one date window."""
    name: str
    start: date
    end: date
    metrics: PerformanceMetrics
    cost_drag: float             # weighted-average annual expense ratio (informational)
    equity_curve: pd.Series
    substitutions: list[str]     # e.g. ["VXUS->EFA"] if a backtest-only proxy was used


def _resolve_tickers(tickers: list[str], start: date) -> tuple[dict[str, str], list[str]]:
    """Map each ticker to itself or, if its history postdates `start`, a BACKTEST_PROXIES entry.

    Substitution is always reported, never silent.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    resolved: dict[str, str] = {}
    substitutions: list[str] = []
    for ticker in tickers:
        df = store.load_prices(ticker)
        needs_proxy = df.empty or df["date"].min() > start_ts
        if needs_proxy and ticker in BACKTEST_PROXIES:
            proxy = BACKTEST_PROXIES[ticker]
            resolved[ticker] = proxy
            substitutions.append(f"{ticker}->{proxy}")
        else:
            resolved[ticker] = ticker
    return resolved, substitutions


def _cost_adjust_close(close: pd.DataFrame, expense_ratios: dict[str, float]) -> pd.DataFrame:
    """Bake each ticker's own annual expense ratio into its price series as continuous drag."""
    raw_returns = daily_returns(close)
    adjusted = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for ticker in close.columns:
        first_price = close[ticker].iloc[0]
        dragged = apply_annual_drag(raw_returns[ticker], expense_ratios[ticker])
        cum = (1.0 + dragged).cumprod()
        adjusted.loc[close.index[0], ticker] = first_price
        adjusted.loc[close.index[1:], ticker] = (first_price * cum).values
    return adjusted


def run_backtest(
    allocation: dict[str, float],
    start: date,
    end: date,
    *,
    initial: float = 10_000.0,
    monthly_contribution: float = 0.0,
    rebalance: str = "annual",
    name: str = "backtest",
) -> BacktestResult:
    """Simulate a fixed-target allocation over [start, end] with periodic rebalancing.

    No-lookahead is enforced twice: (1) store.load_prices(until=end) never returns
    data past `end`; (2) every rebalance/contribution decision below only reads
    `close.loc[:date_t]`, asserted to contain no date after date_t.
    """
    if rebalance not in {"annual", "none"}:
        raise ValueError(f"rebalance must be 'annual' or 'none', got {rebalance!r}")

    tickers = sorted(allocation)
    resolved, substitutions = _resolve_tickers(tickers, start)
    if substitutions:
        logger.info("Backtest '%s' using proxies: %s", name, substitutions)

    fetch_tickers = list(dict.fromkeys(resolved.values()))
    until = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
    raw_close = load_close_matrix(fetch_tickers, start=start, until=until)
    raw_close = raw_close.rename(columns={proxy: original for original, proxy in resolved.items()})
    raw_close = raw_close[tickers]  # drop any accidental duplicate columns, fix order

    if raw_close.empty or len(raw_close) < 2:
        raise ValueError(f"Insufficient price data for {name} over {start}..{end}")

    # Layer 2 of no-lookahead protection: independent of store.load_prices' own
    # `until` filter (layer 1), verify the loaded data never actually extends
    # past `end`. This is a hard failure, not a warning — if it ever fires, a
    # bug upstream let future data leak into a historical simulation.
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    assert raw_close.index.max() <= end_ts, (
        f"Lookahead detected: price data extends to {raw_close.index.max()}, past backtest end {end}"
    )

    expense_ratios = {t: EXPENSE_RATIOS[t] for t in tickers}
    close = _cost_adjust_close(raw_close, expense_ratios)

    shares = {t: 0.0 for t in tickers}
    first_date = close.index[0]
    for t in tickers:
        shares[t] = (allocation[t] * initial) / close.loc[first_date, t]

    equity_values: list[float] = []
    last_rebalance_year: int | None = first_date.year
    last_month_key: tuple[int, int] | None = (first_date.year, first_date.month)

    for i, date_t in enumerate(close.index):
        if i > 0:
            month_key = (date_t.year, date_t.month)
            if monthly_contribution > 0 and month_key != last_month_key:
                for t in tickers:
                    shares[t] += (allocation[t] * monthly_contribution) / close.loc[date_t, t]
                last_month_key = month_key

            if rebalance == "annual" and date_t.year != last_rebalance_year:
                available = close.loc[:date_t]
                assert available.index.max() <= date_t, "Lookahead detected in backtest rebalance"
                port_value = sum(shares[t] * close.loc[date_t, t] for t in tickers)
                for t in tickers:
                    shares[t] = (allocation[t] * port_value) / close.loc[date_t, t]
                last_rebalance_year = date_t.year

        port_value = sum(shares[t] * close.loc[date_t, t] for t in tickers)
        equity_values.append(port_value)

    equity_curve = pd.Series(equity_values, index=close.index, name=name)
    metrics = compute_metrics(equity_curve)
    cost_drag = portfolio_expense_ratio(allocation, EXPENSE_RATIOS)

    return BacktestResult(
        name=name,
        start=start,
        end=end,
        metrics=metrics,
        cost_drag=cost_drag,
        equity_curve=equity_curve,
        substitutions=substitutions,
    )


def run_regime_suite(allocations: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Backtest every allocation against every named regime; return a tidy comparison table."""
    allocations = allocations or BASELINES
    rows: list[dict] = []

    for alloc_name, allocation in allocations.items():
        for regime_name, regime in REGIMES.items():
            try:
                result = run_backtest(allocation, regime.start, regime.end, name=f"{alloc_name}/{regime_name}")
                rows.append({
                    "allocation": alloc_name,
                    "regime": regime_name,
                    "cagr": result.metrics.cagr,
                    "sharpe": result.metrics.sharpe,
                    "sortino": result.metrics.sortino,
                    "max_drawdown": result.metrics.max_drawdown,
                    "worst_year": result.metrics.worst_year,
                    "recovery_days": result.metrics.recovery_days,
                    "cost_drag": result.cost_drag,
                    "substitutions": ", ".join(result.substitutions) or "-",
                })
            except Exception as exc:
                logger.warning("Backtest failed for %s/%s: %s", alloc_name, regime_name, exc)
                rows.append({"allocation": alloc_name, "regime": regime_name, "error": str(exc)})

    return pd.DataFrame(rows)


def _log_suite(df: pd.DataFrame) -> None:
    logger.info("=" * 100)
    logger.info("Regime backtest suite (%d runs)", len(df))
    logger.info("=" * 100)
    for _, row in df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            logger.info("%-20s %-16s ERROR: %s", row["allocation"], row["regime"], row["error"])
            continue
        logger.info(
            "%-20s %-16s CAGR=%6.2f%%  Sharpe=%5.2f  Sortino=%5.2f  MaxDD=%6.2f%%  WorstYr=%6.2f%%  Recovery=%s  Cost=%.3f%%  Proxies=%s",
            row["allocation"], row["regime"],
            row["cagr"] * 100, row["sharpe"], row["sortino"],
            row["max_drawdown"] * 100, row["worst_year"] * 100 if pd.notna(row["worst_year"]) else float("nan"),
            row["recovery_days"] if pd.notna(row["recovery_days"]) else "n/a",
            row["cost_drag"] * 100, row["substitutions"],
        )
    logger.info("=" * 100)


if __name__ == "__main__":
    suite = run_regime_suite()
    _log_suite(suite)
