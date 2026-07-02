"""Backtest tests — regimes/baselines are constants; engine tests enforce no-lookahead."""
from __future__ import annotations

from datetime import date, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest import baselines, regimes

UTC = timezone.utc


def _patch_data_dir(monkeypatch, tmp_path: Path) -> None:
    """Redirect all store writes to a temp directory (mirrors tests/test_data.py)."""
    from utils.config import Settings, get_settings

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


def _save_price_fixture(ticker: str, dates: list[str], closes: list[float]) -> None:
    from data import store

    idx = pd.to_datetime(dates, utc=True)
    df = pd.DataFrame({
        "date": idx,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000] * len(closes),
        "ticker": [ticker] * len(closes),
        "fetched_at": [pd.Timestamp("2026-01-01", tz="UTC")] * len(closes),
    })
    store.save_prices(df, ticker)


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

def test_all_regimes_have_start_before_end():
    for name, regime in regimes.REGIMES.items():
        assert regime.start < regime.end, f"{name} has start >= end"


def test_regime_names_match_dict_keys():
    for key, regime in regimes.REGIMES.items():
        assert regime.name == key


def test_expected_regimes_present():
    expected = {"gfc_2008", "covid_2020", "rate_shock_2022", "bull_2010s"}
    assert expected <= set(regimes.REGIMES)


def test_no_regime_overlaps_holdout():
    """No named regime window may extend into the holdout — never tune against it."""
    for name, regime in regimes.REGIMES.items():
        assert regime.end < regimes.HOLDOUT_START, f"{name} overlaps the holdout period"


def test_backtest_proxies_reference_full_universe_tickers():
    from utils.config import FULL_UNIVERSE
    for original, proxy in regimes.BACKTEST_PROXIES.items():
        assert original in FULL_UNIVERSE


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def test_all_baselines_sum_to_one():
    for name, alloc in baselines.BASELINES.items():
        assert sum(alloc.values()) == pytest.approx(1.0), f"{name} doesn't sum to 1.0"


def test_sixty_forty_is_actually_sixty_forty():
    alloc = baselines.BASELINES["sixty_forty"]
    equity = alloc.get("VTI", 0) + alloc.get("VXUS", 0)
    bonds = alloc.get("BND", 0)
    assert equity == pytest.approx(0.60)
    assert bonds == pytest.approx(0.40)


def test_all_equity_baseline_has_no_bonds():
    alloc = baselines.BASELINES["all_equity"]
    assert alloc.get("BND", 0) == 0


# ---------------------------------------------------------------------------
# backtest/engine.py — no-lookahead (keystone test)
# ---------------------------------------------------------------------------

def test_backtest_raises_if_data_leaks_past_end(tmp_path, monkeypatch):
    """If a bug let price data past `end` slip through, run_backtest must refuse to use it."""
    _patch_data_dir(monkeypatch, tmp_path)
    _save_price_fixture("A", ["2020-01-01", "2020-01-02"], [100.0, 101.0])

    from backtest import engine

    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0})

    # Simulate a loader bug: return a frame that extends past the requested end date.
    leaked = pd.DataFrame(
        {"A": [100.0, 101.0, 999.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-06-01"], utc=True),
    )
    monkeypatch.setattr("backtest.engine.load_close_matrix", lambda *a, **k: leaked)

    with pytest.raises(AssertionError):
        engine.run_backtest({"A": 1.0}, date(2020, 1, 1), date(2020, 1, 2))


# ---------------------------------------------------------------------------
# backtest/engine.py — simulation correctness
# ---------------------------------------------------------------------------

def test_backtest_buy_and_hold_single_asset_reproduces_own_return(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_price_fixture("A", ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], [100.0, 110.0, 121.0, 90.0])

    from backtest import engine
    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0})

    result = engine.run_backtest({"A": 1.0}, date(2020, 1, 1), date(2020, 1, 4), initial=1000.0, rebalance="none")

    expected = [1000.0, 1100.0, 1210.0, 900.0]
    assert result.equity_curve.tolist() == pytest.approx(expected)


def test_backtest_annual_rebalance_two_asset_hand_computed(tmp_path, monkeypatch):
    """Hand-computed: two assets, one rebalance at the 2020->2021 year boundary."""
    _patch_data_dir(monkeypatch, tmp_path)
    dates = ["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"]
    _save_price_fixture("A", dates, [100.0, 150.0, 150.0, 180.0])
    _save_price_fixture("B", dates, [100.0, 100.0, 100.0, 90.0])

    from backtest import engine
    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0, "B": 0.0})

    result = engine.run_backtest(
        {"A": 0.5, "B": 0.5}, date(2020, 1, 1), date(2021, 6, 1),
        initial=1000.0, rebalance="annual",
    )

    # day0: 5 shares each @100 = 1000
    # day1 (same year, no rebalance): 5*150 + 5*100 = 1250
    # day2 (year boundary -> rebalance at unchanged prices): value unchanged = 1250
    # day3: new shares (4.1666.., 6.25) @ (180, 90) = 750 + 562.5 = 1312.5
    expected = [1000.0, 1250.0, 1250.0, 1312.5]
    assert result.equity_curve.tolist() == pytest.approx(expected)


def test_backtest_cost_drag_reduces_returns(tmp_path, monkeypatch):
    """A nonzero expense ratio must produce a lower final value than the zero-cost case."""
    _patch_data_dir(monkeypatch, tmp_path)
    dates = [f"2020-01-{d:02d}" for d in range(1, 11)]
    closes = [100.0 * (1.001 ** i) for i in range(10)]
    _save_price_fixture("A", dates, closes)

    from backtest import engine

    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0})
    free = engine.run_backtest({"A": 1.0}, date(2020, 1, 1), date(2020, 1, 10), rebalance="none")

    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.05})  # 5% annual ER — exaggerated for a visible effect
    costly = engine.run_backtest({"A": 1.0}, date(2020, 1, 1), date(2020, 1, 10), rebalance="none")

    assert costly.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert costly.cost_drag == pytest.approx(0.05)


def test_backtest_insufficient_data_raises(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_price_fixture("A", ["2020-01-01"], [100.0])

    from backtest import engine
    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0})

    with pytest.raises(ValueError):
        engine.run_backtest({"A": 1.0}, date(2020, 1, 1), date(2020, 1, 10))


# ---------------------------------------------------------------------------
# backtest/engine.py — regime suite end-to-end (on fixtures)
# ---------------------------------------------------------------------------

def test_run_regime_suite_end_to_end_on_fixtures(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    dates = [f"2020-01-{d:02d}" for d in range(1, 11)]
    closes = [100.0 * (1.001 ** i) for i in range(10)]
    _save_price_fixture("A", dates, closes)

    from backtest import engine
    from backtest.regimes import Regime

    monkeypatch.setattr("backtest.engine.EXPENSE_RATIOS", {"A": 0.0})
    monkeypatch.setattr("backtest.engine.REGIMES", {
        "mini_regime": Regime(name="mini_regime", start=date(2020, 1, 1), end=date(2020, 1, 10), description="test"),
    })

    suite = engine.run_regime_suite(allocations={"all_a": {"A": 1.0}})

    assert len(suite) == 1
    row = suite.iloc[0]
    assert row["allocation"] == "all_a"
    assert row["regime"] == "mini_regime"
    assert "error" not in suite.columns or pd.isna(row.get("error"))
    assert row["cagr"] > 0  # closes are monotonically increasing
