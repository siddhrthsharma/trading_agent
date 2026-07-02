"""Engine tests — pure math, no LLM, no network. Golden values first."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from engine import allocation, costs, metrics, optimizer, returns, signals

UTC = timezone.utc


def _series(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates, utc=True),
        "value": values,
    })


# ---------------------------------------------------------------------------
# latest_value
# ---------------------------------------------------------------------------

def test_latest_value_returns_most_recent():
    s = _series(["2026-01-01", "2026-03-01", "2026-02-01"], [1.0, 3.0, 2.0])
    assert signals.latest_value(s) == 3.0


def test_latest_value_empty_raises():
    s = _series([], [])
    with pytest.raises(ValueError):
        signals.latest_value(s)


# ---------------------------------------------------------------------------
# yoy_change
# ---------------------------------------------------------------------------

def test_yoy_change_pct():
    s = _series(["2025-01-01", "2025-06-01", "2026-01-01"], [100.0, 102.0, 105.0])
    # latest = 105.0 at 2026-01-01, year-ago ~= 2025-01-02 -> closest is 2025-01-01 (100.0)
    result = signals.yoy_change(s, mode="pct")
    assert result == pytest.approx((105.0 - 100.0) / 100.0)


def test_yoy_change_diff():
    s = _series(["2025-01-01", "2026-01-01"], [4.0, 5.5])
    result = signals.yoy_change(s, mode="diff")
    assert result == pytest.approx(1.5)


def test_yoy_change_no_year_ago_data_raises():
    s = _series(["2026-01-01"], [5.0])
    with pytest.raises(ValueError):
        signals.yoy_change(s)


def test_yoy_change_unknown_mode_raises():
    s = _series(["2025-01-01", "2026-01-01"], [1.0, 2.0])
    with pytest.raises(ValueError):
        signals.yoy_change(s, mode="bogus")


# ---------------------------------------------------------------------------
# build_macro_snapshot
# ---------------------------------------------------------------------------

def test_build_macro_snapshot_normal_curve():
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    series = {
        "FEDFUNDS": _series(["2025-01-01", "2026-01-01"], [5.0, 5.0]),
        "DGS10": _series(["2025-01-01", "2026-01-01"], [4.0, 4.5]),
        "DGS2": _series(["2025-01-01", "2026-01-01"], [4.0, 4.0]),
        "CPIAUCSL": _series(["2025-01-01", "2026-01-01"], [300.0, 309.0]),
        "UNRATE": _series(["2025-01-01", "2026-01-01"], [4.0, 4.2]),
    }
    snap = signals.build_macro_snapshot(series, as_of)

    assert snap.as_of == as_of
    assert snap.fed_funds == 5.0
    assert snap.dgs10 == 4.5
    assert snap.dgs2 == 4.0
    assert snap.yield_curve_spread == pytest.approx(0.5)
    assert snap.curve_inverted is False
    assert snap.cpi_yoy == pytest.approx((309.0 - 300.0) / 300.0)
    assert snap.real_fed_funds == pytest.approx(5.0 - snap.cpi_yoy * 100)
    assert snap.unemployment == 4.2
    assert snap.unemployment_change_1y == pytest.approx(0.2)


def test_build_macro_snapshot_inverted_curve():
    as_of = datetime(2022, 9, 30, tzinfo=UTC)
    series = {
        "FEDFUNDS": _series(["2021-09-30", "2022-09-30"], [0.25, 3.0]),
        "DGS10": _series(["2021-09-30", "2022-09-30"], [1.5, 3.7]),
        "DGS2": _series(["2021-09-30", "2022-09-30"], [0.3, 4.1]),
        "CPIAUCSL": _series(["2021-09-30", "2022-09-30"], [274.0, 296.8]),
        "UNRATE": _series(["2021-09-30", "2022-09-30"], [4.7, 3.5]),
    }
    snap = signals.build_macro_snapshot(series, as_of)

    assert snap.yield_curve_spread == pytest.approx(3.7 - 4.1)
    assert snap.curve_inverted is True
    assert snap.unemployment_change_1y == pytest.approx(3.5 - 4.7)


# ---------------------------------------------------------------------------
# apply_tilts
# ---------------------------------------------------------------------------

def test_apply_tilts_sums_to_one():
    baseline = {"VTI": 0.54, "VXUS": 0.26, "BND": 0.20}
    tilts = {"VTI": 0.05, "BND": -0.05}
    result = allocation.apply_tilts(baseline, tilts)
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_tilts_clamps_hallucinated_tilt():
    """A wildly out-of-bounds tilt must still clamp to MAX_TILT, not blow up the allocation."""
    baseline = {"VTI": 0.54, "VXUS": 0.26, "BND": 0.20}
    tilts = {"VTI": 0.90}  # hallucinated — way beyond MAX_TILT
    result = allocation.apply_tilts(baseline, tilts)
    expected_vti = (0.54 + allocation.MAX_TILT) / (1.0 + allocation.MAX_TILT)
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["VTI"] == pytest.approx(expected_vti)


def test_apply_tilts_floors_negative_weight():
    baseline = {"VTI": 0.05, "VXUS": 0.25, "BND": 0.70}
    tilts = {"VTI": -0.10}  # would go negative before flooring
    result = allocation.apply_tilts(baseline, tilts)
    assert "VTI" not in result or result["VTI"] >= 0
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_tilts_introduces_new_ticker():
    """Tilting a ticker absent from the baseline (e.g. BIL for a crisis tilt) should add it."""
    baseline = {"VTI": 0.60, "VXUS": 0.20, "BND": 0.20}
    tilts = {"BIL": 0.10}
    result = allocation.apply_tilts(baseline, tilts)
    assert result.get("BIL", 0) > 0
    assert sum(result.values()) == pytest.approx(1.0)


def test_apply_tilts_zero_total_raises():
    with pytest.raises(ValueError):
        allocation.apply_tilts({"VTI": 0.10}, {"VTI": -0.10})


# ---------------------------------------------------------------------------
# check_rules
# ---------------------------------------------------------------------------

def test_check_rules_clean_three_fund_passes():
    alloc = {"VTI": 0.54, "VXUS": 0.26, "BND": 0.20}
    assert allocation.check_rules(alloc) == []


def test_check_rules_flags_duplicate_index():
    alloc = {"VOO": 0.50, "IVV": 0.50}
    violations = allocation.check_rules(alloc)
    rules = {v.rule for v in violations}
    assert "duplicate_index" in rules
    assert "unknown_ticker" in rules  # VOO/IVV are BENCHMARKS, not in FULL_UNIVERSE


def test_check_rules_flags_qqq_concentration():
    alloc = {"QQQ": 0.30, "BND": 0.70}
    violations = allocation.check_rules(alloc)
    assert any(v.rule == "qqq_concentration" and v.severity == "serious" for v in violations)


def test_check_rules_flags_gld_as_minor():
    alloc = {"GLD": 0.15, "VTI": 0.85}
    violations = allocation.check_rules(alloc)
    assert any(v.rule == "gld_concentration" and v.severity == "minor" for v in violations)


def test_check_rules_flags_single_position_concentration():
    """A single-fund bet beyond even the all-equity baseline (VTI 70%) must flag."""
    alloc = {"VTI": 0.85, "BND": 0.15}
    violations = allocation.check_rules(alloc)
    assert any(v.rule == "concentration" for v in violations)


def test_check_rules_all_equity_baseline_does_not_flag_concentration():
    """core.portfolios.all_equity() (VTI 70%) must not trip the concentration rule."""
    from core.portfolios import all_equity
    violations = allocation.check_rules(all_equity())
    assert not any(v.rule == "concentration" for v in violations)


def test_check_rules_flags_bad_weight_sum():
    alloc = {"VTI": 0.30, "BND": 0.30}  # sums to 0.6
    violations = allocation.check_rules(alloc)
    assert any(v.rule == "weight_sum" for v in violations)


# ---------------------------------------------------------------------------
# engine/returns.py
# ---------------------------------------------------------------------------

def test_daily_returns_pct_change():
    close = pd.DataFrame({
        "A": [100.0, 110.0, 121.0],
        "B": [50.0, 45.0, 45.0],
    }, index=pd.date_range("2026-01-01", periods=3))
    result = returns.daily_returns(close)
    assert len(result) == 2
    assert result["A"].iloc[0] == pytest.approx(0.10)
    assert result["B"].iloc[1] == pytest.approx(0.0)


def test_annualized_return_zero_for_flat_series():
    r = pd.Series([0.0] * 252)
    assert returns.annualized_return(r) == pytest.approx(0.0)


def test_annualized_return_doubling_series():
    """252 identical daily returns that compound to exactly 2x → ~100% annualized."""
    daily_r = 2.0 ** (1 / 252) - 1
    r = pd.Series([daily_r] * 252)
    assert returns.annualized_return(r) == pytest.approx(1.0, rel=1e-6)


def test_annualized_return_empty_raises():
    with pytest.raises(ValueError):
        returns.annualized_return(pd.Series([], dtype=float))


def test_annualized_volatility_matches_manual_calc():
    r = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015])
    expected = r.std(ddof=1) * np.sqrt(252)
    assert returns.annualized_volatility(r) == pytest.approx(expected)


def test_annualized_volatility_single_point_is_nan():
    assert np.isnan(returns.annualized_volatility(pd.Series([0.01])))


def test_covariance_is_annualized():
    r = pd.DataFrame({"A": [0.01, -0.01, 0.02], "B": [0.02, 0.0, -0.01]})
    result = returns.covariance(r)
    expected = r.cov() * 252
    assert result.loc["A", "A"] == pytest.approx(expected.loc["A", "A"])


def test_portfolio_returns_weighted_sum():
    r = pd.DataFrame({
        "A": [0.10, 0.0],
        "B": [0.0, 0.10],
    })
    result = returns.portfolio_returns(r, {"A": 0.5, "B": 0.5})
    assert result.iloc[0] == pytest.approx(0.05)
    assert result.iloc[1] == pytest.approx(0.05)


def test_portfolio_returns_missing_ticker_raises():
    r = pd.DataFrame({"A": [0.01, 0.02]})
    with pytest.raises(ValueError):
        returns.portfolio_returns(r, {"A": 0.5, "MISSING": 0.5})


# ---------------------------------------------------------------------------
# engine/metrics.py
# ---------------------------------------------------------------------------

def test_compute_metrics_cagr_monthly_compounding():
    """1% growth per month for exactly 12 months → CAGR ≈ (1.01)^12 - 1."""
    dates = pd.date_range("2020-01-01", periods=13, freq="MS")  # 13 points = 12 intervals
    values = [100.0 * (1.01 ** i) for i in range(13)]
    curve = pd.Series(values, index=dates)
    result = metrics.compute_metrics(curve)
    # ~1 year span (dates[-1]-dates[0]).days / 365.25 is close to but not exactly 1.0
    years = (dates[-1] - dates[0]).days / 365.25
    expected_cagr = (values[-1] / values[0]) ** (1 / years) - 1
    assert result.cagr == pytest.approx(expected_cagr)


def test_compute_metrics_max_drawdown_and_recovery_hand_computed():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    values = [100.0, 80.0, 50.0, 90.0, 110.0]
    curve = pd.Series(values, index=dates)
    result = metrics.compute_metrics(curve)

    assert result.max_drawdown == pytest.approx(-0.5)   # trough at day 3 (50 vs peak 100)
    assert result.recovery_days == 2                     # day3(idx2) -> day5(idx4) = 2 days


def test_compute_metrics_no_recovery_returns_none():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    values = [100.0, 50.0, 60.0]  # never gets back above 100
    curve = pd.Series(values, index=dates)
    result = metrics.compute_metrics(curve)
    assert result.recovery_days is None


def test_compute_metrics_sharpe_and_sortino_hand_computed():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    daily_r = [0.01, -0.01, 0.02, -0.005, 0.015]
    values = [100.0]
    for r in daily_r:
        values.append(values[-1] * (1 + r))
    curve = pd.Series(values, index=dates)
    result = metrics.compute_metrics(curve)

    r_series = pd.Series(daily_r)
    expected_sharpe = (r_series.mean() / r_series.std(ddof=1)) * np.sqrt(252)
    downside = r_series[r_series < 0]
    expected_sortino = (r_series.mean() / downside.std(ddof=1)) * np.sqrt(252)

    assert result.sharpe == pytest.approx(expected_sharpe)
    assert result.sortino == pytest.approx(expected_sortino)


def test_compute_metrics_zero_volatility_returns_nan_sharpe():
    """A perfectly flat equity curve must not raise a divide-by-zero — sharpe/sortino are nan."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    curve = pd.Series([100.0] * 10, index=dates)
    result = metrics.compute_metrics(curve)

    assert result.volatility == pytest.approx(0.0)
    assert np.isnan(result.sharpe)
    assert np.isnan(result.sortino)
    assert result.max_drawdown == pytest.approx(0.0)


def test_compute_metrics_too_few_points_raises():
    curve = pd.Series([100.0], index=pd.date_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        metrics.compute_metrics(curve)


def test_worst_year_hand_computed():
    dates = pd.to_datetime(["2019-12-31", "2020-12-31", "2021-12-31"], utc=False)
    curve = pd.Series([100.0, 70.0, 98.0], index=dates)
    result = metrics._worst_year(curve)
    assert result == pytest.approx(-0.30)


def test_worst_year_insufficient_span_is_nan():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    curve = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0], index=dates)
    result = metrics._worst_year(curve)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# engine/costs.py
# ---------------------------------------------------------------------------

def test_portfolio_expense_ratio_weighted_average():
    allocation_ = {"VTI": 0.6, "BND": 0.4}
    ers = {"VTI": 0.0003, "BND": 0.0003}
    result = costs.portfolio_expense_ratio(allocation_, ers)
    assert result == pytest.approx(0.0003)


def test_portfolio_expense_ratio_mixed_costs():
    allocation_ = {"VTI": 0.5, "QQQ": 0.5}
    ers = {"VTI": 0.0003, "QQQ": 0.0020}
    result = costs.portfolio_expense_ratio(allocation_, ers)
    assert result == pytest.approx(0.5 * 0.0003 + 0.5 * 0.0020)


def test_portfolio_expense_ratio_missing_ticker_raises():
    allocation_ = {"VTI": 0.5, "MYSTERY": 0.5}
    ers = {"VTI": 0.0003}
    with pytest.raises(ValueError):
        costs.portfolio_expense_ratio(allocation_, ers)


def test_apply_annual_drag_reduces_flat_return_by_exact_amount():
    """252 days of 0% returns with a 0.5% annual drag → final value = 1 - 0.005 exactly."""
    flat = pd.Series([0.0] * 252)
    dragged = costs.apply_annual_drag(flat, annual_drag=0.005)
    cumulative = (1 + dragged).prod()
    assert cumulative == pytest.approx(1 - 0.005)


def test_apply_annual_drag_zero_drag_is_noop():
    r = pd.Series([0.01, -0.02, 0.03])
    dragged = costs.apply_annual_drag(r, annual_drag=0.0)
    assert dragged.tolist() == pytest.approx(r.tolist())


# ---------------------------------------------------------------------------
# engine/optimizer.py
# ---------------------------------------------------------------------------

def _synthetic_close_prices() -> pd.DataFrame:
    """Deterministic (no RNG) synthetic prices: A has best Sharpe, C has worst."""
    n = 252
    dates = pd.date_range("2020-01-01", periods=n)
    pattern = np.array(([0.001, -0.001, 0.002, -0.002, 0.0005, -0.0015] * (n // 6 + 1))[:n])

    returns_a = 0.0008 + 0.2 * pattern    # high drift, low vol
    returns_b = 0.0003 + 1.0 * pattern    # medium drift, medium vol
    returns_c = 0.0001 + 2.0 * pattern    # low drift, high vol

    return pd.DataFrame({
        "A": 100 * (1 + returns_a).cumprod(),
        "B": 100 * (1 + returns_b).cumprod(),
        "C": 100 * (1 + returns_c).cumprod(),
    }, index=dates)


def test_mean_variance_weights_sum_to_one_within_bounds():
    close = _synthetic_close_prices()
    result = optimizer.mean_variance(close, bounds=(0.0, 0.6))
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)
    for weight in result.weights.values():
        assert -1e-6 <= weight <= 0.6 + 1e-6


def test_mean_variance_favors_best_sharpe_asset():
    close = _synthetic_close_prices()
    result = optimizer.mean_variance(close, bounds=(0.0, 0.6))
    assert result.weights["A"] >= result.weights["B"]
    assert result.weights["A"] >= result.weights["C"]


def test_mean_variance_min_volatility_objective_runs():
    close = _synthetic_close_prices()
    result = optimizer.mean_variance(close, bounds=(0.0, 0.6), objective="min_volatility")
    assert result.method == "min_volatility"
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)


def test_mean_variance_unknown_objective_raises():
    close = _synthetic_close_prices()
    with pytest.raises(ValueError):
        optimizer.mean_variance(close, objective="bogus")


def test_mean_variance_single_asset_infeasible_bounds_fails_loudly():
    """A single asset can't satisfy a < 1.0 upper bound — must raise, not silently misbehave."""
    close = _synthetic_close_prices()[["A"]]
    with pytest.raises(Exception):
        optimizer.mean_variance(close, bounds=(0.0, 0.6))


def test_inverse_volatility_favors_low_vol_asset():
    close = _synthetic_close_prices()
    result = optimizer.inverse_volatility(close)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    # A has the lowest vol multiplier, C the highest -> inverse-vol weights A highest
    assert result.weights["A"] > result.weights["B"] > result.weights["C"]
