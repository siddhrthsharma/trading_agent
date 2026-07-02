"""Tool function tests — deterministic given stored data; no live network calls."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


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


def _save_macro_fixture(series_id: str, dates: list[str], values: list[float]) -> None:
    from data import store
    df = pd.DataFrame({
        "series_id": [series_id] * len(dates),
        "date": pd.to_datetime(dates, utc=True),
        "value": values,
        "fetched_at": [pd.Timestamp("2026-01-01", tz="UTC")] * len(dates),
    })
    store.save_macro(df, series_id)


def _save_fundamentals_fixture(ticker: str, as_of_dates: list[str], pes: list[float | None]) -> None:
    from data import store
    records = [
        {
            "ticker": ticker,
            "as_of": pd.Timestamp(d, tz="UTC"),
            "pe": pe,
            "dividend_yield": 0.02,
            "fetched_at": pd.Timestamp(d, tz="UTC"),
        }
        for d, pe in zip(as_of_dates, pes)
    ]
    store.save_fundamentals(records)


# ---------------------------------------------------------------------------
# macro_tools
# ---------------------------------------------------------------------------

def test_compute_yield_curve_returns_fixture_spread(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_macro_fixture("DGS10", ["2026-01-01"], [4.2])
    _save_macro_fixture("DGS2", ["2026-01-01"], [4.5])

    from agents.tools.macro_tools import compute_yield_curve
    result = compute_yield_curve()

    assert result["spread"] == pytest.approx(4.2 - 4.5)
    assert result["inverted"] is True


def test_compute_yield_curve_missing_data_returns_error_dict(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    from agents.tools.macro_tools import compute_yield_curve
    result = compute_yield_curve()
    assert "error" in result


def test_fetch_fred_series_rejects_non_allowlisted():
    from agents.tools.macro_tools import fetch_fred_series
    result = fetch_fred_series("SOME_RANDOM_SERIES")
    assert "error" in result
    assert "not in the allowed" in result["error"]


def test_fetch_fred_series_serves_from_local_store(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_macro_fixture("FEDFUNDS", ["2025-01-01", "2026-01-01"], [4.5, 5.0])

    from agents.tools.macro_tools import fetch_fred_series
    result = fetch_fred_series("FEDFUNDS")

    assert result["latest_value"] == 5.0
    assert result["series_id"] == "FEDFUNDS"


def test_compute_real_rate_matches_signals_math(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_macro_fixture("FEDFUNDS", ["2025-01-01", "2026-01-01"], [5.0, 5.0])
    _save_macro_fixture("CPIAUCSL", ["2025-01-01", "2026-01-01"], [300.0, 309.0])

    from agents.tools.macro_tools import compute_real_rate
    result = compute_real_rate()

    expected_cpi_yoy = (309.0 - 300.0) / 300.0
    assert result["cpi_yoy"] == pytest.approx(expected_cpi_yoy)
    assert result["real_fed_funds"] == pytest.approx(5.0 - expected_cpi_yoy * 100)


def test_get_macro_snapshot_returns_serializable_dict(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    for series_id, values in {
        "FEDFUNDS": [5.0, 5.0], "DGS10": [4.0, 4.5], "DGS2": [4.0, 4.0],
        "CPIAUCSL": [300.0, 309.0], "UNRATE": [4.0, 4.2],
    }.items():
        _save_macro_fixture(series_id, ["2025-01-01", "2026-01-01"], values)

    from agents.tools.macro_tools import get_macro_snapshot
    result = get_macro_snapshot()

    assert isinstance(result["as_of"], str)  # datetime serialized, not a raw object
    assert result["fed_funds"] == 5.0


# ---------------------------------------------------------------------------
# valuation_tools
# ---------------------------------------------------------------------------

def test_fetch_etf_fundamentals_rejects_disallowed_ticker():
    from agents.tools.valuation_tools import fetch_etf_fundamentals
    result = fetch_etf_fundamentals("NOTATICKER")
    assert "error" in result


def test_fetch_etf_fundamentals_serves_from_local_store(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    _save_fundamentals_fixture("VTI", ["2026-01-01"], [26.0])

    from agents.tools.valuation_tools import fetch_etf_fundamentals
    result = fetch_etf_fundamentals("VTI")

    assert result["pe"] == 26.0
    assert result["expense_ratio"] is not None


def test_compare_historical_pe_single_snapshot_is_honest(tmp_path, monkeypatch):
    """A single snapshot must yield a valid, thin-history comparison — not crash or fabricate a trend."""
    _patch_data_dir(monkeypatch, tmp_path)
    _save_fundamentals_fixture("VTI", ["2026-01-01"], [26.0])

    from agents.tools.valuation_tools import compare_historical_pe
    result = compare_historical_pe("VTI")

    assert result["current_pe"] == 26.0
    assert result["historical_min"] == result["historical_max"] == 26.0
    assert result["n_snapshots"] == 1
    assert "note" in result  # honest about thin history


def test_compare_historical_pe_no_pe_data_reports_note(tmp_path, monkeypatch):
    """Bond ETFs have no P/E — must report a clear note, not an error or a fabricated number."""
    _patch_data_dir(monkeypatch, tmp_path)
    _save_fundamentals_fixture("BND", ["2026-01-01"], [None])

    from agents.tools.valuation_tools import compare_historical_pe
    result = compare_historical_pe("BND")

    assert "note" in result
    assert "current_pe" not in result


def test_compare_historical_pe_missing_ticker_returns_error(tmp_path, monkeypatch):
    _patch_data_dir(monkeypatch, tmp_path)
    from agents.tools.valuation_tools import compare_historical_pe
    result = compare_historical_pe("VTI")
    assert "error" in result


# ---------------------------------------------------------------------------
# portfolio_tools
# ---------------------------------------------------------------------------

def test_drift_from_target_computes_percentage_point_drift():
    from agents.tools.portfolio_tools import drift_from_target

    result = drift_from_target(
        current_holdings={"VTI": 6000, "BND": 4000},
        target={"VTI": 0.5, "BND": 0.5},
    )
    assert result["drift"]["VTI"] == pytest.approx(0.1)
    assert result["drift"]["BND"] == pytest.approx(-0.1)
    assert result["max_drift"] == pytest.approx(0.1)


def test_drift_from_target_zero_holdings_returns_error():
    from agents.tools.portfolio_tools import drift_from_target
    result = drift_from_target(current_holdings={}, target={"VTI": 1.0})
    assert "error" in result


def test_check_allocation_rules_wraps_engine_allocation():
    from agents.tools.portfolio_tools import check_allocation_rules

    result = check_allocation_rules({"QQQ": 0.3, "BND": 0.7})
    assert result["passed"] is False
    assert any(v["rule"] == "qqq_concentration" for v in result["violations"])


def test_check_allocation_rules_clean_allocation_passes():
    from agents.tools.portfolio_tools import check_allocation_rules
    result = check_allocation_rules({"VTI": 0.54, "VXUS": 0.26, "BND": 0.20})
    assert result["passed"] is True
    assert result["violations"] == []


def test_project_growth_wraps_core_contributions():
    from agents.tools.portfolio_tools import project_growth
    result = project_growth(monthly_contribution=500, horizon_years=30, lump_sum=10000)
    assert result["final_balance"] > result["total_contributed"] + 10000  # growth is positive
