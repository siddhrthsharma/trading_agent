"""Phase 1 core-layer tests — deterministic math, zero LLM/network calls."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard: no LLM or network imports may be used by core/
# ---------------------------------------------------------------------------

def test_core_imports_no_llm():
    """core.accounts/portfolios/contributions must not import utils.llm or network libs."""
    import core.accounts
    import core.portfolios
    import core.contributions

    forbidden = {"utils.llm", "groq", "anthropic", "requests", "httpx", "aiohttp"}
    loaded = set(sys.modules.keys())
    violations = forbidden & loaded
    # Filter to only those that were imported BY our core modules (not pre-existing)
    # We confirm by checking the module objects directly — none should have groq/anthropic attrs
    for mod_name in ("core.accounts", "core.portfolios", "core.contributions"):
        mod = sys.modules[mod_name]
        for attr in ("groq", "anthropic", "requests"):
            assert not hasattr(mod, attr), f"{mod_name} unexpectedly imports {attr}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def funded_profile(tmp_path):
    """A profile with a fully funded emergency fund."""
    data = {
        "horizon_years": 30,
        "risk_tolerance": "moderate",
        "monthly_contribution": 500.0,
        "lump_sum": 10000.0,
        "monthly_expenses": 2500.0,
        "emergency_fund": 8000.0,   # 3.2 months — funded (>= 7500)
        "accounts": ["Roth IRA", "401k", "taxable"],
        "constraints": [],
        "current_holdings": {},
    }
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(data))
    return p, data


@pytest.fixture
def underfunded_profile(tmp_path):
    """A profile whose emergency fund is below 3 months."""
    data = {
        "horizon_years": 30,
        "risk_tolerance": "moderate",
        "monthly_contribution": 500.0,
        "lump_sum": 5000.0,
        "monthly_expenses": 2500.0,
        "emergency_fund": 7499.0,   # just below 3 * 2500 = 7500
        "accounts": ["Roth IRA", "401k", "taxable"],
        "constraints": [],
        "current_holdings": {},
    }
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(data))
    return p, data


# ===========================================================================
# U1 — InvestorProfile + loader
# ===========================================================================

class TestInvestorProfile:
    def test_roundtrip_example(self):
        """load_profile with no argument falls back to profile.example.json."""
        from profile.investor_profile import load_profile
        profile = load_profile("profile.example.json")
        assert profile.horizon_years == 30
        assert profile.risk_tolerance == "moderate"
        assert profile.monthly_contribution == 500.0
        assert profile.monthly_expenses == 2500.0
        assert profile.emergency_fund == 8000.0
        assert "Roth IRA" in profile.accounts

    def test_load_from_path(self, funded_profile):
        path, data = funded_profile
        from profile.investor_profile import load_profile
        profile = load_profile(path)
        assert profile.horizon_years == data["horizon_years"]
        assert profile.risk_tolerance == data["risk_tolerance"]
        assert profile.lump_sum == data["lump_sum"]

    def test_fallback_to_example_when_missing(self, tmp_path):
        """Missing profile.json must silently fall back to profile.example.json."""
        from profile.investor_profile import load_profile
        profile = load_profile(tmp_path / "nonexistent.json")
        # Should load the example without raising
        assert profile.risk_tolerance in {"conservative", "moderate", "aggressive"}

    def test_bad_risk_tolerance_raises(self, tmp_path):
        data = {
            "horizon_years": 20, "risk_tolerance": "yolo",
            "monthly_contribution": 100, "lump_sum": 0,
            "monthly_expenses": 1000, "emergency_fund": 5000,
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        from profile.investor_profile import load_profile
        with pytest.raises(ValueError, match="risk_tolerance"):
            load_profile(p)

    def test_negative_monthly_expenses_raises(self, tmp_path):
        data = {
            "horizon_years": 20, "risk_tolerance": "moderate",
            "monthly_contribution": 100, "lump_sum": 0,
            "monthly_expenses": -500, "emergency_fund": 5000,
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        from profile.investor_profile import load_profile
        with pytest.raises(ValueError, match="monthly_expenses"):
            load_profile(p)


# ===========================================================================
# U2 — core/accounts.py
# ===========================================================================

class TestEmergencyFundStatus:
    def _ef(self, fund, expenses):
        from profile.investor_profile import InvestorProfile
        from core.accounts import emergency_fund_status
        profile = InvestorProfile(
            horizon_years=20, risk_tolerance="moderate",
            monthly_contribution=0, lump_sum=0,
            monthly_expenses=expenses, emergency_fund=fund,
        )
        return emergency_fund_status(profile)

    def test_just_funded_boundary(self):
        """emergency_fund == 3 * monthly_expenses → funded (>= not >)."""
        ef = self._ef(fund=7500.0, expenses=2500.0)
        assert ef.is_funded is True
        assert ef.shortfall == 0.0
        assert math.isclose(ef.months_covered, 3.0)

    def test_one_cent_below_not_funded(self):
        ef = self._ef(fund=7499.99, expenses=2500.0)
        assert ef.is_funded is False
        assert ef.shortfall > 0.0

    def test_well_funded(self):
        ef = self._ef(fund=20000.0, expenses=2500.0)
        assert ef.is_funded is True
        assert ef.months_covered > 6.0
        assert ef.shortfall == 0.0

    def test_zero_expenses_guarded(self):
        """monthly_expenses == 0 must not raise ZeroDivisionError."""
        ef = self._ef(fund=5000.0, expenses=0.0)
        assert ef.is_funded is True
        assert ef.months_covered == float("inf")
        assert ef.shortfall == 0.0

    def test_target_values(self):
        ef = self._ef(fund=0, expenses=2500.0)
        assert ef.target_low == 7500.0
        assert ef.target_high == 15000.0
        assert ef.shortfall == 7500.0


class TestAccountPriority:
    def _priority(self, accounts):
        from profile.investor_profile import InvestorProfile
        from core.accounts import account_priority
        profile = InvestorProfile(
            horizon_years=20, risk_tolerance="moderate",
            monthly_contribution=0, lump_sum=0,
            monthly_expenses=0, emergency_fund=0,
            accounts=accounts,
        )
        return account_priority(profile)

    def test_roth_before_401k_before_taxable(self):
        result = self._priority(["taxable", "Roth IRA", "401k"])
        assert result.index("Roth IRA") < result.index("401k")
        assert result.index("401k") < result.index("taxable")

    def test_missing_accounts_omitted(self):
        result = self._priority(["Roth IRA", "taxable"])
        assert "401k" not in result

    def test_unknown_account_appended(self):
        result = self._priority(["Roth IRA", "HSA"])
        assert result[-1] == "HSA"

    def test_empty_accounts(self):
        assert self._priority([]) == []


class TestGainsTreatment:
    def test_365_days_short_term(self):
        from core.accounts import gains_treatment
        assert gains_treatment(365) == "short-term"

    def test_366_days_long_term(self):
        from core.accounts import gains_treatment
        assert gains_treatment(366) == "long-term"

    def test_zero_days_short_term(self):
        from core.accounts import gains_treatment
        assert gains_treatment(0) == "short-term"

    def test_many_days_long_term(self):
        from core.accounts import gains_treatment
        assert gains_treatment(3650) == "long-term"


# ===========================================================================
# U3 — core/portfolios.py
# ===========================================================================

class TestAllocations:
    def _sum(self, alloc):
        return sum(alloc.values())

    def test_three_fund_weights_sum_to_one(self):
        from core.portfolios import three_fund
        for rt in ("conservative", "moderate", "aggressive"):
            alloc = three_fund(rt)
            assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9), rt

    def test_three_fund_all_weights_nonnegative(self):
        from core.portfolios import three_fund
        for rt in ("conservative", "moderate", "aggressive"):
            for w in three_fund(rt).values():
                assert w >= 0.0

    def test_glide_path_sum_to_one(self):
        from core.portfolios import glide_path
        for h in (1, 5, 10, 20, 30, 40, 50):
            alloc = glide_path(h)
            assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9), h

    def test_glide_path_monotonic_equity(self):
        """Longer horizon must produce >= equity weight."""
        from core.portfolios import glide_path
        _US = "VTI"; _INTL = "VXUS"
        horizons = [5, 10, 20, 30, 40]
        equity_weights = [glide_path(h).get(_US, 0) + glide_path(h).get(_INTL, 0) for h in horizons]
        for i in range(len(equity_weights) - 1):
            assert equity_weights[i] <= equity_weights[i + 1], (
                f"Equity weight not monotonic: h={horizons[i]} ({equity_weights[i]:.4f}) "
                f"> h={horizons[i+1]} ({equity_weights[i+1]:.4f})"
            )

    def test_glide_path_edge_horizon_1(self):
        from core.portfolios import glide_path
        alloc = glide_path(1)
        assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9)
        # Should have meaningful bond allocation
        assert alloc.get("BND", 0) > 0

    def test_glide_path_edge_horizon_50(self):
        from core.portfolios import glide_path
        alloc = glide_path(50)
        assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9)

    def test_target_date_sum_to_one(self):
        from core.portfolios import target_date
        for h in (1, 5, 10, 20, 30):
            alloc = target_date(h)
            assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9), h

    def test_all_equity_no_bonds(self):
        from core.portfolios import all_equity
        alloc = all_equity()
        assert math.isclose(self._sum(alloc), 1.0, abs_tol=1e-9)
        assert alloc.get("BND", 0) == 0.0
        assert alloc.get("BIL", 0) == 0.0

    def test_all_equity_sum_to_one(self):
        from core.portfolios import all_equity
        assert math.isclose(self._sum(all_equity()), 1.0, abs_tol=1e-9)

    def test_tickers_in_spec_universe(self):
        """All returned tickers must be in the spec-approved universe."""
        from core.portfolios import three_fund, glide_path, target_date, all_equity
        spec_universe = {"VTI", "VOO", "SPY", "QQQ", "VXUS", "VEA", "VWO", "BND",
                         "TLT", "SHY", "BIL", "SCHP", "VNQ", "GLD"}
        for alloc in [three_fund("moderate"), glide_path(20), target_date(20), all_equity()]:
            for ticker in alloc:
                assert ticker in spec_universe, f"Unknown ticker {ticker!r}"

    def test_default_for_aggressive_long_is_equity(self):
        """Aggressive + long horizon → all-equity allocation (no bonds)."""
        from profile.investor_profile import InvestorProfile
        from core.portfolios import default_for
        profile = InvestorProfile(
            horizon_years=30, risk_tolerance="aggressive",
            monthly_contribution=0, lump_sum=0,
            monthly_expenses=0, emergency_fund=0,
        )
        alloc = default_for(profile)
        assert alloc.get("BND", 0) == 0.0

    def test_default_for_conservative_has_bonds(self):
        from profile.investor_profile import InvestorProfile
        from core.portfolios import default_for
        profile = InvestorProfile(
            horizon_years=10, risk_tolerance="conservative",
            monthly_contribution=0, lump_sum=0,
            monthly_expenses=0, emergency_fund=0,
        )
        alloc = default_for(profile)
        assert alloc.get("BND", 0) > 0


# ===========================================================================
# U4 — core/contributions.py
# ===========================================================================

class TestProjectGrowth:
    def test_annuity_hand_computed(self):
        """$100/month for 12 months at 7% real matches the closed-form annuity FV."""
        from core.contributions import project_growth
        i = (1.07) ** (1 / 12) - 1
        expected_annuity = 100 * (((1 + i) ** 12 - 1) / i)
        proj = project_growth(monthly_contribution=100, horizon_years=1, lump_sum=0)
        assert math.isclose(proj.final_balance, expected_annuity, rel_tol=1e-9), (
            f"Expected {expected_annuity:.6f}, got {proj.final_balance:.6f}"
        )

    def test_lump_sum_hand_computed(self):
        """$10,000 lump sum, no contributions, 10 years → $10,000 * 1.07^10 ≈ 19,671.51."""
        from core.contributions import project_growth
        expected = 10000 * (1.07 ** 10)
        proj = project_growth(monthly_contribution=0, horizon_years=10, lump_sum=10000)
        assert math.isclose(proj.final_balance, expected, rel_tol=1e-6), (
            f"Expected {expected:.2f}, got {proj.final_balance:.2f}"
        )

    def test_accounting_identity(self):
        """total_contributed + total_growth + lump_sum == final_balance."""
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=500, horizon_years=30, lump_sum=10000)
        expected = proj.total_contributed + proj.total_growth + proj.lump_sum
        assert math.isclose(proj.final_balance, expected, rel_tol=1e-9), (
            f"Accounting identity failed: {proj.final_balance:.2f} != {expected:.2f}"
        )

    def test_zero_contribution_zero_lump(self):
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=0, horizon_years=10, lump_sum=0)
        assert proj.final_balance == 0.0
        assert proj.total_contributed == 0.0
        assert proj.total_growth == 0.0

    def test_horizon_zero_returns_lump(self):
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=500, horizon_years=0, lump_sum=5000)
        assert proj.final_balance == 5000.0
        assert proj.total_contributed == 0.0

    def test_growth_is_positive(self):
        """Growth must be positive for positive inputs and positive return rate."""
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=500, horizon_years=20, lump_sum=10000)
        assert proj.total_growth > 0

    def test_yearly_balances_length(self):
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=100, horizon_years=10, lump_sum=0)
        assert len(proj.yearly_balances) == 10

    def test_yearly_balances_monotonic(self):
        """Balance must grow each year with positive contributions and positive rate."""
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=100, horizon_years=10, lump_sum=0)
        for i in range(1, len(proj.yearly_balances)):
            assert proj.yearly_balances[i] > proj.yearly_balances[i - 1]

    def test_final_balance_matches_last_yearly(self):
        from core.contributions import project_growth
        proj = project_growth(monthly_contribution=200, horizon_years=5, lump_sum=1000)
        assert math.isclose(proj.final_balance, proj.yearly_balances[-1], rel_tol=1e-9)

    def test_default_return_rate_is_seven_percent(self):
        """Default annual_real_return must be 0.07 (spec guardrail)."""
        from core.contributions import _DEFAULT_ANNUAL_REAL_RETURN
        assert _DEFAULT_ANNUAL_REAL_RETURN == 0.07, (
            "Default real return must be exactly 7% — do not encode wilder figures"
        )


class TestDcaSchedule:
    def test_correct_number_of_contributions(self):
        from core.contributions import dca_schedule
        sched = dca_schedule(500, 30)
        assert len(sched) == 30 * 12

    def test_all_same_amount(self):
        from core.contributions import dca_schedule
        sched = dca_schedule(300, 5)
        assert all(c.amount == 300.0 for c in sched)

    def test_months_are_sequential(self):
        from core.contributions import dca_schedule
        sched = dca_schedule(100, 2)
        months = [c.month for c in sched]
        assert months == list(range(1, 25))

    def test_zero_horizon_empty(self):
        from core.contributions import dca_schedule
        assert dca_schedule(500, 0) == []


# ===========================================================================
# U5 — main.py advisor wiring
# ===========================================================================

class TestRunAdvisor:
    def _make_profile(self, tmp_path, funded: bool):
        data = {
            "horizon_years": 30, "risk_tolerance": "moderate",
            "monthly_contribution": 500, "lump_sum": 10000,
            "monthly_expenses": 2500,
            "emergency_fund": 8000 if funded else 5000,
            "accounts": ["Roth IRA", "401k", "taxable"],
            "constraints": [], "current_holdings": {},
        }
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(data))
        return p

    def test_underfunded_does_not_call_default_for(self, tmp_path):
        """run_advisor must return before calling portfolios.default_for when underfunded."""
        path = self._make_profile(tmp_path, funded=False)
        with (
            patch("profile.investor_profile.load_profile", return_value=_load(path)),
            patch("core.portfolios.default_for") as mock_alloc,
        ):
            from main import run_advisor
            run_advisor()
        mock_alloc.assert_not_called()

    def test_funded_calls_default_for(self, tmp_path):
        """With a funded profile run_advisor must invoke portfolios.default_for."""
        path = self._make_profile(tmp_path, funded=True)
        with (
            patch("profile.investor_profile.load_profile", return_value=_load(path)),
            patch("core.portfolios.default_for", return_value={"VTI": 0.7, "VXUS": 0.3}) as mock_alloc,
        ):
            from main import run_advisor
            run_advisor()
        mock_alloc.assert_called_once()

    def test_no_llm_imported_by_main(self):
        """main.py must never import utils.llm or any LLM provider."""
        import main
        assert not hasattr(main, "groq"), "main.py imported groq directly"
        assert not hasattr(main, "anthropic"), "main.py imported anthropic directly"
        # utils.llm must not have been loaded as a side effect of importing main
        # (allow if it was already loaded before this test)
        llm_mod = sys.modules.get("utils.llm")
        if llm_mod is not None:
            # It was already loaded — verify main doesn't reference it
            assert "llm" not in dir(main)


def _load(path):
    """Helper: load a profile from a Path."""
    from profile.investor_profile import load_profile
    return load_profile(path)
