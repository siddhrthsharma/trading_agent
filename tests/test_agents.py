"""Agent tests — LLM calls always mocked. Agents never do math; only interpret."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from engine.signals import MacroSnapshot
from data.snapshots import ValuationSnapshot
from profile.investor_profile import InvestorProfile

UTC = timezone.utc


def _profile(**overrides) -> InvestorProfile:
    defaults = dict(
        horizon_years=25,
        risk_tolerance="moderate",
        monthly_contribution=500.0,
        lump_sum=10_000.0,
        monthly_expenses=3000.0,
        emergency_fund=15_000.0,
        accounts=["Roth IRA"],
    )
    defaults.update(overrides)
    return InvestorProfile(**defaults)


def _macro_snapshot(**overrides) -> MacroSnapshot:
    defaults = dict(
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        fed_funds=5.0,
        dgs10=4.2,
        dgs2=4.5,
        yield_curve_spread=-0.3,
        curve_inverted=True,
        cpi_yoy=0.03,
        real_fed_funds=2.0,
        unemployment=4.1,
        unemployment_change_1y=0.3,
    )
    defaults.update(overrides)
    return MacroSnapshot(**defaults)


def _valuation_snapshot(**overrides) -> ValuationSnapshot:
    defaults = dict(
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        funds={
            "VTI": {"pe": 26.0, "dividend_yield": 0.01, "expense_ratio": 0.0003},
            "VXUS": {"pe": 17.7, "dividend_yield": 0.027, "expense_ratio": 0.0007},
            "BND": {"pe": None, "dividend_yield": 0.039, "expense_ratio": 0.0003},
        },
    )
    defaults.update(overrides)
    return ValuationSnapshot(**defaults)


# ---------------------------------------------------------------------------
# macro_agent
# ---------------------------------------------------------------------------

def test_macro_agent_parses_valid_response():
    from agents import macro_agent

    fake_response = {
        "regime": "late-cycle restrictive",
        "rate_environment": "elevated",
        "recession_signal": 0.4,
        "implications": {"equities": "cautious", "bonds": "attractive yields", "cash": "still earning real return"},
        "confidence": 0.7,
        "reasoning": "Curve is inverted but unemployment hasn't turned yet.",
        "suggested_route": "normal",
    }
    with patch("agents.macro_agent.call_llm_json", return_value=fake_response):
        result = macro_agent.run_static(_macro_snapshot(), _profile())

    assert result.regime == "late-cycle restrictive"
    assert result.recession_signal == 0.4
    assert result.suggested_route == "normal"
    assert 0.0 <= result.confidence <= 1.0


def test_macro_agent_rejects_out_of_range_recession_signal():
    from agents import macro_agent

    bad_response = {
        "regime": "x", "rate_environment": "elevated", "recession_signal": 1.5,
        "implications": {}, "confidence": 0.5, "reasoning": "x", "suggested_route": "normal",
    }
    with patch("agents.macro_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            macro_agent.run_static(_macro_snapshot(), _profile())


def test_macro_agent_rejects_unknown_suggested_route():
    from agents import macro_agent

    bad_response = {
        "regime": "x", "rate_environment": "elevated", "recession_signal": 0.5,
        "implications": {}, "confidence": 0.5, "reasoning": "x", "suggested_route": "bullish",
    }
    with patch("agents.macro_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            macro_agent.run_static(_macro_snapshot(), _profile())


def test_macro_agent_rejects_unknown_rate_environment():
    from agents import macro_agent

    bad_response = {
        "regime": "x", "rate_environment": "hot", "recession_signal": 0.5,
        "implications": {}, "confidence": 0.5, "reasoning": "x", "suggested_route": "normal",
    }
    with patch("agents.macro_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            macro_agent.run_static(_macro_snapshot(), _profile())


# ---------------------------------------------------------------------------
# valuation_agent
# ---------------------------------------------------------------------------

def test_valuation_agent_parses_valid_response():
    from agents import valuation_agent

    fake_response = {
        "cheap": ["intl_equity"],
        "rich": ["us_equity"],
        "fair": ["bonds"],
        "notes": {"intl_equity": "lower P/E than US", "us_equity": "elevated P/E", "bonds": "yields near long-run average"},
        "confidence": 0.5,
        "reasoning": "Current snapshot only; no historical comparison available.",
    }
    with patch("agents.valuation_agent.call_llm_json", return_value=fake_response):
        result = valuation_agent.run_static(_valuation_snapshot(), _profile())

    assert result.cheap == ["intl_equity"]
    assert result.rich == ["us_equity"]
    assert 0.0 <= result.confidence <= 1.0


def test_valuation_agent_rejects_non_list_cheap():
    from agents import valuation_agent

    bad_response = {
        "cheap": "intl_equity",  # should be a list
        "rich": [], "fair": [], "notes": {}, "confidence": 0.5, "reasoning": "x",
    }
    with patch("agents.valuation_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            valuation_agent.run_static(_valuation_snapshot(), _profile())


def test_valuation_agent_rejects_bad_confidence():
    from agents import valuation_agent

    bad_response = {
        "cheap": [], "rich": [], "fair": [], "notes": {}, "confidence": 1.5, "reasoning": "x",
    }
    with patch("agents.valuation_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            valuation_agent.run_static(_valuation_snapshot(), _profile())


# ---------------------------------------------------------------------------
# macro_agent — tool-driven path (Phase 6 acceptance: agent calls a tool mid-reasoning)
# ---------------------------------------------------------------------------

def test_macro_agent_tool_driven_calls_fetch_fred_series_and_incorporates_result():
    """Acceptance test: macro_agent.run can call fetch_fred_series("T10Y2Y") mid-reasoning
    and incorporate the tool's result into its final answer — entirely offline (Groq client stubbed)."""
    from types import SimpleNamespace
    from agents import macro_agent

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="fetch_fred_series", arguments='{"series_id": "T10Y2Y"}'),
    )
    round1 = SimpleNamespace(content=None, tool_calls=[tool_call])
    final_json = {
        "regime": "late-cycle restrictive", "rate_environment": "elevated", "recession_signal": 0.6,
        "implications": {"equities": "x", "bonds": "y", "cash": "z"}, "confidence": 0.7,
        "reasoning": "T10Y2Y spread came back negative, indicating an inverted curve.",
        "suggested_route": "normal",
    }
    round2 = SimpleNamespace(content=json.dumps(final_json), tool_calls=None)

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=round1 if len(calls) == 1 else round2)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    tool_executed = {}

    def fake_fetch_fred_series(series_id, lookback_years=5):
        tool_executed["series_id"] = series_id
        return {"series_id": series_id, "latest_value": -0.35, "latest_date": "2026-01-01"}

    from utils.llm import ToolSpec
    from agents.tools.macro_tools import MACRO_TOOLS as REAL_MACRO_TOOLS
    patched_tools = [
        ToolSpec(name=t.name, description=t.description, parameters=t.parameters,
                 fn=fake_fetch_fred_series if t.name == "fetch_fred_series" else t.fn)
        for t in REAL_MACRO_TOOLS
    ]

    with patch("utils.llm._get_client", return_value=fake_client), \
         patch("agents.tools.macro_tools.MACRO_TOOLS", patched_tools):
        result = macro_agent.run(_profile(), as_of="2022-09-30")

    assert tool_executed["series_id"] == "T10Y2Y"
    assert result.regime == "late-cycle restrictive"
    assert len(calls) == 2
    # The tool's result must appear in the messages sent for round 2
    round2_messages = calls[1]["messages"]
    tool_msgs = [m for m in round2_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("-0.35" in m["content"] for m in tool_msgs)


# ---------------------------------------------------------------------------
# allocator_agent
# ---------------------------------------------------------------------------

def _macro_assessment(**overrides):
    from agents.macro_agent import MacroAssessment
    defaults = dict(
        regime="late-cycle restrictive", rate_environment="elevated", recession_signal=0.4,
        implications={"equities": "x", "bonds": "y", "cash": "z"}, confidence=0.7,
        reasoning="test", suggested_route="normal",
    )
    defaults.update(overrides)
    return MacroAssessment(**defaults)


def _valuation_assessment(**overrides):
    from agents.valuation_agent import ValuationAssessment
    defaults = dict(cheap=["intl_equity"], rich=["us_equity"], fair=["bonds"], notes={}, confidence=0.5, reasoning="test")
    defaults.update(overrides)
    return ValuationAssessment(**defaults)


def test_allocator_agent_applies_tilts_via_engine():
    """The final allocation must equal apply_tilts(baseline, clamped) — never LLM-invented."""
    from agents import allocator_agent
    from core.portfolios import default_for
    from engine.allocation import apply_tilts

    profile = _profile()
    baseline = default_for(profile)

    fake_response = {
        "tilts": {"BND": 0.05, "VTI": -0.05},
        "rationale": "Elevated recession signal argues for a modest defensive tilt.",
        "confidence": 0.6,
    }
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response):
        proposal = allocator_agent.run(profile, _macro_assessment(), _valuation_assessment())

    expected = apply_tilts(baseline, {"BND": 0.05, "VTI": -0.05})
    assert proposal.allocation == expected
    assert proposal.baseline == baseline
    assert sum(proposal.allocation.values()) == pytest.approx(1.0)


def test_allocator_agent_clamps_absurd_tilt():
    """A hallucinated tilt far outside MAX_TILT must still produce a legal allocation."""
    from agents import allocator_agent
    from engine.allocation import MAX_TILT

    fake_response = {
        "tilts": {"VTI": 5.0},  # absurd — 500% tilt
        "rationale": "test",
        "confidence": 0.9,
    }
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response):
        proposal = allocator_agent.run(_profile(), _macro_assessment(), _valuation_assessment())

    assert sum(proposal.allocation.values()) == pytest.approx(1.0)
    assert proposal.tilts["VTI"] == pytest.approx(MAX_TILT)
    for ticker, weight in proposal.allocation.items():
        assert 0.0 <= weight <= 1.0


def test_allocator_agent_drops_disallowed_tickers():
    """A tilt on a ticker outside baseline+BIL must be filtered out, not applied."""
    from agents import allocator_agent

    fake_response = {
        "tilts": {"GLD": 0.05},  # not in baseline, not BIL
        "rationale": "test",
        "confidence": 0.5,
    }
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response):
        proposal = allocator_agent.run(_profile(), _macro_assessment(), _valuation_assessment())

    assert "GLD" not in proposal.allocation
    assert proposal.allocation == proposal.baseline


def test_allocator_agent_empty_tilts_returns_baseline():
    from agents import allocator_agent

    fake_response = {"tilts": {}, "rationale": "No tilt warranted.", "confidence": 0.8}
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response):
        proposal = allocator_agent.run(_profile(), _macro_assessment(), _valuation_assessment())

    assert proposal.allocation == proposal.baseline


def test_allocator_agent_accepts_optional_optimizer_hint():
    """optimizer_hint is informational only — it must not change the anchor/bounds mechanism."""
    from agents import allocator_agent
    from engine.optimizer import OptimizationResult

    hint = OptimizationResult(
        weights={"VTI": 0.9, "BND": 0.1}, expected_return=0.12, expected_vol=0.18, sharpe=0.6, method="max_sharpe",
    )
    fake_response = {"tilts": {}, "rationale": "Optimizer suggests more equity, but baseline stands.", "confidence": 0.7}
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response):
        proposal = allocator_agent.run(_profile(), _macro_assessment(), _valuation_assessment(), optimizer_hint=hint)

    assert proposal.allocation == proposal.baseline  # still bounded by the same mechanism


def test_allocator_agent_none_optimizer_hint_is_default():
    from agents import allocator_agent
    import inspect

    assert inspect.signature(allocator_agent.run).parameters["optimizer_hint"].default is None


def test_allocator_agent_none_feedback_is_default():
    from agents import allocator_agent
    import inspect

    assert inspect.signature(allocator_agent.run).parameters["feedback"].default is None


def test_allocator_agent_folds_feedback_into_prompt():
    """Phase 7: revision-loop feedback (critic's violations/issues) must reach the prompt."""
    from agents import allocator_agent, critic_agent
    from engine.allocation import RuleViolation

    feedback = critic_agent.CriticReport(
        passed=False,
        violations=[RuleViolation(rule="concentration", severity="serious", detail="VTI at 80.0% exceeds 75% limit")],
        qualitative_issues=["Tilt not well justified by the stated macro read"],
        severity="serious",
        confidence=0.6,
        reasoning="Too concentrated.",
    )
    fake_response = {"tilts": {}, "rationale": "reverted", "confidence": 0.7}
    with patch("agents.allocator_agent.call_llm_json", return_value=fake_response) as mock_call:
        allocator_agent.run(_profile(), _macro_assessment(), _valuation_assessment(), feedback=feedback)

    prompt = mock_call.call_args.args[0]
    assert "concentration" in prompt
    assert "VTI at 80.0% exceeds 75% limit" in prompt
    assert "Tilt not well justified" in prompt


# ---------------------------------------------------------------------------
# critic_agent
# ---------------------------------------------------------------------------

def test_critic_agent_passes_clean_allocation():
    from agents import allocator_agent, critic_agent
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)
    proposal = allocator_agent.AllocationProposal(
        allocation=baseline, baseline=baseline, tilts={}, rationale="none", confidence=0.8,
    )

    fake_response = {"qualitative_issues": [], "confidence": 0.8, "reasoning": "Looks fine."}
    with patch("agents.critic_agent.call_llm_json", return_value=fake_response):
        report = critic_agent.run(proposal, _macro_assessment(), _valuation_assessment(), profile)

    assert report.passed is True
    assert report.severity == "none"
    assert report.violations == []


def test_critic_agent_flags_serious_violation():
    from agents.allocator_agent import AllocationProposal

    bad_allocation = {"QQQ": 0.30, "BND": 0.70}
    proposal = AllocationProposal(
        allocation=bad_allocation, baseline=bad_allocation, tilts={}, rationale="none", confidence=0.5,
    )

    from agents import critic_agent
    fake_response = {"qualitative_issues": ["Heavy QQQ concentration"], "confidence": 0.9, "reasoning": "Too concentrated."}
    with patch("agents.critic_agent.call_llm_json", return_value=fake_response):
        report = critic_agent.run(proposal, _macro_assessment(), _valuation_assessment(), _profile())

    assert report.passed is False
    assert report.severity == "serious"
    assert any(v.rule == "qqq_concentration" for v in report.violations)


def test_critic_agent_rejects_non_list_issues():
    from agents.allocator_agent import AllocationProposal
    from agents import critic_agent

    proposal = AllocationProposal(
        allocation={"VTI": 1.0}, baseline={"VTI": 1.0}, tilts={}, rationale="none", confidence=0.5,
    )
    bad_response = {"qualitative_issues": "not a list", "confidence": 0.5, "reasoning": "x"}
    with patch("agents.critic_agent.call_llm_json", return_value=bad_response):
        with pytest.raises(ValueError):
            critic_agent.run(proposal, _macro_assessment(), _valuation_assessment(), _profile())


def test_critic_agent_accepts_none_valuation_on_crisis_path():
    """Phase 9: the crisis path skips the valuation agent entirely; critic must still run."""
    from agents.allocator_agent import AllocationProposal
    from agents import critic_agent

    profile = _profile()
    proposal = AllocationProposal(
        allocation={"VTI": 0.45, "VXUS": 0.20, "BND": 0.35}, baseline={"VTI": 0.45, "VXUS": 0.20, "BND": 0.35},
        tilts={}, rationale="defensive: none needed", confidence=0.7,
    )
    fake_response = {"qualitative_issues": [], "confidence": 0.7, "reasoning": "Looks fine."}
    with patch("agents.critic_agent.call_llm_json", return_value=fake_response) as mock_call:
        report = critic_agent.run(proposal, _macro_assessment(suggested_route="crisis"), None, profile)

    assert report.passed is True
    prompt = mock_call.call_args.args[0]
    assert "skipped on the crisis path" in prompt


# ---------------------------------------------------------------------------
# crisis_agent
# ---------------------------------------------------------------------------

def test_crisis_agent_applies_tilts_via_engine():
    """Same anti-invention mechanism as the allocator: the LLM never outputs a percentage."""
    from agents import crisis_agent
    from core.portfolios import default_for
    from engine.allocation import apply_tilts

    profile = _profile()
    baseline = default_for(profile)

    fake_response = {
        "tilts": {"BND": 0.05, "VTI": -0.05},
        "rationale": "Elevated recession signal argues for a defensive tilt.",
        "confidence": 0.6,
    }
    with patch("agents.crisis_agent.call_llm_json", return_value=fake_response):
        proposal = crisis_agent.run(profile, _macro_assessment(recession_signal=0.9, suggested_route="crisis"))

    expected = apply_tilts(baseline, {"BND": 0.05, "VTI": -0.05})
    assert proposal.allocation == expected
    assert proposal.baseline == baseline


def test_crisis_agent_clamps_absurd_tilt():
    from agents import crisis_agent
    from engine.allocation import MAX_TILT

    fake_response = {"tilts": {"BND": 5.0}, "rationale": "test", "confidence": 0.9}
    with patch("agents.crisis_agent.call_llm_json", return_value=fake_response):
        proposal = crisis_agent.run(_profile(), _macro_assessment(recession_signal=0.9, suggested_route="crisis"))

    assert sum(proposal.allocation.values()) == pytest.approx(1.0)
    assert proposal.tilts["BND"] == pytest.approx(MAX_TILT)


def test_crisis_agent_drops_disallowed_tickers():
    from agents import crisis_agent

    fake_response = {"tilts": {"GLD": 0.05}, "rationale": "test", "confidence": 0.5}
    with patch("agents.crisis_agent.call_llm_json", return_value=fake_response):
        proposal = crisis_agent.run(_profile(), _macro_assessment(recession_signal=0.9, suggested_route="crisis"))

    assert "GLD" not in proposal.allocation
    assert proposal.allocation == proposal.baseline


def test_crisis_agent_valuation_defaults_to_none():
    from agents import crisis_agent
    import inspect

    assert inspect.signature(crisis_agent.run).parameters["valuation"].default is None


def test_crisis_agent_works_without_valuation():
    from agents import crisis_agent

    fake_response = {"tilts": {}, "rationale": "no tilt warranted", "confidence": 0.8}
    with patch("agents.crisis_agent.call_llm_json", return_value=fake_response) as mock_call:
        proposal = crisis_agent.run(_profile(), _macro_assessment(recession_signal=0.9, suggested_route="crisis"))

    assert proposal.allocation == proposal.baseline
    prompt = mock_call.call_args.args[0]
    assert "Valuation read" not in prompt
