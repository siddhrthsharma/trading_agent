"""LangGraph advisor-graph tests — LLM always mocked; asserts real revision/routing behavior."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from profile.investor_profile import InvestorProfile


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


def _supervisor_plan(**overrides):
    from agents.supervisor import SupervisorPlan
    defaults = dict(
        goal="Allocate long-term investments for this profile",
        steps=["check emergency fund", "read macro regime", "propose allocation", "critique"],
        rationale="test plan", confidence=0.7,
    )
    defaults.update(overrides)
    return SupervisorPlan(**defaults)


def _build_initial_state(profile, baseline, max_iterations=3):
    from agents.graph.state import initial_state
    return initial_state(profile, baseline, max_iterations=max_iterations)


# ---------------------------------------------------------------------------
# Phase 7: real-code-path — aggressive profile all_equity baseline, tilt breaches position cap
# ---------------------------------------------------------------------------

def test_graph_revision_loop_converges():
    """A tilt that pushes VTI over the 75% position cap must trigger exactly one revision."""
    from agents.graph.build import build_graph
    from core.portfolios import all_equity

    profile = _profile(risk_tolerance="aggressive", horizon_years=30)
    baseline = all_equity()  # VTI 70% / VXUS 30%
    assert baseline["VTI"] == pytest.approx(0.70)

    bad_tilt = {"tilts": {"VTI": 0.10, "VXUS": -0.10}, "rationale": "aggressive tilt", "confidence": 0.6}
    good_tilt = {"tilts": {}, "rationale": "reverted to baseline", "confidence": 0.7}
    critic_response = {"qualitative_issues": [], "confidence": 0.6, "reasoning": "ok"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(suggested_route="normal", recession_signal=0.4)), \
         patch("agents.valuation_agent.run", return_value=_valuation_assessment()), \
         patch("agents.allocator_agent.call_llm_json", side_effect=[bad_tilt, good_tilt]) as mock_alloc, \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["route"] == "normal"
    assert final["critic_report"].passed is True
    assert final["iteration"] == 2
    assert len(final["history"]) == 2
    assert final["history"][0]["passed"] is False
    assert any("concentration" in v for v in final["history"][0]["violations"])

    # second allocator call's prompt must include the concentration feedback
    second_call_prompt = mock_alloc.call_args_list[1].args[0]
    assert "concentration" in second_call_prompt.lower()


# ---------------------------------------------------------------------------
# Phase 7: spec-literal — 80% QQQ / 20% BND hand-built proposal
# ---------------------------------------------------------------------------

def test_graph_loop_on_literal_qqq_proposal():
    from agents.allocator_agent import AllocationProposal
    from agents.graph.build import build_graph
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)

    qqq_proposal = AllocationProposal(
        allocation={"QQQ": 0.8, "BND": 0.2}, baseline=baseline, tilts={}, rationale="bad", confidence=0.9,
    )
    clean_proposal = AllocationProposal(
        allocation=baseline, baseline=baseline, tilts={}, rationale="reverted", confidence=0.8,
    )
    critic_response = {"qualitative_issues": [], "confidence": 0.6, "reasoning": "ok"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(suggested_route="normal", recession_signal=0.4)), \
         patch("agents.valuation_agent.run", return_value=_valuation_assessment()), \
         patch("agents.allocator_agent.run", side_effect=[qqq_proposal, clean_proposal]), \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["iteration"] == 2
    assert final["critic_report"].passed is True
    assert final["history"][0]["passed"] is False
    assert any("qqq_concentration" in v for v in final["history"][0]["violations"])


# ---------------------------------------------------------------------------
# Phase 7: max_iterations halts cleanly
# ---------------------------------------------------------------------------

def test_graph_halts_at_max_iterations():
    from agents.graph.build import build_graph
    from core.portfolios import all_equity

    profile = _profile(risk_tolerance="aggressive", horizon_years=30)
    baseline = all_equity()

    bad_tilt = {"tilts": {"VTI": 0.10, "VXUS": -0.10}, "rationale": "always bad", "confidence": 0.6}
    critic_response = {"qualitative_issues": [], "confidence": 0.6, "reasoning": "still bad"}

    state = _build_initial_state(profile, baseline, max_iterations=3)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(suggested_route="normal", recession_signal=0.4)), \
         patch("agents.valuation_agent.run", return_value=_valuation_assessment()), \
         patch("agents.allocator_agent.call_llm_json", return_value=bad_tilt), \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["iteration"] == 3
    assert final["critic_report"].passed is False
    assert len(final["history"]) == 3


# ---------------------------------------------------------------------------
# Phase 7: clean first try, no loop
# ---------------------------------------------------------------------------

def test_graph_passes_first_try_no_loop():
    from agents.graph.build import build_graph
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)

    good_tilt = {"tilts": {}, "rationale": "no tilt warranted", "confidence": 0.8}
    critic_response = {"qualitative_issues": [], "confidence": 0.8, "reasoning": "looks fine"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(suggested_route="normal", recession_signal=0.4)), \
         patch("agents.valuation_agent.run", return_value=_valuation_assessment()), \
         patch("agents.allocator_agent.call_llm_json", return_value=good_tilt), \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["iteration"] == 1
    assert len(final["history"]) == 1
    assert final["critic_report"].passed is True


# ---------------------------------------------------------------------------
# Phase 9: dynamic routing
# ---------------------------------------------------------------------------

def test_graph_routes_to_crisis_on_high_recession_signal():
    """A high recession signal must route to the crisis node and skip valuation entirely."""
    from agents.graph.build import build_graph
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)

    crisis_tilt = {"tilts": {"BND": 0.05, "BIL": 0.05, "VTI": -0.10}, "rationale": "defensive", "confidence": 0.7}
    critic_response = {"qualitative_issues": [], "confidence": 0.7, "reasoning": "ok"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(recession_signal=0.9, suggested_route="crisis")), \
         patch("agents.valuation_agent.run") as mock_valuation, \
         patch("agents.crisis_agent.call_llm_json", return_value=crisis_tilt), \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["route"] == "crisis"
    assert final["valuation"] is None
    mock_valuation.assert_not_called()
    assert final["proposal"].allocation["BND"] > baseline["BND"]
    assert final["critic_report"].passed is True


def test_graph_routes_to_normal_on_low_signal():
    """A low recession signal must route to the normal valuation -> allocator path."""
    from agents.graph.build import build_graph
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)

    good_tilt = {"tilts": {}, "rationale": "no tilt warranted", "confidence": 0.8}
    critic_response = {"qualitative_issues": [], "confidence": 0.8, "reasoning": "looks fine"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(recession_signal=0.3, suggested_route="normal")), \
         patch("agents.valuation_agent.run", return_value=_valuation_assessment()) as mock_valuation, \
         patch("agents.allocator_agent.call_llm_json", return_value=good_tilt), \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["route"] == "normal"
    mock_valuation.assert_called_once()
    assert final["valuation"] is not None
    assert final["proposal"].allocation == final["proposal"].baseline


def test_crisis_branch_still_loops_through_critic():
    """A rule-violating crisis proposal must revise via the crisis node, not the normal allocator."""
    from agents.graph.build import build_graph
    from core.portfolios import all_equity

    profile = _profile(risk_tolerance="aggressive", horizon_years=30)
    baseline = all_equity()  # VTI 70% / VXUS 30% — a +10 VTI tilt breaches the 75% position cap

    bad_tilt = {"tilts": {"VTI": 0.10, "VXUS": -0.10}, "rationale": "too aggressive a defensive lean", "confidence": 0.6}
    good_tilt = {"tilts": {}, "rationale": "reverted", "confidence": 0.7}
    critic_response = {"qualitative_issues": [], "confidence": 0.6, "reasoning": "ok"}

    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", return_value=_macro_assessment(recession_signal=0.9, suggested_route="crisis")), \
         patch("agents.valuation_agent.run") as mock_valuation, \
         patch("agents.crisis_agent.call_llm_json", side_effect=[bad_tilt, good_tilt]) as mock_crisis, \
         patch("agents.allocator_agent.run") as mock_allocator, \
         patch("agents.critic_agent.call_llm_json", return_value=critic_response):
        final = build_graph().invoke(state)

    assert final["route"] == "crisis"
    mock_valuation.assert_not_called()
    mock_allocator.assert_not_called()
    assert mock_crisis.call_count == 2
    assert final["iteration"] == 2
    assert final["critic_report"].passed is True


# ---------------------------------------------------------------------------
# Phase 9: unit tests for supervisor
# ---------------------------------------------------------------------------

def test_decide_route_threshold():
    from agents.supervisor import decide_route

    assert decide_route(_macro_assessment(recession_signal=0.8, suggested_route="normal")) == "normal"
    assert decide_route(_macro_assessment(recession_signal=0.81, suggested_route="normal")) == "crisis"
    assert decide_route(_macro_assessment(recession_signal=0.2, suggested_route="crisis")) == "crisis"


def test_supervisor_plans_for_bonus_goal():
    from agents import supervisor

    fake_response = {
        "steps": ["check emergency fund", "read macro regime", "propose allocation", "critique"],
        "rationale": "Standard pipeline applies to any lump-sum allocation goal.",
        "confidence": 0.75,
    }
    with patch("agents.supervisor.call_llm_json", return_value=fake_response) as mock_call:
        plan = supervisor.run("where do I put my bonus?", _profile())

    assert plan.goal == "where do I put my bonus?"
    assert plan.steps == fake_response["steps"]
    assert 0.0 <= plan.confidence <= 1.0
    prompt = mock_call.call_args.args[0]
    assert "where do I put my bonus?" in prompt


# ---------------------------------------------------------------------------
# Phase 9: AgentDataUnavailable when both macro paths fail
# ---------------------------------------------------------------------------

def test_macro_node_raises_agent_data_unavailable_when_both_paths_fail():
    from agents.graph.build import build_graph
    from agents.graph.nodes import AgentDataUnavailable
    from core.portfolios import default_for

    profile = _profile()
    baseline = default_for(profile)
    state = _build_initial_state(profile, baseline)

    with patch("agents.supervisor.run", return_value=_supervisor_plan()), \
         patch("agents.macro_agent.run", side_effect=RuntimeError("no tools available")), \
         patch("agents.macro_agent.run_static", side_effect=ValueError("no local macro data")):
        with pytest.raises(AgentDataUnavailable):
            build_graph().invoke(state)
