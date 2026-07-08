"""Shared LangGraph state — nodes read and write this; all math stays in core/ and engine/."""
from __future__ import annotations

from typing import TypedDict

from agents.allocator_agent import AllocationProposal
from agents.critic_agent import CriticReport
from agents.macro_agent import MacroAssessment
from agents.valuation_agent import ValuationAssessment
from engine.optimizer import OptimizationResult
from profile.investor_profile import InvestorProfile

DEFAULT_MAX_ITERATIONS = 3


class AdvisorState(TypedDict, total=False):
    """Everything the advisor graph reads/writes; one iteration = allocator -> critic."""
    profile: InvestorProfile
    goal: str                                  # used by supervisor (Phase 9)
    baseline: dict[str, float]                 # core/-computed anchor
    macro: MacroAssessment | None
    valuation: ValuationAssessment | None
    optimizer_hint: OptimizationResult | None
    proposal: AllocationProposal | None
    critic_report: CriticReport | None
    feedback: CriticReport | None              # serious findings fed back to the allocator
    iteration: int                             # incremented by the proposing node
    max_iterations: int
    history: list[dict]                        # one summary dict per iteration
    route: str                                 # "normal" | "crisis" (Phase 9 sets "crisis")
    plan: object | None                        # SupervisorPlan (Phase 9)


def initial_state(
    profile: InvestorProfile,
    baseline: dict[str, float],
    macro: MacroAssessment | None,
    valuation: ValuationAssessment | None,
    optimizer_hint: OptimizationResult | None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    goal: str = "Allocate long-term investments for this profile",
) -> AdvisorState:
    """Build the graph's entry state from pre-computed pipeline inputs."""
    return AdvisorState(
        profile=profile,
        goal=goal,
        baseline=baseline,
        macro=macro,
        valuation=valuation,
        optimizer_hint=optimizer_hint,
        proposal=None,
        critic_report=None,
        feedback=None,
        iteration=0,
        max_iterations=max_iterations,
        history=[],
        route="normal",
        plan=None,
    )
