"""Supervisor — plans steps for a goal (LLM, qualitative) and routes (pure Python, deterministic).

The LLM's judgment about recession risk lives entirely in macro_agent's
MacroAssessment (recession_signal, suggested_route). decide_route() reads
that typed output and returns a graph-edge decision -- it is a pure function,
unit-testable with no LLM call of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_fixed

from agents.macro_agent import MacroAssessment
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json

CRISIS_THRESHOLD = 0.8

_REQUIRED_KEYS = ["steps", "rationale", "confidence"]

_SYSTEM = """\
You are a planning supervisor for a long-term, buy-and-hold investment advisor system.
Ground every plan in these first principles:
- The system answers "where/how should I allocate my money?" for long-term investing,
  never short-term trades. The human is always the final decision-maker.
- Emergency fund and account priority (Roth IRA, 401k, taxable) come before any
  allocation decision.
- You plan the STEPS the system will take (which specialist agents run, in what
  order); you never propose an allocation percentage yourself.

Respond ONLY with valid JSON (no markdown fences).
"""


@dataclass
class SupervisorPlan:
    """A plan for how to answer the investor's goal — qualitative only, no numbers invented."""
    goal: str
    steps: list[str]
    rationale: str
    confidence: float          # REQUIRED — always surface uncertainty


def _validate(data: dict) -> None:
    if not isinstance(data["steps"], list):
        raise ValueError(f"steps must be a list, got {type(data['steps'])!r}")
    if not 0.0 <= float(data["confidence"]) <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {data['confidence']!r}")


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _generate(prompt: str) -> dict:
    """Call the LLM and validate the response; retries the whole round-trip on failure."""
    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)
    _validate(data)
    return data


def _build_prompt(goal: str, profile: InvestorProfile) -> str:
    return f"""\
Investor goal: "{goal}"

Investor context: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance,
monthly contribution ${profile.monthly_contribution:.0f}, constraints: {profile.constraints or "none"}.

Return a JSON object with exactly these keys:
- "steps": ordered list of short strings describing the steps the system will take to
  answer this goal (e.g. "check emergency fund", "read macro regime", "read valuations",
  "propose an allocation", "critique the proposal")
- "rationale": 2-4 sentences explaining why these steps, in this order, answer the goal
- "confidence": float 0.0-1.0 in this plan
"""


def run(goal: str, profile: InvestorProfile) -> SupervisorPlan:
    """Plan the steps to answer the investor's goal (LLM, qualitative only)."""
    prompt = _build_prompt(goal, profile)
    data = _generate(prompt)
    return SupervisorPlan(
        goal=goal,
        steps=list(data["steps"]),
        rationale=data["rationale"],
        confidence=float(data["confidence"]),
    )


def decide_route(macro: MacroAssessment) -> str:
    """Deterministic routing read: crisis iff the macro agent says so or the signal is extreme."""
    if macro.suggested_route == "crisis" or macro.recession_signal > CRISIS_THRESHOLD:
        return "crisis"
    return "normal"
