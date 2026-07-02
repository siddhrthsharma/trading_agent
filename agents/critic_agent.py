"""Critic agent — always runs; deterministic rule checks first, LLM judgment second.

Phase 4: flags only, no revision loop yet (that's Phase 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_fixed

from agents.allocator_agent import AllocationProposal
from agents.macro_agent import MacroAssessment
from agents.valuation_agent import ValuationAssessment
from engine.allocation import RuleViolation, check_rules
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json

_REQUIRED_KEYS = ["qualitative_issues", "confidence", "reasoning"]

_SYSTEM = """\
You are a skeptical devil's-advocate critic for a long-term investment allocation.
Ground every critique in these first principles:
- The honest low-cost default is sacred; any deviation from it must earn its place.
- Diversification, low cost, and buy-and-hold are the priors — flag anything that
  works against them (unnecessary complexity, cost creep, concentration, market timing).
- Be skeptical of the macro/valuation reasoning itself: is the proposed tilt actually
  justified by the stated regime and valuation reads, or is it a bigger bet than the
  evidence supports?
- Check whether the allocation fits the investor's stated risk tolerance and horizon.
- Uncertainty must be surfaced, never hidden.

You do not recompute or re-derive any numbers — deterministic rule checks already ran
in Python. You only add qualitative judgment on top.
Respond ONLY with valid JSON (no markdown fences).
"""


@dataclass
class CriticReport:
    """Critique of an AllocationProposal — deterministic checks + qualitative judgment."""
    passed: bool                                    # computed from violation severities
    violations: list[RuleViolation] = field(default_factory=list)
    qualitative_issues: list[str] = field(default_factory=list)
    severity: str = "none"                          # "none" | "minor" | "serious"
    confidence: float = 0.0                           # REQUIRED — always surface uncertainty
    reasoning: str = ""


def _validate(data: dict) -> None:
    if not isinstance(data["qualitative_issues"], list):
        raise ValueError(f"qualitative_issues must be a list, got {type(data['qualitative_issues'])!r}")
    if not 0.0 <= float(data["confidence"]) <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {data['confidence']!r}")


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _generate(prompt: str) -> dict:
    """Call the LLM and validate the response; retries the whole round-trip on failure."""
    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)
    _validate(data)
    return data


def _severity_of(violations: list[RuleViolation]) -> str:
    if any(v.severity == "serious" for v in violations):
        return "serious"
    if violations:
        return "minor"
    return "none"


def _build_prompt(
    proposal: AllocationProposal,
    macro: MacroAssessment,
    valuation: ValuationAssessment,
    profile: InvestorProfile,
    violations: list[RuleViolation],
) -> str:
    alloc_lines = "\n".join(f"  {t}: {w:.1%}" for t, w in sorted(proposal.allocation.items(), key=lambda x: -x[1]))
    violation_lines = "\n".join(f"  [{v.severity}] {v.rule}: {v.detail}" for v in violations) or "  (none)"
    return f"""\
Proposed allocation:
{alloc_lines}

Baseline it was tilted from: {proposal.baseline}
Tilts applied: {proposal.tilts}
Allocator's rationale: {proposal.rationale}

Deterministic rule check results (already computed — do not re-derive):
{violation_lines}

Macro regime: {macro.regime} (confidence {macro.confidence:.0%}, route {macro.suggested_route})
Valuation read: cheap={valuation.cheap}, rich={valuation.rich} (confidence {valuation.confidence:.0%})
Investor: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance,
constraints: {profile.constraints or "none"}.

Return a JSON object with exactly these keys:
- "qualitative_issues": list of strings, each a specific concern (empty list if none) —
  e.g. mismatched risk tolerance, tilt not well-justified by the stated regime/valuation,
  cost creep, or anything else a skeptical reviewer would flag
- "confidence": float 0.0-1.0 in this critique
- "reasoning": 2-4 sentences summarizing your overall assessment
"""


def run(
    proposal: AllocationProposal,
    macro: MacroAssessment,
    valuation: ValuationAssessment,
    profile: InvestorProfile,
) -> CriticReport:
    """Run deterministic rule checks, then layer qualitative LLM critique on top."""
    violations = check_rules(proposal.allocation)
    severity = _severity_of(violations)

    prompt = _build_prompt(proposal, macro, valuation, profile, violations)
    data = _generate(prompt)

    return CriticReport(
        passed=severity != "serious",
        violations=violations,
        qualitative_issues=list(data["qualitative_issues"]),
        severity=severity,
        confidence=float(data["confidence"]),
        reasoning=data["reasoning"],
    )
