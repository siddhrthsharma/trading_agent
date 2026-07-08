"""Defensive-allocation specialist — routed to when the macro read signals crisis.

Same discipline as allocator_agent: the LLM never outputs a percentage, only
bounded tilts on top of a core/-computed baseline; engine.allocation.apply_tilts
produces the actual weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_fixed

from agents.allocator_agent import AllocationProposal
from agents.macro_agent import MacroAssessment
from agents.valuation_agent import ValuationAssessment
from core.portfolios import default_for
from engine.allocation import MAX_TILT, apply_tilts
from engine.optimizer import OptimizationResult
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json

if TYPE_CHECKING:
    from agents.critic_agent import CriticReport

_REQUIRED_KEYS = ["tilts", "rationale", "confidence"]

_SYSTEM = f"""\
You are a defensive-allocation specialist for a long-term, buy-and-hold investor,
brought in specifically because the macro read signals a high recession risk.
Ground every decision in these first principles:
- Capital preservation is the priority right now: favor tilts toward short-duration
  bonds and cash-equivalents (BND, BIL) and away from equities, without abandoning
  the diversified baseline entirely — this is a defensive lean, not an all-out exit.
- The default (three-fund / glide-path baseline) is still the anchor. You are
  adjusting it at the margin for the crisis regime, not replacing it.
- You NEVER output an allocation percentage directly. You only propose small tilts
  (deltas) on top of a baseline that Python has already computed.
- Each tilt must be between -{MAX_TILT:.0%} and +{MAX_TILT:.0%}. Tilts outside this
  range will be clamped automatically, so propose values you actually intend.
- Only tilt tickers that are already in the baseline, or BIL (cash-equivalent —
  the most useful lever you have in a crisis-leaning macro read). Do not invent
  other tickers.
- Never recommend market timing or an all-in/all-out move; reason about a bounded
  defensive tilt, not a call on when the crisis ends.

Respond ONLY with valid JSON (no markdown fences).
"""


def _build_feedback_section(feedback: "CriticReport | None") -> str:
    if feedback is None:
        return ""
    violation_lines = "\n".join(f"  [{v.severity}] {v.rule}: {v.detail}" for v in feedback.violations) or "  (none)"
    issue_lines = "\n".join(f"  - {issue}" for issue in feedback.qualitative_issues) or "  (none)"
    return f"""
Your previous proposal was rejected by a critic for the issues below. Revise your
tilts to resolve them -- moving back toward the baseline is usually the right fix.
Rule violations:
{violation_lines}
Qualitative issues:
{issue_lines}
"""


def _validate(data: dict) -> None:
    if not isinstance(data["tilts"], dict):
        raise ValueError(f"tilts must be a dict, got {type(data['tilts'])!r}")
    for ticker, tilt in data["tilts"].items():
        if not isinstance(tilt, (int, float)):
            raise ValueError(f"tilt for {ticker!r} must be numeric, got {tilt!r}")
    if not 0.0 <= float(data["confidence"]) <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {data['confidence']!r}")


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _generate(prompt: str) -> dict:
    """Call the LLM and validate the response; retries the whole round-trip on failure."""
    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)
    _validate(data)
    return data


def _build_prompt(
    profile: InvestorProfile,
    baseline: dict[str, float],
    macro: MacroAssessment,
    valuation: ValuationAssessment | None,
    allowed_tickers: set[str],
    optimizer_hint: OptimizationResult | None,
    feedback: "CriticReport | None" = None,
) -> str:
    baseline_lines = "\n".join(f"  {t}: {w:.1%}" for t, w in sorted(baseline.items(), key=lambda x: -x[1]))
    feedback_section = _build_feedback_section(feedback)

    optimizer_section = ""
    if optimizer_hint is not None:
        opt_lines = "\n".join(f"  {t}: {w:.1%}" for t, w in sorted(optimizer_hint.weights.items(), key=lambda x: -x[1]) if w > 0.001)
        optimizer_section = f"""
Unconstrained mean-variance optimizer read ({optimizer_hint.method}, informational only —
estimation error makes this fragile, do not treat it as a target):
{opt_lines}
  expected return: {optimizer_hint.expected_return:.1%}, expected vol: {optimizer_hint.expected_vol:.1%}, sharpe: {optimizer_hint.sharpe:.2f}
"""

    valuation_section = ""
    if valuation is not None:
        valuation_section = f"""
Valuation read (confidence {valuation.confidence:.0%}):
  Cheap: {valuation.cheap} | Rich: {valuation.rich} | Fair: {valuation.fair}
  Reasoning: {valuation.reasoning}
"""

    return f"""\
Baseline allocation (already computed by core/portfolios.py — this is the anchor):
{baseline_lines}
{optimizer_section}{feedback_section}
Macro regime read (confidence {macro.confidence:.0%}): {macro.regime}
  Recession signal: {macro.recession_signal:.2f} | suggested route: {macro.suggested_route}
  Reasoning: {macro.reasoning}
{valuation_section}
Investor context: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance,
constraints: {profile.constraints or "none"}.

Tickers you may tilt: {sorted(allowed_tickers)}

Return a JSON object with exactly these keys:
- "tilts": object mapping ticker -> tilt delta (e.g. {{"BND": 0.05, "BIL": 0.05, "VTI": -0.10}}).
  Use an empty object {{}} if no defensive tilt is warranted — that is a valid answer.
- "rationale": 2-4 sentences explaining why these defensive tilts (or lack thereof)
  follow from the macro read above
- "confidence": float 0.0-1.0 in this proposal
"""


def run(
    profile: InvestorProfile,
    macro: MacroAssessment,
    valuation: ValuationAssessment | None = None,
    optimizer_hint: OptimizationResult | None = None,
    feedback: "CriticReport | None" = None,
) -> AllocationProposal:
    """Propose bounded defensive tilts on top of the core/ baseline; Python applies and clamps them.

    `valuation` is optional -- the crisis path may skip the valuation agent
    entirely, since capital preservation doesn't depend on cheap/rich reads.
    """
    baseline = default_for(profile)
    allowed_tickers = set(baseline) | {"BIL"}

    prompt = _build_prompt(profile, baseline, macro, valuation, allowed_tickers, optimizer_hint, feedback)
    data = _generate(prompt)

    raw_tilts = {t: float(v) for t, v in data["tilts"].items() if t in allowed_tickers}
    allocation = apply_tilts(baseline, raw_tilts)
    applied_tilts = {t: max(-MAX_TILT, min(MAX_TILT, v)) for t, v in raw_tilts.items()}

    return AllocationProposal(
        allocation=allocation,
        baseline=baseline,
        tilts=applied_tilts,
        rationale=data["rationale"],
        confidence=float(data["confidence"]),
    )
