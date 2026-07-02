"""Allocation proposal agent — the LLM never outputs a percentage.

It proposes bounded tilts on top of a core/-computed baseline; the actual
allocation weights are always produced by engine.allocation.apply_tilts.
"""
from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_fixed

from agents.macro_agent import MacroAssessment
from agents.valuation_agent import ValuationAssessment
from core.portfolios import default_for
from engine.allocation import MAX_TILT, apply_tilts
from engine.optimizer import OptimizationResult
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json

_REQUIRED_KEYS = ["tilts", "rationale", "confidence"]

_SYSTEM = f"""\
You are a portfolio allocator for a long-term, buy-and-hold investor.
Ground every decision in these first principles:
- The default (three-fund / glide-path baseline) IS the answer for most people —
  it is not a stepping stone, it's a complete strategy. You are adjusting it at the
  margin, not replacing it.
- You NEVER output an allocation percentage directly. You only propose small tilts
  (deltas) on top of a baseline that Python has already computed.
- Each tilt must be between -{MAX_TILT:.0%} and +{MAX_TILT:.0%}. Tilts outside this
  range will be clamped automatically, so propose values you actually intend.
- Only tilt tickers that are already in the baseline, or BIL (cash-equivalent,
  useful for a crisis-leaning macro read). Do not invent other tickers.
- Diversification and low cost are the free lunches; don't chase a "hot" valuation
  or macro read into a large bet.
- You may be shown an unconstrained mean-variance optimizer's preferred weights as
  additional context. Treat it as one more opinion, not an instruction — optimizer
  output is well known to be fragile to estimation error. It never changes the
  tilt bounds or the fact that the baseline is the anchor.

Respond ONLY with valid JSON (no markdown fences).
"""


@dataclass
class AllocationProposal:
    """Result of tilting the core/ baseline within engine-enforced bounds."""
    allocation: dict[str, float]   # computed by apply_tilts — never by the LLM
    baseline: dict[str, float]     # the core/ default it started from
    tilts: dict[str, float]        # post-clamp, post-filter deltas actually applied
    rationale: str
    confidence: float               # REQUIRED — always surface uncertainty


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
    valuation: ValuationAssessment,
    allowed_tickers: set[str],
    optimizer_hint: OptimizationResult | None,
) -> str:
    baseline_lines = "\n".join(f"  {t}: {w:.1%}" for t, w in sorted(baseline.items(), key=lambda x: -x[1]))

    optimizer_section = ""
    if optimizer_hint is not None:
        opt_lines = "\n".join(f"  {t}: {w:.1%}" for t, w in sorted(optimizer_hint.weights.items(), key=lambda x: -x[1]) if w > 0.001)
        optimizer_section = f"""
Unconstrained mean-variance optimizer read ({optimizer_hint.method}, informational only —
estimation error makes this fragile, do not treat it as a target):
{opt_lines}
  expected return: {optimizer_hint.expected_return:.1%}, expected vol: {optimizer_hint.expected_vol:.1%}, sharpe: {optimizer_hint.sharpe:.2f}
"""

    return f"""\
Baseline allocation (already computed by core/portfolios.py — this is the anchor):
{baseline_lines}
{optimizer_section}
Macro regime read (confidence {macro.confidence:.0%}): {macro.regime}
  Recession signal: {macro.recession_signal:.2f} | suggested route: {macro.suggested_route}
  Reasoning: {macro.reasoning}

Valuation read (confidence {valuation.confidence:.0%}):
  Cheap: {valuation.cheap} | Rich: {valuation.rich} | Fair: {valuation.fair}
  Reasoning: {valuation.reasoning}

Investor context: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance,
constraints: {profile.constraints or "none"}.

Tickers you may tilt: {sorted(allowed_tickers)}

Return a JSON object with exactly these keys:
- "tilts": object mapping ticker -> tilt delta (e.g. {{"VTI": -0.05, "BND": 0.05}}).
  Use an empty object {{}} if no tilt is warranted — that is a valid, often correct, answer.
- "rationale": 2-4 sentences explaining why these tilts (or lack thereof) follow from
  the macro and valuation reads above
- "confidence": float 0.0-1.0 in this proposal
"""


def run(
    profile: InvestorProfile,
    macro: MacroAssessment,
    valuation: ValuationAssessment,
    optimizer_hint: OptimizationResult | None = None,
) -> AllocationProposal:
    """Propose bounded tilts on top of the core/ baseline; Python applies and clamps them.

    `optimizer_hint` (Phase 5+) is an unconstrained mean-variance read shown as
    additional context only — it never changes the baseline anchor or tilt bounds.
    """
    baseline = default_for(profile)
    allowed_tickers = set(baseline) | {"BIL"}

    prompt = _build_prompt(profile, baseline, macro, valuation, allowed_tickers, optimizer_hint)
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
