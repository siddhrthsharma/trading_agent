"""Macro regime interpreter — LLM reasons over engine-computed signals, never math."""
from __future__ import annotations

from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_fixed

from engine.signals import MacroSnapshot
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json, call_llm_with_tools

_RATE_ENVIRONMENTS = {"elevated", "neutral", "easing"}
_ROUTES = {"normal", "crisis"}
_REQUIRED_KEYS = [
    "regime", "rate_environment", "recession_signal",
    "implications", "confidence", "reasoning", "suggested_route",
]

_SYSTEM = """\
You are a macro regime analyst for a long-term, buy-and-hold investor.
Ground every judgment in these first principles:
- Time in the market beats timing the market; never make confident point predictions.
- Broad equities return ~10% nominal / ~7% real over the long run — don't imply otherwise.
- Yield curve inversion (10Y-2Y spread < 0) has preceded every US recession in 50 years,
  but is not a perfect predictor — treat it as one signal among several.
- Real rates (fed funds minus inflation) above zero are restrictive; below zero are accommodative.
- Rising unemployment is a lagging, late-cycle signal.
- "Don't fight the Fed" — rate direction matters more than the current level.
- Never recommend market timing or all-in/all-out moves; reason about tilts and risk, not calls.

You interpret pre-computed macro numbers. You never compute or invent numbers yourself.
Respond ONLY with valid JSON (no markdown fences).
"""

_TOOL_SYSTEM = _SYSTEM + """
You have tools to pull the macro data you need: get_macro_snapshot (the full
derived snapshot), fetch_fred_series (any allowlisted FRED series), compute_yield_curve,
and compute_real_rate. Call what you need before answering — don't guess at numbers
you could look up.
"""


@dataclass
class MacroAssessment:
    """Qualitative regime read derived from a MacroSnapshot."""
    regime: str                    # e.g. "late-cycle restrictive"
    rate_environment: str          # "elevated" | "neutral" | "easing"
    recession_signal: float        # 0.0-1.0
    implications: dict[str, str]   # {"equities": ..., "bonds": ..., "cash": ...}
    confidence: float               # REQUIRED — always surface uncertainty
    reasoning: str
    suggested_route: str           # "normal" | "crisis" — supervisor reads this (Phase 9+)


def _validate(data: dict) -> None:
    if data["rate_environment"] not in _RATE_ENVIRONMENTS:
        raise ValueError(f"rate_environment must be one of {_RATE_ENVIRONMENTS}, got {data['rate_environment']!r}")
    if data["suggested_route"] not in _ROUTES:
        raise ValueError(f"suggested_route must be one of {_ROUTES}, got {data['suggested_route']!r}")
    if not 0.0 <= float(data["recession_signal"]) <= 1.0:
        raise ValueError(f"recession_signal must be in [0,1], got {data['recession_signal']!r}")
    if not 0.0 <= float(data["confidence"]) <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {data['confidence']!r}")
    if not isinstance(data["implications"], dict):
        raise ValueError(f"implications must be a dict, got {type(data['implications'])!r}")


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _generate(prompt: str) -> dict:
    """Call the LLM and validate the response; retries the whole round-trip on failure."""
    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)
    _validate(data)
    return data


def _build_prompt(snapshot: MacroSnapshot, profile: InvestorProfile) -> str:
    return f"""\
Interpret the following pre-computed macro snapshot as of {snapshot.as_of.date()}:

- Fed funds rate: {snapshot.fed_funds:.2f}%
- 10Y Treasury yield: {snapshot.dgs10:.2f}%
- 2Y Treasury yield: {snapshot.dgs2:.2f}%
- Yield curve spread (10Y-2Y): {snapshot.yield_curve_spread:.2f} points
- Curve inverted: {snapshot.curve_inverted}
- CPI year-over-year (inflation): {snapshot.cpi_yoy:.2%}
- Real fed funds rate (nominal minus inflation): {snapshot.real_fed_funds:.2f} points
- Unemployment rate: {snapshot.unemployment:.2f}%
- Unemployment change vs 1 year ago: {snapshot.unemployment_change_1y:+.2f} points

Investor context: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance.

Return a JSON object with exactly these keys:
- "regime": short label, e.g. "late-cycle restrictive" or "early recovery"
- "rate_environment": one of "elevated", "neutral", "easing"
- "recession_signal": float 0.0-1.0, your judgment of recession risk grounded in the signals above
- "implications": object with keys "equities", "bonds", "cash" — one sentence each
- "confidence": float 0.0-1.0 — how confident you are in this regime read
- "reasoning": 2-4 sentences grounding your read in the specific numbers above
- "suggested_route": "crisis" if recession_signal > 0.8, else "normal"
"""


def _to_assessment(data: dict) -> MacroAssessment:
    return MacroAssessment(
        regime=data["regime"],
        rate_environment=data["rate_environment"],
        recession_signal=float(data["recession_signal"]),
        implications=dict(data["implications"]),
        confidence=float(data["confidence"]),
        reasoning=data["reasoning"],
        suggested_route=data["suggested_route"],
    )


def run_static(snapshot: MacroSnapshot, profile: InvestorProfile) -> MacroAssessment:
    """Interpret a pre-packaged macro snapshot into a regime assessment (Phase 4 path).

    Kept as the offline/test entry point — no network calls beyond the LLM
    itself, since the snapshot is already computed by the caller.
    """
    prompt = _build_prompt(snapshot, profile)
    data = _generate(prompt)
    return _to_assessment(data)


def _build_tool_prompt(profile: InvestorProfile, as_of: str | None) -> str:
    as_of_clause = f" as of {as_of}" if as_of else " (use the latest available data)"
    return f"""\
Assess the current macro regime{as_of_clause} for an investor with a
{profile.horizon_years}-year horizon and {profile.risk_tolerance} risk tolerance.

Use your tools to gather the yield curve, real rate, CPI, and unemployment picture,
then return a JSON object with exactly these keys:
- "regime": short label, e.g. "late-cycle restrictive" or "early recovery"
- "rate_environment": one of "elevated", "neutral", "easing"
- "recession_signal": float 0.0-1.0, your judgment of recession risk grounded in the data you gathered
- "implications": object with keys "equities", "bonds", "cash" — one sentence each
- "confidence": float 0.0-1.0 — how confident you are in this regime read
- "reasoning": 2-4 sentences grounding your read in the specific numbers you looked up
- "suggested_route": "crisis" if recession_signal > 0.8, else "normal"
"""


def run(profile: InvestorProfile, as_of: str | None = None) -> MacroAssessment:
    """Assess the macro regime by pulling data via MACRO_TOOLS on demand (Phase 6).

    `as_of`, if given, is an ISO date string (e.g. "2022-09-30") passed through
    to get_macro_snapshot for a historical read.
    """
    from agents.tools.macro_tools import MACRO_TOOLS

    prompt = _build_tool_prompt(profile, as_of)
    data = call_llm_with_tools(prompt, tools=MACRO_TOOLS, required_keys=_REQUIRED_KEYS, system=_TOOL_SYSTEM)
    _validate(data)
    return _to_assessment(data)
