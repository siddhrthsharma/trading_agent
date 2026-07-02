"""Valuation interpreter — LLM reasons over fundamentals data, never math."""
from __future__ import annotations

from dataclasses import dataclass, field

from tenacity import retry, stop_after_attempt, wait_fixed

from data.snapshots import ValuationSnapshot
from profile.investor_profile import InvestorProfile
from utils.llm import call_llm_json, call_llm_with_tools

_REQUIRED_KEYS = ["cheap", "rich", "fair", "notes", "confidence", "reasoning"]

_SYSTEM = """\
You are a valuation analyst for a long-term, buy-and-hold investor.
Ground every judgment in these first principles:
- Diversification (broad index funds) is the only free lunch — you are assessing
  broad asset classes, not stock-picking within them.
- Cost (expense ratio) is fully controllable; a cheap, diversified fund is rarely "rich"
  in a way that should change an allocation meaningfully.
- Valuation reads are humble, not confident predictions — "cheap" can stay cheap or get
  cheaper; never suggest all-in/all-out moves based on valuation alone.
- Only yfinance's CURRENT snapshot is available — you have no reliable historical P/E
  series for these ETFs. If asked to judge whether something is cheap or rich vs its own
  history, say so honestly with a lower confidence rather than inventing a comparison.

You interpret pre-computed fundamentals numbers. You never compute or invent numbers.
Respond ONLY with valid JSON (no markdown fences).
"""

_TOOL_SYSTEM = _SYSTEM + """
You have tools to pull fundamentals: fetch_etf_fundamentals (current P/E, yield,
expense ratio for one ticker) and compare_historical_pe (against locally accumulated
snapshots — usually thin). Call what you need for the tickers relevant to the
investor's universe before answering.
"""


@dataclass
class ValuationAssessment:
    """Qualitative cheap/rich read derived from a ValuationSnapshot."""
    cheap: list[str] = field(default_factory=list)   # asset-class labels, e.g. "intl_equity"
    rich: list[str] = field(default_factory=list)
    fair: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0          # REQUIRED — always surface uncertainty
    reasoning: str = ""


def _validate(data: dict) -> None:
    for key in ("cheap", "rich", "fair"):
        if not isinstance(data[key], list):
            raise ValueError(f"{key} must be a list, got {type(data[key])!r}")
    if not isinstance(data["notes"], dict):
        raise ValueError(f"notes must be a dict, got {type(data['notes'])!r}")
    if not 0.0 <= float(data["confidence"]) <= 1.0:
        raise ValueError(f"confidence must be in [0,1], got {data['confidence']!r}")


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def _generate(prompt: str) -> dict:
    """Call the LLM and validate the response; retries the whole round-trip on failure."""
    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)
    _validate(data)
    return data


def _format_fund(ticker: str, fields: dict[str, float | None]) -> str:
    pe = f"{fields['pe']:.1f}" if fields.get("pe") is not None else "unavailable"
    div_yield = f"{fields['dividend_yield']:.2%}" if fields.get("dividend_yield") is not None else "unavailable"
    expense_ratio = f"{fields['expense_ratio']:.2%}" if fields.get("expense_ratio") is not None else "unavailable"
    return f"  {ticker}: P/E={pe}, dividend yield={div_yield}, expense ratio={expense_ratio}"


def _build_prompt(snapshot: ValuationSnapshot, profile: InvestorProfile) -> str:
    fund_lines = "\n".join(_format_fund(t, f) for t, f in snapshot.funds.items())
    return f"""\
Interpret the following current fundamentals snapshot as of {snapshot.as_of.date()}
(this is a CURRENT-only snapshot; no reliable historical series is available):

{fund_lines}

Investor context: {profile.horizon_years}-year horizon, {profile.risk_tolerance} risk tolerance.

Return a JSON object with exactly these keys:
- "cheap": list of asset-class labels (e.g. "us_equity", "intl_equity", "bonds", "reits")
  you judge relatively cheap right now, based on P/E and yield levels shown above
- "rich": list of asset-class labels you judge relatively rich
- "fair": list of asset-class labels you judge fairly valued
- "notes": object mapping each labeled asset class to a one-sentence rationale
- "confidence": float 0.0-1.0 — lower this if the data is thin or unavailable
- "reasoning": 2-4 sentences grounding your read in the specific numbers above,
  explicitly noting you lack historical comparison data
"""


def _to_assessment(data: dict) -> ValuationAssessment:
    return ValuationAssessment(
        cheap=list(data["cheap"]),
        rich=list(data["rich"]),
        fair=list(data["fair"]),
        notes=dict(data["notes"]),
        confidence=float(data["confidence"]),
        reasoning=data["reasoning"],
    )


def run_static(snapshot: ValuationSnapshot, profile: InvestorProfile) -> ValuationAssessment:
    """Interpret a pre-packaged fundamentals snapshot into a cheap/rich/fair read (Phase 4 path).

    Kept as the offline/test entry point — no network calls beyond the LLM
    itself, since the snapshot is already computed by the caller.
    """
    prompt = _build_prompt(snapshot, profile)
    data = _generate(prompt)
    return _to_assessment(data)


def _build_tool_prompt(profile: InvestorProfile) -> str:
    return f"""\
Assess current valuations across the core asset-class universe (US equity, international
equity, emerging markets, bonds, REITs — whichever apply) for an investor with a
{profile.horizon_years}-year horizon and {profile.risk_tolerance} risk tolerance.

Use your tools to gather current fundamentals for the relevant tickers, then return
a JSON object with exactly these keys:
- "cheap": list of asset-class labels you judge relatively cheap right now
- "rich": list of asset-class labels you judge relatively rich
- "fair": list of asset-class labels you judge fairly valued
- "notes": object mapping each labeled asset class to a one-sentence rationale
- "confidence": float 0.0-1.0 — lower this if the data you gathered is thin or unavailable
- "reasoning": 2-4 sentences grounding your read in the specific numbers you looked up,
  explicitly noting you lack historical comparison data
"""


def run(profile: InvestorProfile) -> ValuationAssessment:
    """Assess valuations by pulling fundamentals via VALUATION_TOOLS on demand (Phase 6)."""
    from agents.tools.valuation_tools import VALUATION_TOOLS

    prompt = _build_tool_prompt(profile)
    data = call_llm_with_tools(prompt, tools=VALUATION_TOOLS, required_keys=_REQUIRED_KEYS, system=_TOOL_SYSTEM)
    _validate(data)
    return _to_assessment(data)
