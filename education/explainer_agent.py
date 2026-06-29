"""Plain-language financial education agent — LLM, read-only, no math."""
from __future__ import annotations

from dataclasses import dataclass, field

from utils.llm import call_llm_json

# Full glossary from CURSOR_PROJECT_SPEC.md §Beginner Knowledge Base
GLOSSARY: set[str] = {
    "stock", "share", "market cap", "P/E ratio", "EPS", "earnings per share",
    "dividend", "dividend yield", "DRIP", "volume", "52-week range",
    "index", "index fund", "ETF", "expense ratio",
    "bull market", "bear market", "diversification",
    "dollar-cost averaging", "DCA",
    "capital gains", "short-term capital gains", "long-term capital gains",
    "realized gains", "unrealized gains",
    "wash sale", "wash-sale rule", "tax-loss harvesting",
    "Roth IRA", "traditional IRA", "Roth vs traditional",
    "brokerage account",
    "quantitative easing", "quantitative tightening", "QE", "QT",
    "don't fight the Fed",
}

_SYSTEM = """\
You are a patient, clear financial educator for a long-term investor who wants to learn.
Ground every answer in these first principles:
- Time in market beats timing the market; equities return ~10% nominal / ~7% real over the long run.
- Diversification (broad index funds/ETFs) is the only free lunch.
- Low cost (expense ratio 0.03–0.10%) compounds into enormous advantages over decades.
- Buy-and-hold, dollar-cost average, and automate — never go all-in.
- Tax-advantaged accounts (Roth IRA, 401k) first; Roth is especially powerful for young investors.

Explain terms in plain language. Never output allocation percentages or investment advice.
Respond ONLY with valid JSON (no markdown fences).
"""

_REQUIRED_KEYS = ["term", "explanation", "confidence", "related_terms"]


@dataclass
class Explanation:
    """Structured educational response from the explainer agent."""
    term: str
    explanation: str
    confidence: float            # 0.0–1.0; always surfaced, never hidden
    related_terms: list[str] = field(default_factory=list)


def run(term: str, profile=None) -> Explanation:  # noqa: ARG001
    """Explain a financial term in plain language.

    profile is accepted for future personalisation but not used in the prompt today.
    confidence is always set — uncertainty is surfaced, never hidden.
    """
    in_glossary = term.lower() in {g.lower() for g in GLOSSARY}
    confidence_hint = (
        "This is a core concept you know well — set confidence to 0.9 or higher."
        if in_glossary
        else "This term may be outside the standard glossary — be honest and set confidence accordingly."
    )

    prompt = f"""\
Explain the following financial term to a long-term investor who is learning:

Term: "{term}"

{confidence_hint}

Return a JSON object with exactly these keys:
- "term": the term as provided
- "explanation": clear, plain-language explanation (2–4 sentences)
- "confidence": float 0.0–1.0 indicating how confident you are in this explanation
- "related_terms": list of 2–4 related terms the learner might want to explore next
"""

    data = call_llm_json(prompt, required_keys=_REQUIRED_KEYS, system=_SYSTEM)

    return Explanation(
        term=data["term"],
        explanation=data["explanation"],
        confidence=float(data["confidence"]),
        related_terms=list(data.get("related_terms", [])),
    )
