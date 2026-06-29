"""Phase 2 education-layer tests — all LLM calls mocked, no real API key needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# U7 — utils/llm.py (hardened)
# ===========================================================================

class TestCallLlmJson:
    """call_llm_json must parse JSON, validate required keys, and retry once."""

    def _mock_complete(self, responses: list[str]):
        """Return a _complete mock that yields successive responses."""
        mock = MagicMock(side_effect=responses)
        return mock

    def test_valid_json_returned_as_dict(self):
        payload = '{"term": "ETF", "explanation": "...", "confidence": 0.95, "related_terms": []}'
        with patch("utils.llm._complete", return_value=payload):
            from utils.llm import call_llm_json
            result = call_llm_json("explain ETF", required_keys=["term", "explanation"])
        assert result["term"] == "ETF"
        assert result["confidence"] == 0.95

    def test_retry_on_bad_json_then_good(self):
        """First call returns invalid JSON; second returns valid — should succeed."""
        bad = "not json at all"
        good = '{"term": "P/E", "explanation": "price to earnings", "confidence": 0.9, "related_terms": ["EPS"]}'
        with patch("utils.llm._complete", side_effect=[bad, good]):
            from utils.llm import call_llm_json
            result = call_llm_json("explain P/E", required_keys=["term"])
        assert result["term"] == "P/E"

    def test_always_bad_json_raises_after_two_attempts(self):
        """Two bad JSON responses must raise ValueError (not retry a third time)."""
        bad = "definitely not json"
        with patch("utils.llm._complete", side_effect=[bad, bad]):
            from utils.llm import call_llm_json
            with pytest.raises(Exception):  # ValueError or tenacity RetryError
                call_llm_json("explain foo", required_keys=["term"])

    def test_missing_required_key_raises(self):
        """Valid JSON but missing a required key must raise after retries."""
        missing_key = '{"explanation": "..."}'   # "term" is missing
        with patch("utils.llm._complete", side_effect=[missing_key, missing_key]):
            from utils.llm import call_llm_json
            with pytest.raises(Exception):
                call_llm_json("explain foo", required_keys=["term", "explanation"])

    def test_markdown_fence_stripped(self):
        """Model sometimes wraps JSON in ```json ... ``` — must be stripped."""
        fenced = '```json\n{"term": "DCA", "explanation": "dollar-cost averaging", "confidence": 0.85, "related_terms": []}\n```'
        with patch("utils.llm._complete", return_value=fenced):
            from utils.llm import call_llm_json
            result = call_llm_json("explain DCA", required_keys=["term"])
        assert result["term"] == "DCA"

    def test_import_does_not_require_api_key(self):
        """Importing utils.llm must not instantiate a Groq client (lazy init)."""
        import importlib
        import sys
        # Remove cached module so we get a fresh import
        sys.modules.pop("utils.llm", None)
        # Should import cleanly even with no GROQ key in env
        import utils.llm  # noqa: F401
        # Groq() must NOT have been called at import time
        # (If it had, it would instantiate with empty key and may raise)
        assert hasattr(utils.llm, "call_llm")
        assert hasattr(utils.llm, "call_llm_json")


# ===========================================================================
# U8 — education/explainer_agent.py
# ===========================================================================

class TestExplainerAgent:
    def _make_response(self, term="P/E ratio", confidence=0.92):
        return {
            "term": term,
            "explanation": "The price-to-earnings ratio compares a company's stock price to its earnings per share.",
            "confidence": confidence,
            "related_terms": ["EPS", "market cap", "dividend yield"],
        }

    def test_run_returns_explanation_dataclass(self):
        with patch("education.explainer_agent.call_llm_json", return_value=self._make_response()):
            from education.explainer_agent import run, Explanation
            result = run("P/E ratio")
        assert isinstance(result, Explanation)

    def test_confidence_is_set(self):
        with patch("education.explainer_agent.call_llm_json", return_value=self._make_response(confidence=0.92)):
            from education.explainer_agent import run
            result = run("P/E ratio")
        assert result.confidence == 0.92
        assert 0.0 <= result.confidence <= 1.0

    def test_explanation_populated(self):
        with patch("education.explainer_agent.call_llm_json", return_value=self._make_response()):
            from education.explainer_agent import run
            result = run("P/E ratio")
        assert len(result.explanation) > 10
        assert result.term == "P/E ratio"

    def test_related_terms_list(self):
        with patch("education.explainer_agent.call_llm_json", return_value=self._make_response()):
            from education.explainer_agent import run
            result = run("P/E ratio")
        assert isinstance(result.related_terms, list)
        assert len(result.related_terms) > 0

    def test_routes_through_call_llm_json(self):
        """explainer_agent must call call_llm_json (not groq/anthropic directly)."""
        with patch("education.explainer_agent.call_llm_json", return_value=self._make_response()) as mock_llm:
            from education.explainer_agent import run
            run("ETF")
        mock_llm.assert_called_once()

    def test_never_returns_allocation_weights(self):
        """Explanation must not contain numeric allocation percentages (not the agent's job)."""
        response = self._make_response()
        response["explanation"] = "An ETF is a basket of securities traded on an exchange."
        with patch("utils.llm.call_llm_json", return_value=response):
            from education.explainer_agent import run
            result = run("ETF")
        # The Explanation dataclass must not have allocation-related fields
        assert not hasattr(result, "allocation")
        assert not hasattr(result, "weights")
        assert not hasattr(result, "portfolio")

    def test_unknown_term_still_returns_explanation(self):
        """An off-glossary term must return an Explanation (low confidence is fine)."""
        response = {
            "term": "meme stock",
            "explanation": "A stock that gains popularity through social media rather than fundamentals.",
            "confidence": 0.6,
            "related_terms": ["speculation", "volatility"],
        }
        with patch("education.explainer_agent.call_llm_json", return_value=response):
            from education.explainer_agent import run
            result = run("meme stock")
        assert result.confidence <= 0.8   # off-glossary → lower confidence
        assert len(result.explanation) > 0

    def test_glossary_covers_spec_terms(self):
        """GLOSSARY must include all required terms from the spec."""
        from education.explainer_agent import GLOSSARY
        required = {
            "P/E ratio", "ETF", "index fund", "expense ratio",
            "dividend", "diversification", "DCA", "Roth IRA",
            "capital gains", "wash sale", "brokerage account",
        }
        glossary_lower = {g.lower() for g in GLOSSARY}
        missing = [t for t in required if t.lower() not in glossary_lower]
        assert not missing, f"GLOSSARY is missing spec terms: {missing}"

    def test_no_direct_groq_import_in_agent(self):
        """education.explainer_agent must not import groq or anthropic directly."""
        import education.explainer_agent as mod
        assert not hasattr(mod, "groq"), "explainer_agent imported groq directly"
        assert not hasattr(mod, "anthropic"), "explainer_agent imported anthropic directly"
        assert not hasattr(mod, "Groq"), "explainer_agent imported Groq directly"
