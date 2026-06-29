"""Single LLM interface — all agents import from here, never directly from groq/anthropic."""
from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a helpful financial educator. Be concise and accurate."
_DEFAULT_TEMPERATURE = 0.3
_JSON_TEMPERATURE = 0.2


def _get_client():
    """Lazy-init the Groq client so importing this module never requires a key."""
    from groq import Groq
    from utils.config import get_settings
    return Groq(api_key=get_settings().groq_api_key)


def _complete(prompt: str, system: str, temperature: float) -> str:
    """Execute one LLM completion and return the raw text response."""
    client = _get_client()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content


def call_llm(
    prompt: str,
    system: str = _DEFAULT_SYSTEM,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> str:
    """Call the LLM and return raw text. Use for non-structured responses."""
    return _complete(prompt, system, temperature)


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
def call_llm_json(
    prompt: str,
    required_keys: list[str],
    system: str = _DEFAULT_SYSTEM,
    temperature: float = _JSON_TEMPERATURE,
) -> dict[str, Any]:
    """Call the LLM expecting a JSON response; validate required_keys are present.

    Retries once on parse failure or missing keys (2 total attempts).
    Raises ValueError if the response is invalid after all retries.
    """
    json_prompt = (
        prompt
        + "\n\nRespond with valid JSON only — no markdown fences, no explanation outside the JSON."
    )
    raw = _complete(json_prompt, system, temperature)

    try:
        # Strip markdown fences if the model added them despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON (will retry): %s", raw[:200])
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.warning("LLM JSON missing keys %s (will retry): %s", missing, raw[:200])
        raise ValueError(f"LLM JSON missing required keys: {missing}")

    return data
