"""Single LLM interface — all agents import from here, never directly from groq/anthropic."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a helpful financial educator. Be concise and accurate."
_DEFAULT_TEMPERATURE = 0.3
_JSON_TEMPERATURE = 0.2
_MODEL = "llama-3.3-70b-versatile"


def _get_client():
    """Lazy-init the Groq client so importing this module never requires a key."""
    from groq import Groq
    from utils.config import get_settings
    return Groq(api_key=get_settings().groq_api_key)


def _complete(prompt: str, system: str, temperature: float) -> str:
    """Execute one LLM completion and return the raw text response."""
    client = _get_client()
    completion = client.chat.completions.create(
        model=_MODEL,
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


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences if the model added them despite instructions."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _parse_json_response(raw: str, required_keys: list[str]) -> dict[str, Any]:
    """Parse and validate an LLM JSON response; raises ValueError on any failure."""
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned non-JSON (will retry): %s", (raw or "")[:200])
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.warning("LLM JSON missing keys %s (will retry): %s", missing, (raw or "")[:200])
        raise ValueError(f"LLM JSON missing required keys: {missing}")

    return data


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
    return _parse_json_response(raw, required_keys)


@dataclass(frozen=True)
class ToolSpec:
    """A single tool an agent can call mid-reasoning via call_llm_with_tools."""
    name: str
    description: str
    parameters: dict              # JSON Schema for the tool's arguments
    fn: Callable[..., dict]       # executes the tool; returns a JSON-serializable dict


def _to_api_tool(spec: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _execute_tool_call(tool_call, tools_by_name: dict[str, ToolSpec]) -> dict:
    """Run one tool call; exceptions become an error payload, never raised — the
    model gets a chance to recover instead of the whole turn crashing."""
    name = tool_call.function.name
    spec = tools_by_name.get(name)
    if spec is None:
        return {"error": f"Unknown tool: {name!r}"}
    try:
        raw_args = tool_call.function.arguments
        args = json.loads(raw_args) if raw_args and raw_args != "null" else {}
        return spec.fn(**args)
    except Exception as exc:
        logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def call_llm_with_tools(
    prompt: str,
    tools: list[ToolSpec],
    required_keys: list[str],
    *,
    system: str = _DEFAULT_SYSTEM,
    max_rounds: int = 5,
    temperature: float = _JSON_TEMPERATURE,
) -> dict[str, Any]:
    """Let the LLM call tools mid-reasoning, then return its final validated JSON answer.

    Loop: send tools -> if the model requests a call, execute it and feed the
    result back -> repeat until the model answers with no tool calls, then
    validate that answer as JSON with required_keys (nudging it to retry, within
    the round budget, if it isn't). Hard stop at max_rounds.
    """
    tools_by_name = {t.name: t for t in tools}
    api_tools = [_to_api_tool(t) for t in tools]
    client = _get_client()

    json_prompt = (
        prompt
        + "\n\nUse the available tools to gather any data you need, then respond with "
        "valid JSON only — no markdown fences, no explanation outside the JSON."
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": json_prompt},
    ]

    for _ in range(max_rounds):
        completion = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            tools=api_tools,
            tool_choice="auto",
            temperature=temperature,
            max_completion_tokens=1024,
        )
        message = completion.choices[0].message

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                result = _execute_tool_call(tool_call, tools_by_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result, default=str),
                })
            continue

        try:
            return _parse_json_response(message.content or "", required_keys)
        except ValueError as exc:
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({"role": "user", "content": f"{exc} Respond again with valid JSON containing all required keys."})

    raise ValueError(f"call_llm_with_tools exhausted {max_rounds} rounds without a valid final answer")
