"""Tests for utils/llm.py's tool-calling loop — Groq client always stubbed."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeClient:
    """Returns scripted completions in sequence, one per call() invocation."""
    def __init__(self, messages: list[_FakeMessage]):
        self._messages = iter(messages)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = next(self._messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_spec(name="fetch_yield_curve", fn=None):
    from utils.llm import ToolSpec
    return ToolSpec(
        name=name,
        description="Get the yield curve spread",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn or (lambda: {"spread": -0.35, "inverted": True}),
    )


# ---------------------------------------------------------------------------
# call_llm_with_tools
# ---------------------------------------------------------------------------

def test_call_llm_with_tools_executes_tool_and_returns_final_json():
    from utils.llm import call_llm_with_tools

    tool_call = _FakeToolCall("call_1", "fetch_yield_curve", "null")
    round1 = _FakeMessage(content=None, tool_calls=[tool_call])
    round2 = _FakeMessage(content='{"regime": "inverted", "confidence": 0.7}', tool_calls=None)
    fake_client = _FakeClient([round1, round2])

    with patch("utils.llm._get_client", return_value=fake_client):
        result = call_llm_with_tools(
            "What's the regime?", tools=[_tool_spec()], required_keys=["regime", "confidence"],
        )

    assert result == {"regime": "inverted", "confidence": 0.7}
    assert len(fake_client.calls) == 2
    # The tool's result must have been fed back into round 2's message history
    round2_messages = fake_client.calls[1]["messages"]
    tool_messages = [m for m in round2_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "-0.35" in tool_messages[0]["content"]


def test_call_llm_with_tools_no_tool_use_still_works():
    """The model may answer directly without calling any tool."""
    from utils.llm import call_llm_with_tools

    direct_answer = _FakeMessage(content='{"regime": "normal", "confidence": 0.5}', tool_calls=None)
    fake_client = _FakeClient([direct_answer])

    with patch("utils.llm._get_client", return_value=fake_client):
        result = call_llm_with_tools(
            "What's the regime?", tools=[_tool_spec()], required_keys=["regime", "confidence"],
        )
    assert result["regime"] == "normal"


def test_call_llm_with_tools_tool_error_becomes_message_not_crash():
    from utils.llm import call_llm_with_tools

    def broken_tool():
        raise RuntimeError("network down")

    tool_call = _FakeToolCall("call_1", "fetch_yield_curve", "null")
    round1 = _FakeMessage(content=None, tool_calls=[tool_call])
    round2 = _FakeMessage(content='{"regime": "unknown", "confidence": 0.2}', tool_calls=None)
    fake_client = _FakeClient([round1, round2])

    with patch("utils.llm._get_client", return_value=fake_client):
        result = call_llm_with_tools(
            "What's the regime?", tools=[_tool_spec(fn=broken_tool)], required_keys=["regime", "confidence"],
        )

    assert result["regime"] == "unknown"  # the loop recovered instead of raising
    round2_messages = fake_client.calls[1]["messages"]
    tool_messages = [m for m in round2_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert "network down" in tool_messages[0]["content"]


def test_call_llm_with_tools_unknown_tool_name_reported_not_crash():
    from utils.llm import call_llm_with_tools

    tool_call = _FakeToolCall("call_1", "some_other_tool", "null")
    round1 = _FakeMessage(content=None, tool_calls=[tool_call])
    round2 = _FakeMessage(content='{"regime": "unknown", "confidence": 0.2}', tool_calls=None)
    fake_client = _FakeClient([round1, round2])

    with patch("utils.llm._get_client", return_value=fake_client):
        result = call_llm_with_tools(
            "What's the regime?", tools=[_tool_spec()], required_keys=["regime", "confidence"],
        )
    assert result["regime"] == "unknown"


def test_call_llm_with_tools_max_rounds_exhausted_raises():
    from utils.llm import call_llm_with_tools

    tool_call = _FakeToolCall("call_1", "fetch_yield_curve", "null")
    # Every round calls the tool again — never answers — should hit max_rounds
    endless = [_FakeMessage(content=None, tool_calls=[tool_call]) for _ in range(10)]
    fake_client = _FakeClient(endless)

    with patch("utils.llm._get_client", return_value=fake_client):
        with pytest.raises(ValueError, match="exhausted"):
            call_llm_with_tools(
                "What's the regime?", tools=[_tool_spec()], required_keys=["regime"], max_rounds=3,
            )
    assert len(fake_client.calls) == 3


def test_call_llm_with_tools_missing_required_key_prompts_retry_then_succeeds():
    from utils.llm import call_llm_with_tools

    incomplete = _FakeMessage(content='{"regime": "normal"}', tool_calls=None)  # missing "confidence"
    complete = _FakeMessage(content='{"regime": "normal", "confidence": 0.6}', tool_calls=None)
    fake_client = _FakeClient([incomplete, complete])

    with patch("utils.llm._get_client", return_value=fake_client):
        result = call_llm_with_tools(
            "What's the regime?", tools=[_tool_spec()], required_keys=["regime", "confidence"],
        )
    assert result["confidence"] == 0.6
    assert len(fake_client.calls) == 2
