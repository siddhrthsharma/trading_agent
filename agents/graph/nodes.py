"""Thin LangGraph node adapters — call existing agent run() functions, do no math themselves.

Nodes read from and write to AdvisorState only; they never call an LLM directly and
never touch engine/core math beyond what the wrapped agent already returns.
"""
from __future__ import annotations

from agents import allocator_agent, critic_agent
from agents.graph.state import AdvisorState


def allocator_node(state: AdvisorState) -> dict:
    """Call allocator_agent.run with any critic feedback; bump the iteration counter."""
    proposal = allocator_agent.run(
        state["profile"],
        state["macro"],
        state["valuation"],
        optimizer_hint=state.get("optimizer_hint"),
        feedback=state.get("feedback"),
    )
    return {"proposal": proposal, "iteration": state["iteration"] + 1}


def critic_node(state: AdvisorState) -> dict:
    """Critique the current proposal; record a history entry; set feedback if it failed."""
    report = critic_agent.run(state["proposal"], state["macro"], state["valuation"], state["profile"])
    entry = {
        "iteration": state["iteration"],
        "tilts": state["proposal"].tilts,
        "allocation": state["proposal"].allocation,
        "rationale": state["proposal"].rationale,
        "passed": report.passed,
        "severity": report.severity,
        "violations": [f"[{v.severity}] {v.rule}: {v.detail}" for v in report.violations],
        "qualitative_issues": report.qualitative_issues,
    }
    return {
        "critic_report": report,
        "history": state["history"] + [entry],
        "feedback": None if report.passed else report,
    }
