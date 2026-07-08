"""Thin LangGraph node adapters — call existing agent run() functions, do no math themselves.

Nodes read from and write to AdvisorState only; they never call an LLM directly and
never touch engine/core math beyond what the wrapped agent already returns.
"""
from __future__ import annotations

from agents import allocator_agent, critic_agent
from agents.graph.state import AdvisorState
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentDataUnavailable(Exception):
    """Raised when neither the tool-driven nor snapshot-based agent read is available."""


def supervisor_node(state: AdvisorState) -> dict:
    """Plan the steps for this goal; planning is advisory and never blocks the pipeline."""
    from agents import supervisor

    try:
        plan = supervisor.run(state["goal"], state["profile"])
        logger.info("Supervisor plan (confidence %.0f%%): %s", plan.confidence * 100, plan.steps)
    except Exception:
        logger.warning("Supervisor planning failed — continuing without a plan.")
        plan = None
    return {"plan": plan}


def macro_node(state: AdvisorState) -> dict:
    """Read the macro regime (tool-driven, falling back to the snapshot-based path) and route."""
    from agents import macro_agent
    from agents.supervisor import decide_route

    profile = state["profile"]
    try:
        macro = macro_agent.run(profile)
        logger.info(
            "Macro regime (tool-driven): %s (confidence %.0f%%, route=%s)",
            macro.regime, macro.confidence * 100, macro.suggested_route,
        )
    except Exception:
        logger.warning("Tool-driven macro read failed — falling back to the snapshot-based path.")
        from data.snapshots import macro_snapshot
        try:
            macro = macro_agent.run_static(macro_snapshot(), profile)
            logger.info(
                "Macro regime (static): %s (confidence %.0f%%, route=%s)",
                macro.regime, macro.confidence * 100, macro.suggested_route,
            )
        except ValueError as exc:
            raise AgentDataUnavailable(
                "Macro data not available locally either — skipping agent stage. "
                "Run `python data/fetch_macro.py` first."
            ) from exc

    route = decide_route(macro)
    return {"macro": macro, "route": route}


def valuation_node(state: AdvisorState) -> dict:
    """Read valuations (tool-driven, falling back to the snapshot-based path)."""
    from agents import valuation_agent

    profile = state["profile"]
    try:
        valuation = valuation_agent.run(profile)
        logger.info(
            "Valuation read (tool-driven): cheap=%s rich=%s (confidence %.0f%%)",
            valuation.cheap, valuation.rich, valuation.confidence * 100,
        )
    except Exception:
        logger.warning("Tool-driven valuation read failed — falling back to the snapshot-based path.")
        from data.snapshots import valuation_snapshot
        try:
            valuation = valuation_agent.run_static(valuation_snapshot(), profile)
            logger.info(
                "Valuation read (static): cheap=%s rich=%s (confidence %.0f%%)",
                valuation.cheap, valuation.rich, valuation.confidence * 100,
            )
        except ValueError as exc:
            raise AgentDataUnavailable(
                "Fundamentals data not available locally either — skipping agent stage. "
                "Run `python data/fetch_fundamentals.py` first."
            ) from exc

    return {"valuation": valuation}


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


def crisis_node(state: AdvisorState) -> dict:
    """Call crisis_agent.run (defensive tilts) with any critic feedback; bump the iteration counter."""
    from agents import crisis_agent

    proposal = crisis_agent.run(
        state["profile"],
        state["macro"],
        valuation=state.get("valuation"),
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
