"""Wire the advisor graph: supervisor -> macro -> (crisis | valuation -> allocator) -> critic.

Phase 7 built the allocator <-> critic cycle. Phase 9 adds the supervisor,
macro, valuation, and crisis nodes plus two conditional edges (routing after
macro, and routing the revision loop back to whichever proposer produced the
current proposal) -- this file stays small enough to read end to end.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.graph.nodes import (
    allocator_node,
    critic_node,
    crisis_node,
    macro_node,
    supervisor_node,
    valuation_node,
)
from agents.graph.state import AdvisorState


def _route_after_macro(state: AdvisorState) -> str:
    """Pure routing: the macro node already decided normal vs. crisis and recorded it."""
    return state["route"]


def _after_critic(state: AdvisorState) -> str:
    """Done when the critic passes or the iteration budget is spent; else revise on the same branch."""
    report = state["critic_report"]
    if report.passed or state["iteration"] >= state["max_iterations"]:
        return "done"
    return f"revise_{state['route']}"


def build_graph():
    """Compile the advisor StateGraph."""
    builder = StateGraph(AdvisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("macro", macro_node)
    builder.add_node("valuation", valuation_node)
    builder.add_node("allocator", allocator_node)
    builder.add_node("crisis", crisis_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "macro")
    builder.add_conditional_edges("macro", _route_after_macro, {"crisis": "crisis", "normal": "valuation"})
    builder.add_edge("valuation", "allocator")
    builder.add_edge("allocator", "critic")
    builder.add_edge("crisis", "critic")
    builder.add_conditional_edges(
        "critic", _after_critic, {"revise_normal": "allocator", "revise_crisis": "crisis", "done": END}
    )
    return builder.compile()
