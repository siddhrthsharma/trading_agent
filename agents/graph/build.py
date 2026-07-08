"""Wire the advisor graph: allocator -> critic, looping back on serious findings.

Phase 7 graph. Phase 9 extends this with macro/valuation/supervisor/crisis nodes
and routing edges -- this file stays small enough to read end to end.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.graph.nodes import allocator_node, critic_node
from agents.graph.state import AdvisorState


def _after_critic(state: AdvisorState) -> str:
    """Pure routing: done when the critic passes or the iteration budget is spent."""
    report = state["critic_report"]
    if report.passed or state["iteration"] >= state["max_iterations"]:
        return "done"
    return "revise"


def build_graph():
    """Compile the advisor StateGraph (Phase 7: allocator <-> critic cycle)."""
    builder = StateGraph(AdvisorState)
    builder.add_node("allocator", allocator_node)
    builder.add_node("critic", critic_node)
    builder.add_edge(START, "allocator")
    builder.add_edge("allocator", "critic")
    builder.add_conditional_edges("critic", _after_critic, {"revise": "allocator", "done": END})
    return builder.compile()
