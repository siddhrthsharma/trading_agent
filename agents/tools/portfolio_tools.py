"""Tool functions the allocator/critic agents can call on demand (Phase 6).

Thin adapters over engine/ and core/ — no new math lives here.
"""
from __future__ import annotations

from utils.llm import ToolSpec


def drift_from_target(current_holdings: dict[str, float], target: dict[str, float]) -> dict:
    """Percentage-point drift per ticker between current dollar holdings and a target allocation."""
    total = sum(current_holdings.values())
    if total == 0:
        return {"error": "current_holdings sum to zero"}

    current_weights = {t: v / total for t, v in current_holdings.items()}
    all_tickers = set(current_weights) | set(target)
    drift = {t: current_weights.get(t, 0.0) - target.get(t, 0.0) for t in all_tickers}
    return {"drift": drift, "max_drift": max(abs(v) for v in drift.values())}


def check_allocation_rules(allocation: dict[str, float]) -> dict:
    """Run the deterministic guardrail checks (engine.allocation.check_rules) on an allocation."""
    from engine.allocation import check_rules

    violations = check_rules(allocation)
    return {
        "violations": [{"rule": v.rule, "severity": v.severity, "detail": v.detail} for v in violations],
        "passed": not any(v.severity == "serious" for v in violations),
    }


def project_growth(monthly_contribution: float, horizon_years: int, lump_sum: float = 0.0) -> dict:
    """Compound-growth projection (core.contributions.project_growth) for a given contribution plan."""
    from core.contributions import project_growth as _project_growth

    proj = _project_growth(
        monthly_contribution=monthly_contribution, horizon_years=horizon_years, lump_sum=lump_sum,
    )
    return {
        "final_balance": proj.final_balance,
        "total_contributed": proj.total_contributed,
        "total_growth": proj.total_growth,
    }


PORTFOLIO_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="drift_from_target",
        description="Compute percentage-point drift per ticker between current dollar holdings and a target allocation.",
        parameters={
            "type": "object",
            "properties": {
                "current_holdings": {"type": "object", "description": "Ticker -> dollar amount currently held"},
                "target": {"type": "object", "description": "Ticker -> target weight (0-1)"},
            },
            "required": ["current_holdings", "target"],
        },
        fn=drift_from_target,
    ),
    ToolSpec(
        name="check_allocation_rules",
        description="Run deterministic guardrail checks (duplicate index funds, QQQ/GLD concentration, position limits) on an allocation.",
        parameters={
            "type": "object",
            "properties": {
                "allocation": {"type": "object", "description": "Ticker -> weight (0-1), should sum to 1.0"},
            },
            "required": ["allocation"],
        },
        fn=check_allocation_rules,
    ),
    ToolSpec(
        name="project_growth",
        description="Project compound growth for a DCA + lump-sum contribution plan at the ~7% real long-run return.",
        parameters={
            "type": "object",
            "properties": {
                "monthly_contribution": {"type": "number"},
                "horizon_years": {"type": "integer"},
                "lump_sum": {"type": "number"},
            },
            "required": ["monthly_contribution", "horizon_years"],
        },
        fn=project_growth,
    ),
]
