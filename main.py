from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)


def run_advisor() -> None:
    """Run the investment allocation advisor pipeline.

    Emergency-fund check always runs first. If underfunded, the advisor
    returns immediately with a cash-buffer recommendation — full stop.
    """
    from profile.investor_profile import load_profile
    from core import accounts, portfolios, contributions

    profile = load_profile()

    # ── 1. Emergency-fund gate (FIRST — always) ───────────────────────────
    ef = accounts.emergency_fund_status(profile)
    if not ef.is_funded:
        logger.info("=" * 60)
        logger.info("ACTION REQUIRED: Build your cash buffer first — full stop.")
        logger.info(
            "You have %.1f months of expenses covered (target: 3–6 months).",
            ef.months_covered,
        )
        logger.info(
            "Shortfall: $%.2f  (need $%.2f, have $%.2f)",
            ef.shortfall,
            ef.target_low,
            profile.emergency_fund,
        )
        logger.info(
            "Invest nothing further until your emergency fund reaches $%.2f.",
            ef.target_low,
        )
        logger.info("=" * 60)
        return

    # ── 2. Account priority ───────────────────────────────────────────────
    priority = accounts.account_priority(profile)
    logger.info("Account priority: %s", " → ".join(priority))

    # ── 3. Default allocation ─────────────────────────────────────────────
    allocation = portfolios.default_for(profile)
    logger.info("Default allocation:")
    for ticker, weight in sorted(allocation.items(), key=lambda x: -x[1]):
        logger.info("  %-6s  %5.1f%%", ticker, weight * 100)

    # ── 4. DCA schedule summary ──────────────────────────────────────────
    schedule = contributions.dca_schedule(profile.monthly_contribution, profile.horizon_years)
    logger.info(
        "DCA: $%.0f/month for %d years → %d contributions",
        profile.monthly_contribution,
        profile.horizon_years,
        len(schedule),
    )

    # ── 5. Growth projection ─────────────────────────────────────────────
    proj = contributions.project_growth(
        monthly_contribution=profile.monthly_contribution,
        horizon_years=profile.horizon_years,
        lump_sum=profile.lump_sum,
    )
    logger.info(
        "Growth projection (~7%% real): $%.0f invested → $%.0f in %d years "
        "(+$%.0f growth, today's dollars)",
        proj.total_contributed + profile.lump_sum,
        proj.final_balance,
        profile.horizon_years,
        proj.total_growth,
    )

    # ── 6+. Agent stage (Phase 4) — qualitative reasoning over engine numbers ──
    run_agent_stage(profile, allocation)


def run_agent_stage(profile, baseline_allocation, goal: str = "Allocate long-term investments for this profile"):
    """Supervisor → macro → (crisis | valuation → allocator) → critic, as a LangGraph (Phase 9).

    The macro node pulls its own data via tools when GROQ_API_KEY is set (falling
    back to the Phase-4 snapshot-based path if that fails), then routes to the
    defensive crisis specialist when the recession signal is high, or the normal
    valuation → allocator path otherwise. Both branches loop through the critic
    until it passes or `max_iterations` is hit. Skips the whole stage (logging a
    warning) if there's no API key, or if neither the tool-driven nor
    snapshot-based macro/valuation read is available.

    Returns the final AdvisorState (see agents/graph/state.py) or None if skipped.
    """
    from utils.config import get_settings

    if not get_settings().groq_api_key:
        logger.warning("GROQ_API_KEY not set — skipping agent stage (core allocation above still stands).")
        return None

    from agents.graph.build import build_graph
    from agents.graph.nodes import AgentDataUnavailable
    from agents.graph.state import initial_state

    optimizer_hint = _try_optimizer_hint()
    state = initial_state(profile, baseline_allocation, optimizer_hint=optimizer_hint, goal=goal)

    try:
        final_state = build_graph().invoke(state)
    except AgentDataUnavailable as exc:
        logger.warning(str(exc))
        return None

    proposal = final_state["proposal"]
    critic = final_state["critic_report"]

    logger.info("=" * 60)
    logger.info("Route: %s | Revision iterations: %d", final_state["route"], final_state["iteration"])
    if final_state.get("plan") is not None:
        logger.info("Supervisor plan: %s", final_state["plan"].steps)
    logger.info("Baseline vs. tilted proposal:")
    all_tickers = sorted(set(proposal.baseline) | set(proposal.allocation), key=lambda t: -proposal.allocation.get(t, 0))
    for ticker in all_tickers:
        base_w = proposal.baseline.get(ticker, 0.0)
        prop_w = proposal.allocation.get(ticker, 0.0)
        logger.info("  %-6s  baseline %5.1f%%  →  proposal %5.1f%%", ticker, base_w * 100, prop_w * 100)
    logger.info("Rationale: %s", proposal.rationale)

    logger.info("=" * 60)
    logger.info("Critic: passed=%s severity=%s", critic.passed, critic.severity)
    for v in critic.violations:
        logger.info("  [%s] %s: %s", v.severity, v.rule, v.detail)
    for issue in critic.qualitative_issues:
        logger.info("  - %s", issue)
    logger.info("Critic reasoning: %s", critic.reasoning)
    logger.info("=" * 60)

    if len(final_state["history"]) > 1:
        logger.info("Revision history:")
        for entry in final_state["history"]:
            logger.info(
                "  iter %d: passed=%s severity=%s tilts=%s",
                entry["iteration"], entry["passed"], entry["severity"], entry["tilts"],
            )
        logger.info("=" * 60)

    return final_state


def _try_optimizer_hint():
    """Compute an unconstrained mean-variance read over FULL_UNIVERSE for the allocator's
    context (Phase 5+). Returns None if there isn't enough local price history yet —
    the optimizer is informational only, never required."""
    from data.snapshots import load_close_matrix
    from engine.optimizer import mean_variance
    from utils.config import FULL_UNIVERSE

    try:
        close = load_close_matrix(FULL_UNIVERSE)
        return mean_variance(close, bounds=(0.0, 0.60))
    except Exception:
        logger.warning("Optimizer context unavailable (insufficient price history) — proceeding without it.")
        return None


if __name__ == "__main__":
    run_advisor()
