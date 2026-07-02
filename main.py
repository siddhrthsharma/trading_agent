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


def run_agent_stage(profile, baseline_allocation) -> None:
    """Macro → valuation → allocator → critic (Phase 4 sequence; Phase 6 tool use).

    Macro and valuation agents now pull their own data via tools when
    GROQ_API_KEY is set (Phase 6's `run(profile)`); if data is thin or the
    tool-driven read fails, this falls back to the Phase-4 snapshot-based
    `run_static(...)` path. Skips the whole stage (logging a warning) only if
    there's no API key at all — the core Phase-1 pipeline above already
    produced a usable result without any LLM calls.
    """
    from utils.config import get_settings

    if not get_settings().groq_api_key:
        logger.warning("GROQ_API_KEY not set — skipping agent stage (core allocation above still stands).")
        return

    from agents import allocator_agent, critic_agent, macro_agent, valuation_agent

    try:
        macro = macro_agent.run(profile)
        logger.info("Macro regime (tool-driven): %s (confidence %.0f%%, route=%s)", macro.regime, macro.confidence * 100, macro.suggested_route)
    except Exception:
        logger.warning("Tool-driven macro read failed — falling back to the snapshot-based path.")
        from data.snapshots import macro_snapshot
        try:
            macro = macro_agent.run_static(macro_snapshot(), profile)
            logger.info("Macro regime (static): %s (confidence %.0f%%, route=%s)", macro.regime, macro.confidence * 100, macro.suggested_route)
        except ValueError:
            logger.warning(
                "Macro data not available locally either — skipping agent stage. "
                "Run `python data/fetch_macro.py` first."
            )
            return

    try:
        valuation = valuation_agent.run(profile)
        logger.info("Valuation read (tool-driven): cheap=%s rich=%s (confidence %.0f%%)", valuation.cheap, valuation.rich, valuation.confidence * 100)
    except Exception:
        logger.warning("Tool-driven valuation read failed — falling back to the snapshot-based path.")
        from data.snapshots import valuation_snapshot
        try:
            valuation = valuation_agent.run_static(valuation_snapshot(), profile)
            logger.info("Valuation read (static): cheap=%s rich=%s (confidence %.0f%%)", valuation.cheap, valuation.rich, valuation.confidence * 100)
        except ValueError:
            logger.warning(
                "Fundamentals data not available locally either — skipping agent stage. "
                "Run `python data/fetch_fundamentals.py` first."
            )
            return

    optimizer_hint = _try_optimizer_hint()
    proposal = allocator_agent.run(profile, macro, valuation, optimizer_hint)
    logger.info("=" * 60)
    logger.info("Baseline vs. tilted proposal:")
    all_tickers = sorted(set(proposal.baseline) | set(proposal.allocation), key=lambda t: -proposal.allocation.get(t, 0))
    for ticker in all_tickers:
        base_w = proposal.baseline.get(ticker, 0.0)
        prop_w = proposal.allocation.get(ticker, 0.0)
        logger.info("  %-6s  baseline %5.1f%%  →  proposal %5.1f%%", ticker, base_w * 100, prop_w * 100)
    logger.info("Rationale: %s", proposal.rationale)

    critic = critic_agent.run(proposal, macro, valuation, profile)
    logger.info("=" * 60)
    logger.info("Critic: passed=%s severity=%s", critic.passed, critic.severity)
    for v in critic.violations:
        logger.info("  [%s] %s: %s", v.severity, v.rule, v.detail)
    for issue in critic.qualitative_issues:
        logger.info("  - %s", issue)
    logger.info("Critic reasoning: %s", critic.reasoning)
    logger.info("=" * 60)


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
