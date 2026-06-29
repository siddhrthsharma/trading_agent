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


if __name__ == "__main__":
    run_advisor()
