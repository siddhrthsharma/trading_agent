from __future__ import annotations

from dataclasses import dataclass

from profile.investor_profile import InvestorProfile

# Account types listed in priority order (tax-advantaged first)
_PRIORITY_ORDER = ["Roth IRA", "401k", "Traditional IRA", "taxable"]


@dataclass
class EmergencyFundStatus:
    """Result of the emergency-fund check — always run this first."""
    is_funded: bool          # True iff emergency_fund >= 3 * monthly_expenses
    months_covered: float    # emergency_fund / monthly_expenses (inf if expenses == 0)
    target_low: float        # 3 * monthly_expenses
    target_high: float       # 6 * monthly_expenses
    shortfall: float         # max(0, target_low - emergency_fund)


def emergency_fund_status(profile: InvestorProfile) -> EmergencyFundStatus:
    """Compute emergency-fund coverage relative to the 3–6 month target."""
    if profile.monthly_expenses == 0:
        return EmergencyFundStatus(
            is_funded=True,
            months_covered=float("inf"),
            target_low=0.0,
            target_high=0.0,
            shortfall=0.0,
        )

    target_low = 3.0 * profile.monthly_expenses
    target_high = 6.0 * profile.monthly_expenses
    months_covered = profile.emergency_fund / profile.monthly_expenses
    is_funded = profile.emergency_fund >= target_low
    shortfall = max(0.0, target_low - profile.emergency_fund)

    return EmergencyFundStatus(
        is_funded=is_funded,
        months_covered=months_covered,
        target_low=target_low,
        target_high=target_high,
        shortfall=shortfall,
    )


def account_priority(profile: InvestorProfile) -> list[str]:
    """Return the investor's accounts sorted tax-advantaged first, then taxable.

    Only accounts present in the profile are included; unknown types are
    appended at the end in the order they appear in the profile.
    """
    accounts = list(profile.accounts)
    known = [a for a in _PRIORITY_ORDER if a in accounts]
    unknown = [a for a in accounts if a not in _PRIORITY_ORDER]
    return known + unknown


def gains_treatment(holding_period_days: int) -> str:
    """Return 'long-term' (held >365 days) or 'short-term'.

    Long-term gains are taxed at 0% / 15% / 20% (lowest brackets pay 0%).
    Only *realized* gains (from a sale) trigger a tax event.
    """
    return "long-term" if holding_period_days > 365 else "short-term"
