"""DCA schedule and compound-growth projection — deterministic, no LLM."""
from __future__ import annotations

from dataclasses import dataclass, field

# Documented long-run real return: ~7% per year (CURSOR_PROJECT_SPEC.md §First Principles)
_DEFAULT_ANNUAL_REAL_RETURN = 0.07


@dataclass
class Contribution:
    """A single DCA contribution event."""
    month: int        # 1-based month index from start
    amount: float


@dataclass
class GrowthProjection:
    """Result of a compound-growth projection."""
    final_balance: float
    total_contributed: float    # sum of all DCA contributions (excludes lump sum)
    total_growth: float         # final_balance - (lump_sum + total_contributed)
    lump_sum: float
    annual_real_return: float
    horizon_years: int
    yearly_balances: list[float] = field(default_factory=list)  # balance at end of each year


def dca_schedule(monthly_contribution: float, horizon_years: int) -> list[Contribution]:
    """Return a list of monthly fixed contributions over the full horizon."""
    n_months = horizon_years * 12
    return [Contribution(month=m, amount=monthly_contribution) for m in range(1, n_months + 1)]


def project_growth(
    monthly_contribution: float,
    horizon_years: int,
    lump_sum: float = 0.0,
    annual_real_return: float = _DEFAULT_ANNUAL_REAL_RETURN,
) -> GrowthProjection:
    """Project portfolio growth with a lump-sum plus monthly DCA contributions.

    Uses exact monthly compounding: monthly_rate = (1 + annual_rate)^(1/12) - 1.
    Returns figures in today's dollars when annual_real_return ≈ 0.07.
    """
    if horizon_years == 0:
        return GrowthProjection(
            final_balance=lump_sum,
            total_contributed=0.0,
            total_growth=0.0,
            lump_sum=lump_sum,
            annual_real_return=annual_real_return,
            horizon_years=0,
            yearly_balances=[lump_sum],
        )

    monthly_rate = (1 + annual_real_return) ** (1 / 12) - 1
    n_months = horizon_years * 12

    # Lump-sum future value
    lump_fv = lump_sum * (1 + monthly_rate) ** n_months

    # Ordinary annuity future value (contributions at END of each period)
    if monthly_rate == 0:
        annuity_fv = monthly_contribution * n_months
    else:
        annuity_fv = monthly_contribution * (((1 + monthly_rate) ** n_months - 1) / monthly_rate)

    final_balance = lump_fv + annuity_fv
    total_contributed = monthly_contribution * n_months
    total_growth = final_balance - (lump_sum + total_contributed)

    # Per-year balance series (for charting in later phases)
    yearly_balances: list[float] = []
    balance = lump_sum
    for year in range(1, horizon_years + 1):
        months_this_year = 12
        balance = balance * (1 + monthly_rate) ** months_this_year
        if monthly_rate == 0:
            balance += monthly_contribution * months_this_year
        else:
            balance += monthly_contribution * (((1 + monthly_rate) ** months_this_year - 1) / monthly_rate)
        yearly_balances.append(balance)

    return GrowthProjection(
        final_balance=final_balance,
        total_contributed=total_contributed,
        total_growth=total_growth,
        lump_sum=lump_sum,
        annual_real_return=annual_real_return,
        horizon_years=horizon_years,
        yearly_balances=yearly_balances,
    )
