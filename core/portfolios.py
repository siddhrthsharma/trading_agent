"""Canonical default allocations — deterministic, no LLM, no network."""
from __future__ import annotations

from profile.investor_profile import InvestorProfile, load_profile

# Allocation: ticker → weight; weights must sum to 1.0 ± 1e-9.
Allocation = dict[str, float]

# Asset universe (low-cost, liquid, broad — from CURSOR_PROJECT_SPEC.md)
_US_EQUITY = "VTI"
_INTL_EQUITY = "VXUS"
_TOTAL_BOND = "BND"
_SHORT_TREASURY = "BIL"


def _normalize(alloc: dict[str, float]) -> Allocation:
    """Ensure weights sum to exactly 1.0 by scaling."""
    total = sum(alloc.values())
    if total == 0:
        raise ValueError("Allocation has zero total weight")
    return {k: v / total for k, v in alloc.items() if v > 0}


def three_fund(risk_tolerance: str) -> Allocation:
    """Classic Boglehead three-fund portfolio: US + international + bonds.

    Equity/bond split varies by risk tolerance.
    """
    splits = {
        "conservative": (0.40, 0.20, 0.40),  # 60% equity, 40% bond
        "moderate":     (0.54, 0.26, 0.20),  # 80% equity, 20% bond
        "aggressive":   (0.60, 0.30, 0.10),  # 90% equity, 10% bond
    }
    us, intl, bond = splits.get(risk_tolerance, splits["moderate"])
    return _normalize({_US_EQUITY: us, _INTL_EQUITY: intl, _TOTAL_BOND: bond})


def glide_path(horizon_years: int) -> Allocation:
    """Equity% scales up with horizon; bond% scales down.

    Derived from horizon_years directly (no age field needed).
    Rule of thumb: assume retirement at ~65, so estimated age ≈ 65 - horizon_years,
    and bond% ≈ age.  Clamped to equity [20%, 90%] to avoid extremes.
    """
    # bond% ≈ 65 - horizon_years (longer horizon → younger investor → fewer bonds)
    raw_bond_pct = 65 - horizon_years
    bond_pct = max(10, min(80, raw_bond_pct))   # clamp: equity stays in [20%, 90%]
    equity_pct = 100 - bond_pct
    # Split equity 70/30 US/intl
    us = equity_pct * 0.70 / 100
    intl = equity_pct * 0.30 / 100
    bond = bond_pct / 100
    return _normalize({_US_EQUITY: us, _INTL_EQUITY: intl, _TOTAL_BOND: bond})


def target_date(horizon_years: int) -> Allocation:
    """Pick a preset allocation based on horizon alone (like a target-date fund)."""
    if horizon_years >= 30:
        return _normalize({_US_EQUITY: 0.54, _INTL_EQUITY: 0.26, _TOTAL_BOND: 0.20})
    elif horizon_years >= 20:
        return _normalize({_US_EQUITY: 0.48, _INTL_EQUITY: 0.22, _TOTAL_BOND: 0.30})
    elif horizon_years >= 10:
        return _normalize({_US_EQUITY: 0.36, _INTL_EQUITY: 0.14, _TOTAL_BOND: 0.50})
    elif horizon_years >= 5:
        return _normalize({_US_EQUITY: 0.24, _INTL_EQUITY: 0.06, _TOTAL_BOND: 0.70})
    else:
        # Near-term: capital preservation
        return _normalize({_US_EQUITY: 0.10, _TOTAL_BOND: 0.50, _SHORT_TREASURY: 0.40})


def all_equity() -> Allocation:
    """100% equity split for very long horizons / aggressive investors."""
    return _normalize({_US_EQUITY: 0.70, _INTL_EQUITY: 0.30})


def default_for(profile: InvestorProfile) -> Allocation:
    """Select the most appropriate default allocation for this investor."""
    h = profile.horizon_years
    rt = profile.risk_tolerance

    if rt == "aggressive" and h >= 20:
        return all_equity()
    if rt == "conservative":
        return three_fund(rt)
    # moderate or long-horizon aggressive: use glide path
    return glide_path(h)


if __name__ == "__main__":
    from utils.logger import get_logger
    logger = get_logger(__name__)
    profile = load_profile()
    alloc = default_for(profile)
    logger.info("Default allocation for horizon=%dy risk=%s:", profile.horizon_years, profile.risk_tolerance)
    for ticker, w in sorted(alloc.items(), key=lambda x: -x[1]):
        logger.info("  %-6s  %5.1f%%", ticker, w * 100)
    logger.info("Weight sum: %.10f", sum(alloc.values()))
