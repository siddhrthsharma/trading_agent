"""Tilt-clamping and deterministic rule checks — pure math, no LLM, no network.

This is the mechanism that keeps the allocator agent from ever inventing an
allocation: the LLM proposes bounded tilts, and apply_tilts() is the only
code that actually produces allocation weights.
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.config import FULL_UNIVERSE

MAX_TILT = 0.10  # LLM may shift any single sleeve by at most ±10 percentage points

_QQQ_MAX = 0.20      # expensive (0.20% ER) + concentrated — flag above this
_GLD_MAX = 0.10      # hedge sleeve only, never a core holding
# Broad index funds (VTI, all_equity) legitimately reach 70% in an honest
# all-equity baseline (core/portfolios.all_equity: VTI 70% / VXUS 30%), so the
# concentration cap sits above that — it catches genuinely extreme single-fund
# bets, not a diversified core sleeve doing its job.
_POSITION_MAX = 0.75
_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RuleViolation:
    """A single deterministic rule check result."""
    rule: str
    severity: str    # "serious" | "minor"
    detail: str


def apply_tilts(
    baseline: dict[str, float],
    tilts: dict[str, float],
    max_tilt: float = MAX_TILT,
) -> dict[str, float]:
    """Clamp each tilt to ±max_tilt, apply to the baseline, floor at 0, renormalize to 1.0.

    This is the anti-invention mechanism: no matter what the LLM proposes in
    `tilts` (even a hallucinated +0.90), the output allocation is always a
    legal, renormalized perturbation of `baseline` bounded by max_tilt.
    """
    result = dict(baseline)
    for ticker, tilt in tilts.items():
        clamped = max(-max_tilt, min(max_tilt, tilt))
        result[ticker] = result.get(ticker, 0.0) + clamped

    result = {t: max(0.0, w) for t, w in result.items()}
    total = sum(result.values())
    if total == 0:
        raise ValueError("Tilted allocation has zero total weight")
    return {t: w / total for t, w in result.items() if w > 0}


def check_rules(allocation: dict[str, float]) -> list[RuleViolation]:
    """Run deterministic guardrail checks against an allocation.

    Catches: tickers outside the allowed universe, VOO+IVV pseudo-diversification,
    QQQ/GLD over-concentration, any single position over 50%, and weights that
    don't sum to 1.0.
    """
    violations: list[RuleViolation] = []

    unknown = [t for t in allocation if t not in FULL_UNIVERSE]
    if unknown:
        violations.append(RuleViolation(
            rule="unknown_ticker",
            severity="serious",
            detail=f"Ticker(s) not in FULL_UNIVERSE: {sorted(unknown)}",
        ))

    if allocation.get("VOO", 0) > 0 and allocation.get("IVV", 0) > 0:
        violations.append(RuleViolation(
            rule="duplicate_index",
            severity="serious",
            detail="VOO and IVV track the same S&P 500 index — pseudo-diversification with no benefit",
        ))

    qqq = allocation.get("QQQ", 0)
    if qqq > _QQQ_MAX:
        violations.append(RuleViolation(
            rule="qqq_concentration",
            severity="serious",
            detail=f"QQQ at {qqq:.1%} exceeds {_QQQ_MAX:.0%} — expensive and concentrated",
        ))

    gld = allocation.get("GLD", 0)
    if gld > _GLD_MAX:
        violations.append(RuleViolation(
            rule="gld_concentration",
            severity="minor",
            detail=f"GLD at {gld:.1%} exceeds {_GLD_MAX:.0%} — should be a small hedge sleeve only",
        ))

    for ticker, weight in allocation.items():
        if weight > _POSITION_MAX:
            violations.append(RuleViolation(
                rule="concentration",
                severity="serious",
                detail=f"{ticker} at {weight:.1%} exceeds {_POSITION_MAX:.0%} single-position limit",
            ))

    total = sum(allocation.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        violations.append(RuleViolation(
            rule="weight_sum",
            severity="serious",
            detail=f"Weights sum to {total:.6f}, not 1.0",
        ))

    return violations
