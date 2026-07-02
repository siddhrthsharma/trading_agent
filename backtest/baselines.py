"""The core baselines every allocation must be compared against — no new math.

Thin wrappers over core/portfolios.py. This is the spec's "honest baseline is
sacred" rule made concrete: anything fancier must beat these after costs.
"""
from __future__ import annotations

from core.portfolios import all_equity, target_date, three_fund

# Representative horizon for the target-date baseline — matches the example
# profile (profile.example.json: horizon_years=30) since baselines are fixed
# comparison points, not tied to any single runtime profile.
_TARGET_DATE_HORIZON = 30

BASELINES: dict[str, dict[str, float]] = {
    "three_fund_moderate": three_fund("moderate"),   # 80% equity / 20% bonds
    "sixty_forty": three_fund("conservative"),       # exactly 60% equity / 40% bonds
    "target_date_30y": target_date(_TARGET_DATE_HORIZON),
    "all_equity": all_equity(),
}
