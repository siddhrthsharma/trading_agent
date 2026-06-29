from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_VALID_RISK_TOLERANCES = {"conservative", "moderate", "aggressive"}


@dataclass
class InvestorProfile:
    horizon_years: int
    risk_tolerance: str                # "conservative" | "moderate" | "aggressive"
    monthly_contribution: float
    lump_sum: float                    # current investable cash (AFTER emergency fund)
    monthly_expenses: float
    emergency_fund: float              # current cash buffer — checked BEFORE investing
    accounts: list[str] = field(default_factory=list)     # ["Roth IRA", "401k", "taxable"]
    constraints: list[str] = field(default_factory=list)  # e.g. ["no individual stocks"]
    current_holdings: dict = field(default_factory=dict)  # {"VTI": 5000, "cash": 2000}


def _validate(data: dict) -> None:
    """Raise ValueError if profile data fails basic sanity checks."""
    rt = data.get("risk_tolerance", "")
    if rt not in _VALID_RISK_TOLERANCES:
        raise ValueError(
            f"risk_tolerance must be one of {sorted(_VALID_RISK_TOLERANCES)}, got {rt!r}"
        )
    for key in ("monthly_contribution", "lump_sum", "monthly_expenses", "emergency_fund"):
        val = data.get(key, 0)
        if val < 0:
            raise ValueError(f"{key} must be non-negative, got {val}")
    if data.get("horizon_years", 0) < 0:
        raise ValueError(f"horizon_years must be non-negative, got {data.get('horizon_years')}")


def load_profile(path: str | Path = "profile.json") -> InvestorProfile:
    """Load an InvestorProfile from JSON; fall back to profile.example.json if absent."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    if not p.exists():
        p = _PROJECT_ROOT / "profile.example.json"

    data = json.loads(p.read_text())
    _validate(data)

    return InvestorProfile(
        horizon_years=int(data["horizon_years"]),
        risk_tolerance=data["risk_tolerance"],
        monthly_contribution=float(data["monthly_contribution"]),
        lump_sum=float(data["lump_sum"]),
        monthly_expenses=float(data["monthly_expenses"]),
        emergency_fund=float(data["emergency_fund"]),
        accounts=list(data.get("accounts", [])),
        constraints=list(data.get("constraints", [])),
        current_holdings=dict(data.get("current_holdings", {})),
    )
