"""Streamlit dashboard — a VIEW over core/, engine/, backtest/, and the Phase 7 graph.

Never recomputes math or duplicates pipeline logic: every number here is read
from core/portfolios.py, core/contributions.py, core/accounts.py,
backtest/engine.py, or the compiled advisor graph's final state.

Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from core import accounts, contributions, portfolios
from profile.investor_profile import load_profile
from utils.config import get_settings

st.set_page_config(page_title="Investment Allocation Advisor", layout="wide")


@st.cache_data
def _cached_profile():
    return load_profile()


@st.cache_data
def _cached_glide_path(max_horizon: int = 40) -> pd.DataFrame:
    """Equity% by horizon — repeatedly calling core.portfolios.glide_path is a view, not new math."""
    rows = [{"horizon_years": h, "equity_pct": portfolios.glide_path(h)["VTI"] + portfolios.glide_path(h)["VXUS"]}
            for h in range(1, max_horizon + 1)]
    return pd.DataFrame(rows).set_index("horizon_years")


@st.cache_data
def _cached_regime_suite():
    from backtest.engine import run_regime_suite
    return run_regime_suite()


def _allocation_bar_df(baseline: dict[str, float], proposal: dict[str, float] | None) -> pd.DataFrame:
    tickers = sorted(set(baseline) | set(proposal or {}), key=lambda t: -baseline.get(t, 0))
    data = {"baseline": [baseline.get(t, 0.0) * 100 for t in tickers]}
    if proposal is not None:
        data["proposal"] = [proposal.get(t, 0.0) * 100 for t in tickers]
    return pd.DataFrame(data, index=tickers)


def render_profile_and_emergency_fund(profile) -> bool:
    """Returns True iff the investor is clear to proceed past the emergency-fund gate."""
    st.header("1. Profile & Emergency Fund")
    cols = st.columns(4)
    cols[0].metric("Horizon", f"{profile.horizon_years}y")
    cols[1].metric("Risk tolerance", profile.risk_tolerance)
    cols[2].metric("Monthly contribution", f"${profile.monthly_contribution:,.0f}")
    cols[3].metric("Lump sum", f"${profile.lump_sum:,.0f}")

    ef = accounts.emergency_fund_status(profile)
    if not ef.is_funded:
        st.error(
            f"**ACTION REQUIRED: Build your cash buffer first — full stop.**\n\n"
            f"You have {ef.months_covered:.1f} months of expenses covered (target: 3–6 months). "
            f"Shortfall: ${ef.shortfall:,.2f} (need ${ef.target_low:,.2f}, have ${profile.emergency_fund:,.2f}). "
            f"Invest nothing further until your emergency fund reaches ${ef.target_low:,.2f}."
        )
        return False

    st.success(f"Emergency fund funded: {ef.months_covered:.1f} months of expenses covered.")
    st.caption("Account priority: " + " → ".join(accounts.account_priority(profile)))
    return True


def render_allocation(profile, baseline: dict[str, float], proposal: dict[str, float] | None) -> None:
    st.header("3. Default vs. Proposed Allocation")
    df = _allocation_bar_df(baseline, proposal)
    st.bar_chart(df)
    st.dataframe(df.style.format("{:.1f}%"), width="stretch")


def render_glide_path(profile) -> None:
    st.header("4. Glide Path")
    st.caption("Equity % (VTI + VXUS) as a function of investing horizon — from core/portfolios.glide_path.")
    df = _cached_glide_path()
    st.line_chart(df * 100)
    st.caption(f"Your horizon ({profile.horizon_years}y) implies roughly "
               f"{df.loc[min(profile.horizon_years, df.index.max()), 'equity_pct'] * 100:.0f}% equity under the glide path.")


def render_dca_projection(profile) -> None:
    st.header("5. DCA Growth Projection")
    proj = contributions.project_growth(
        monthly_contribution=profile.monthly_contribution,
        horizon_years=profile.horizon_years,
        lump_sum=profile.lump_sum,
    )
    cols = st.columns(3)
    cols[0].metric("Total invested", f"${proj.total_contributed + proj.lump_sum:,.0f}")
    cols[1].metric("Projected balance", f"${proj.final_balance:,.0f}")
    cols[2].metric("Growth from compounding", f"${proj.total_growth:,.0f}")
    st.caption(f"Assumes {proj.annual_real_return:.0%} annual real return over {proj.horizon_years} years (today's dollars).")

    if proj.yearly_balances:
        chart_df = pd.DataFrame({"balance": proj.yearly_balances}, index=range(1, len(proj.yearly_balances) + 1))
        chart_df.index.name = "year"
        st.area_chart(chart_df)


def render_backtest_suite() -> None:
    st.header("6. Backtest Across Regimes vs. Baselines")
    try:
        df = _cached_regime_suite()
    except Exception as exc:
        st.info(f"Backtest unavailable (insufficient local price history). Run `python data/fetch_prices.py` first.\n\nDetails: {exc}")
        return

    if df.empty:
        st.info("Backtest returned no results. Run `python data/fetch_prices.py` first.")
        return

    display = df.copy()
    for pct_col in ("cagr", "sharpe", "sortino", "max_drawdown", "worst_year", "cost_drag"):
        if pct_col in display.columns:
            display[pct_col] = display[pct_col]
    st.dataframe(display, width="stretch")
    st.caption(
        "Honest baseline comparison: if a tilted proposal doesn't beat these cheap, diversified "
        "baselines after costs, that's the expected, correct outcome to surface — not a failure to hide."
    )


def render_agent_stage(profile, baseline: dict[str, float]):
    st.header("2. Agent Stage: Critic Notes & Revision History")

    if not get_settings().groq_api_key:
        st.info("GROQ_API_KEY not set — agent stage skipped (core allocation above still stands).")
        return None

    if st.button("Run agents (macro → valuation → allocator ↔ critic)"):
        import main as main_module
        with st.spinner("Running the advisor graph..."):
            st.session_state["advisor_state"] = main_module.run_agent_stage(profile, baseline)

    final_state = st.session_state.get("advisor_state")
    if final_state is None:
        st.caption("Click the button above to run the agent stage (calls the LLM).")
        return None

    macro = final_state["macro"]
    valuation = final_state["valuation"]
    proposal = final_state["proposal"]
    critic = final_state["critic_report"]
    plan = final_state.get("plan")

    route = final_state["route"]
    if route == "crisis":
        st.warning(f"**Route: crisis** — defensive specialist engaged (recession signal {macro.recession_signal:.2f}).")
    else:
        st.info(f"**Route: normal** (recession signal {macro.recession_signal:.2f}).")
    if plan is not None:
        with st.expander(f"Supervisor plan (confidence {plan.confidence:.0%})"):
            for step in plan.steps:
                st.markdown(f"- {step}")
            st.caption(plan.rationale)

    st.subheader("Macro & Valuation Reads")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**Regime:** {macro.regime} (confidence {macro.confidence:.0%}, route={macro.suggested_route})")
        st.caption(macro.reasoning)
    with cols[1]:
        if valuation is not None:
            st.markdown(f"**Cheap:** {valuation.cheap} · **Rich:** {valuation.rich} (confidence {valuation.confidence:.0%})")
            st.caption(valuation.reasoning)
        else:
            st.caption("Valuation read skipped on the crisis path — capital preservation takes priority.")

    st.subheader("Final Critic Report")
    if critic.passed:
        st.success(f"Passed (severity={critic.severity}, confidence {critic.confidence:.0%})")
    else:
        st.warning(f"Did not pass (severity={critic.severity}, confidence {critic.confidence:.0%})")
    for v in critic.violations:
        st.markdown(f"- **[{v.severity}] {v.rule}**: {v.detail}")
    for issue in critic.qualitative_issues:
        st.markdown(f"- {issue}")
    st.caption(critic.reasoning)

    st.subheader(f"Revision History ({len(final_state['history'])} iteration(s))")
    for entry in final_state["history"]:
        label = f"Iteration {entry['iteration']} — {'passed' if entry['passed'] else 'revised (' + entry['severity'] + ')'}"
        with st.expander(label):
            st.markdown(f"**Tilts:** {entry['tilts']}")
            st.markdown(f"**Rationale:** {entry['rationale']}")
            if entry["violations"]:
                st.markdown("**Violations:**")
                for v in entry["violations"]:
                    st.markdown(f"- {v}")
            if entry["qualitative_issues"]:
                st.markdown("**Qualitative issues:**")
                for issue in entry["qualitative_issues"]:
                    st.markdown(f"- {issue}")

    return final_state


def render_baseline_comparison(proposal: dict[str, float] | None) -> None:
    st.header("7. Baseline Comparison")
    from backtest.baselines import BASELINES

    tickers = sorted({t for alloc in BASELINES.values() for t in alloc} | set(proposal or {}))
    data = {name: [alloc.get(t, 0.0) * 100 for t in tickers] for name, alloc in BASELINES.items()}
    if proposal is not None:
        data["your_proposal"] = [proposal.get(t, 0.0) * 100 for t in tickers]
    df = pd.DataFrame(data, index=tickers)
    st.dataframe(df.style.format("{:.1f}%"), width="stretch")


def main() -> None:
    st.title("Investment Allocation Advisor")
    st.caption("Not financial advice — a personal research and education tool. You are always the final decision-maker.")

    profile = _cached_profile()
    proceed = render_profile_and_emergency_fund(profile)
    if not proceed:
        return

    baseline = portfolios.default_for(profile)
    final_state = render_agent_stage(profile, baseline)
    proposal = final_state["proposal"].allocation if final_state else None

    render_allocation(profile, baseline, proposal)
    render_glide_path(profile)
    render_dca_projection(profile)
    render_backtest_suite()
    render_baseline_comparison(proposal)


if __name__ == "__main__":
    main()
