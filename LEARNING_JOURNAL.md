# Learning Journal: Investment Allocation Advisor

> I update this whenever I build or change something.
> My goal is to understand every decision I make, not just accept the output.

## The Mental Model: Three Zones

Here is the core idea I am building around:

| Zone | Folders | Rule |
|---|---|---|
| Deterministic math | `core/`, `engine/` | Pure Python. No LLM, no network. Fully unit-tested. |
| Qualitative reasoning | `agents/` | LLM reads pre-computed numbers and gives opinions. Never does math. |
| Infrastructure | `data/`, `utils/`, `profile/` | Fetches data, handles secrets, and defines the investor. |

I split it this way because LLMs are good at language but bad at math. They make up numbers and cannot be unit-tested. Python does not have those problems. So Python handles all the calculations, and the LLM only ever sees finished numbers to reason about.

> The key thing I keep reminding myself: if the LLM ever outputs a percentage from thin air, that is a bug. Allocations come from `core/` and `engine/`. The LLM can adjust them or explain them, but it does not create them.

## The Full Pipeline (When `python main.py` Runs)

```
1.  Load investor profile            -> profile/investor_profile.py
2.  Emergency fund check             -> core/accounts.py      (HARD GATE: stop here if underfunded)
3.  Compute default allocation       -> core/portfolios.py    (usable on its own, no AI needed)
4.  DCA schedule + growth projection -> core/contributions.py

    ==== Phases 4-6 (now built) ====
5.  Fetch and cache macro data       -> data/fetch_macro.py (FRED API)
6.  Fetch ETF prices + fundamentals  -> data/fetch_prices.py (yfinance)
7.  Compute engine signals           -> engine/ (returns, Sharpe, optimizer)
8.  Macro agent: read the regime     -> agents/macro_agent.py (LLM)
9.  Valuation agent: cheap or rich?  -> agents/valuation_agent.py (LLM)
10. Allocator agent: propose plan    -> agents/allocator_agent.py (LLM, bounded by engine output)
11. Critic agent: stress-test it     -> agents/critic_agent.py (LLM, always runs)
12. Display result. I make the call.
```

I made the emergency fund check a hard gate at step 2. It is not optional. If I am underfunded, the program stops. It is the first investing principle built directly into the code.

## Phase 0: Environment Setup (Complete)

**What I built:** Python venv, `.env` for secrets, the project folder structure, a git repo, and `requirements.txt`.

**Technical patterns I learned:**
* `venv` keeps this project's packages separate from system Python. This matters because different projects often need different library versions.
* The `.env` + `python-dotenv` pattern puts secrets in a file that is gitignored. `utils/config.py` loads them at runtime using `get_settings()`. API keys never go in the code.
* `tenacity` is a retry decorator library. It wraps bad network calls so they automatically retry with a pause before giving up, instead of crashing right away.

**Financial context:** I use three free APIs in this project:
* **FRED** (Federal Reserve Economic Data): macro indicators like interest rates, CPI, unemployment, and the yield curve.
* **yfinance**: ETF price history from Yahoo Finance.
* **Groq**: free LLM inference using Llama 3.3 70B.

## Phase 1: The Core (Foundations) (Complete)

**What I built:** The `core/` folder has deterministic allocations, DCA math, and an emergency fund check. No AI is used here. This part alone is already a useful investing tool.

### `core/portfolios.py`: Default Allocations

I built three allocation strategies, all using three ETFs: **VTI** (US total market), **VXUS** (international), and **BND** (total bond market).

**Three-fund portfolio** (`three_fund(risk_tolerance)`)

This is called the Boglehead strategy, named after Vanguard founder John Bogle. I vary the equity and bond split based on risk tolerance:
* Conservative: 60% equity (40 VTI / 20 VXUS), 40% bonds
* Moderate: 80% equity, 20% bonds
* Aggressive: 90% equity, 10% bonds

**Glide path** (`glide_path(horizon_years)`)

I set the bond percentage based on `65 minus horizon_years`. The idea is that with 35 years until retirement I can ride out market drops, so I hold mostly stocks. With only 5 years left I cannot afford a bad year, so I hold more bonds. This is why target-date funds automatically shift as you get older.

**Why I picked these three ETFs:**
* VTI: roughly 4,000 US companies in one fund. Expense ratio is about 0.03%. It is basically buying the whole US stock market.
* VXUS: adds roughly 8,000 non-US companies. Geographic diversification lowers the risk tied to any one country.
* BND: roughly 10,000 US bonds (government and corporate). It tends to hold up or go up when stocks fall, which balances the portfolio in a crash.

**Technical pattern:** `_normalize()` divides each weight by the total so all weights always add up to exactly 1.0. Floating-point math can cause tiny rounding errors that would silently break the optimizer later, so this guards against that.

### `core/contributions.py`: DCA and Compound Growth

**Dollar-cost averaging (DCA):** You invest a fixed amount on a fixed schedule no matter what the market is doing. When prices are low, your fixed amount buys more shares. When prices are high, it buys fewer. Over time, your average cost per share ends up lower than the average price. You automatically buy more during dips without having to think about it.

**The growth projection formula:**

Future value of a lump sum: `FV = PV x (1 + r)^n`

Future value of monthly contributions (ordinary annuity):
`FV = PMT x [((1 + r)^n - 1) / r]`

Where `r` is the monthly rate = `(1 + annual_rate)^(1/12) - 1`. This is exact monthly compounding, not just `annual / 12`.

I default to a **7% real return**. That is the historical long-run equity return after subtracting inflation (roughly 10% nominal minus 3% inflation). I use this number because it is honest. Anything higher is wishful thinking.

**Why this matters:** $500 a month for 30 years at 7% real becomes about $567,000. You only put in $180,000. The other $387,000 comes from compounding. Starting 10 years earlier roughly doubles the final number. That is why time in the market is the first principle I keep front and center.

### `core/accounts.py`: Emergency Fund and Account Priority

**Emergency fund check:** I compute `months_covered = emergency_fund / monthly_expenses`. If it is less than 3, the program tells you to stop investing and build your cash buffer first. The reason is that if you invest while underfunded, you will be forced to sell stocks during a downturn just to pay for an emergency. That locks in losses at the worst possible time.

**Account priority:** Roth IRA, then 401k, then Traditional IRA, then taxable brokerage. Here is my reasoning:
* **Roth IRA**: You pay taxes now at your current rate, which is probably low. After that, all growth and withdrawals are tax-free. This is great when you are young and expect to be in a higher tax bracket later.
* **401k**: Contributions are pre-tax and lower your taxable income today. You pay taxes when you withdraw. Employer match is free money so I always take it first.
* **Taxable brokerage**: No tax advantages, but also no contribution limits. Long-term gains (held more than 1 year) are taxed much better than short-term gains.

**Capital gains logic** (`gains_treatment(holding_period_days)`): Over 365 days is long-term, taxed at 0%, 15%, or 20%. Under 365 days is short-term, taxed as ordinary income, which can be 22% to 37%. This is a real dollar difference and a big reason why low-turnover index investing beats active trading for most people even before counting fees.

## Phase 2: Education Layer (Status: TBD)

**What I will build:** `education/explainer_agent.py` will be my first real LLM usage. It will ask Groq to explain any financial term from a glossary in plain language.

**Why it matters to me:** This is a learning project. Being able to ask "what is a yield curve inversion?" or "explain the wash-sale rule" without leaving the code keeps the learning loop tight.

**The pattern I will follow:**
```python
from utils.llm import call_llm_json   # always via the shared interface, never import groq directly

def run(term: str) -> ExplainerOutput:
    prompt = f"Explain '{term}' for an investor who knows the basics..."
    data = call_llm_json(prompt, required_keys=["explanation", "example", "why_it_matters"])
    return ExplainerOutput(term=term, **data)
```

## Phase 3: Data Layer (Mostly Complete)

**What I built:** The `data/` folder fetches data from FRED and yfinance, stores everything with timestamps, and never overwrites history.

### `data/fetch_macro.py`: FRED Data

FRED is the Federal Reserve's public data API. The series I track:
* **Fed Funds Rate (FEDFUNDS):** The overnight lending rate the Fed controls. I think of this as the most important single number in macro because it affects mortgages, bond yields, corporate borrowing costs, and stock valuations.
* **10Y Treasury (DGS10) and 2Y Treasury (DGS2):** Government bond yields. The difference (10Y minus 2Y) is the yield curve.
* **Yield curve spread:** When the 2Y yield is higher than the 10Y yield (called an inversion), it has come before every US recession in the last 50 years. Not a perfect predictor, but worth tracking.
* **CPI (CPIAUCSL):** The Consumer Price Index. Measures inflation. Important for my real return calculations and for understanding Fed decisions.
* **Unemployment (UNRATE):** A lagging indicator. It goes up after recessions start. Together with CPI, it forms the Fed's "dual mandate."

**Technical patterns:**
* `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))` means if FRED's API is slow or broken, the code retries 3 times with longer waits (2s, 4s, 8s) before giving up. Exponential backoff is standard practice for API calls.
* Every row stores `fetched_at` (when I pulled it) and `date` (when the data point was published). This distinction is critical for backtesting.

### `data/store.py`: Why Timestamps Matter for Backtesting

Every data point gets two timestamps: `date` (when it happened) and `fetched_at` (when I pulled it). When I run a backtest, I follow one strict rule: when simulating a decision on date X, I only use data where `date <= X`. Using future data, even by accident, is called lookahead bias. It makes backtest results look amazing but then completely fall apart in real life. This is one of the most common ways trading strategies look good in testing but fail when used live.

### `data/fetch_prices.py`: ETF Price History

I download OHLCV (Open, High, Low, Close, Volume) data for each ETF using `yfinance`. I adjust automatically for stock splits and dividends so historical comparisons are accurate. Data is saved in Parquet format, which is a columnar compressed format that is much faster to read than CSV for time-series data.

## Phases 7, 8, 9: Status

Phases 4 through 9 are all done now (see the entries near the bottom of this journal for 7, 8, and 9). What's left: Phase 10 (persistent memory across runs, and the open-ended stretch goals — tax-aware asset location, Monte Carlo goal projection, rebalancing drift alerts, Groq → Claude).

## Key Python Patterns I Am Using

### `@dataclass`: Structured Return Values
```python
@dataclass
class GrowthProjection:
    final_balance: float
    total_contributed: float
    ...
```
Instead of returning a plain dict where I never know what keys exist, every function that produces structured output returns a typed dataclass. My editor can autocomplete `proj.final_balance` and a typo causes an error right away instead of silently breaking things.

### `from __future__ import annotations`: Deferred Type Hints
This lets me use type hints that reference types defined later in the file, or in older Python versions. I put it at the top of every file.

### `call_llm_json(prompt, required_keys=[...])`: Safe LLM Calls
The pattern in `utils/llm.py`: always ask for JSON, always check that specific keys are present, retry once on failure. LLMs sometimes return broken output. `required_keys` is a basic check that the data makes sense before it touches my code.

### `tenacity` Retries: Defensive API Calls
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_series(series_id): ...
```
Network calls fail sometimes. This decorator automatically retries with longer waits between attempts. `reraise=True` means if all retries fail, you see the original error instead of a confusing tenacity wrapper error.

## Financial Concepts Quick Reference

| Term | One-liner |
|---|---|
| **Expense ratio** | Annual percentage fee a fund charges. 0.03% (VTI) vs 1% (typical active fund) is a massive difference over decades. |
| **Sharpe ratio** | Return per unit of risk. Higher is better. `(portfolio_return - risk_free_rate) / volatility`. |
| **Max drawdown** | The biggest drop from a peak to a low point. A 60% max drawdown will really test how long you can hold on. |
| **CAGR** | Compound Annual Growth Rate. The one annualized number that summarizes total return. |
| **Sortino ratio** | Like Sharpe but only counts downside volatility as risk. Going up fast is not a problem. |
| **Yield curve inversion** | 2Y yield is higher than 10Y yield. The market is saying it expects things to slow down. Has predicted every recent US recession. |
| **Real return** | Nominal return minus inflation. 10% nominal minus 3% inflation = 7% real. This is what your actual buying power grows by. |
| **Lookahead bias** | Using future data in a backtest. Makes results look great on paper but impossible in real life. |
| **DCA** | Dollar-cost averaging. Fixed contribution on a fixed schedule. Takes emotion out of timing the market. |
| **Rebalancing** | Selling what grew too big and buying what shrank to get back to your target weights. Once a year is usually enough. |

## [Phase 3]: Centralized the Asset Universe in Config (2026-06-15)

**Files changed:** `utils/config.py`

**What I built:**
* I added module-level constants for the asset universe: `CORE_UNIVERSE` (the three-fund holdings), `FULL_UNIVERSE` (everything the optimizer can propose), `BENCHMARKS` (comparison-only tickers), `MUTUAL_FUND_EQUIVALENTS`, and `THREE_FUND`.
* I replaced the leftover default tickers (`AAPL, MSFT, NVDA, SPY`) with `_DEFAULT_TICKERS`, which comes from `FULL_UNIVERSE + BENCHMARKS` with duplicates removed.

**Why I did it this way:**
* The data pipeline was pulling from `settings.tickers`, which defaulted to individual stocks. That was out of sync with `core/portfolios.py`, which already used ETFs. Having one source of truth fixes that drift.
* I kept holdings and benchmarks in separate lists on purpose. Near-duplicate funds like VOO and VTI are about 80% overlapping. If both go into the optimizer, it will split weight between them for no real reason. Keeping them in `BENCHMARKS` lets me fetch their price history for comparison without letting them sneak into an allocation.

**Technical patterns:**
* `list(dict.fromkeys(...))` removes duplicates while keeping the original order. I find this cleaner than using a set, which scrambles the order.

**Financial concepts:**
* **Redundant exposure:** VTI, VOO, and IVV all hold basically the same stocks. Owning more than one just adds complexity without adding diversification. One broad fund per asset class is the whole point.
* **Wrapper vs index:** VOO (ETF) and VFIAX (mutual fund) both track the S&P 500 index but use different wrappers with different tax and liquidity properties. I use `MUTUAL_FUND_EQUIVALENTS` to note this without adding duplicates to the universe.

**How it connects:**
* `get_settings().tickers` now returns the right ETF universe, so the data fetchers (`fetch_prices`, `fetch_news`, `fetch_filings`) all pull the correct instruments. The `core/` folder and my future `engine/` and `agents/` folders all use the same list.

## Side Learning: LangChain and LangGraph (edX) (2026-06-28)

**Context:** I am also going through an edX course on LangChain and LangGraph at the same time as this project. This explains the gaps between journal entries. Some weeks I am doing coursework instead of building the next phase here.

**Why I am doing both:**
* This repo intentionally waits until Phase 4 to use LangGraph, starting with simple Python wiring first. The course is giving me the vocabulary and mental model before I wire up multi-agent flows here.
* LangChain and LangGraph are general frameworks for building AI agents. The core rule of this project still applies no matter what: math lives in `core/` and `engine/`, and the LLM only handles qualitative reasoning. The course teaches orchestration, not a reason to let the LLM invent numbers.

**How it connects:**
* When I get to Phases 4 through 6 (building out `agents/`, the critic, and the allocator), I will know whether plain Python wiring is enough or if LangGraph earns its place for more complex multi-step flows.
* `utils/llm.py` stays the single entry point for all LLM calls either way. The course shapes how I think about orchestration, not how individual agents talk to the model.

---

## [Phase 4] — Static agent pipeline: macro, valuation, allocator, critic — 2026-07-01

**Files changed:** `engine/signals.py`, `engine/allocation.py`, `data/fetch_fundamentals.py`, `data/store.py`, `data/snapshots.py`, `agents/macro_agent.py`, `agents/valuation_agent.py`, `agents/allocator_agent.py`, `agents/critic_agent.py`, `main.py`, tests

**What I built:** The first full agent pipeline. Four LLM agents run in a fixed order: macro reads the economic regime, valuation says what's cheap or rich, allocator proposes a plan, and critic stress-tests it. `engine/signals.py` turns raw FRED data into a clean `MacroSnapshot` (yield-curve spread, CPI year-over-year, real fed funds rate, unemployment). `engine/allocation.py` holds the important part: `apply_tilts()` and the deterministic rule checks. `data/snapshots.py` loads from storage and hands the engine clean DataFrames so the engine never touches disk. I also added a `fundamentals` table (P/E and dividend yield per ETF, append-only) so a later tool has history to compare against.

**Why this way:** The allocator never asks the LLM for a percentage — it asks for *tilts*, and Python turns tilts into weights. Every tilt is clamped to ±10 points, so even a hallucinated +90% tilt becomes +10% before renormalizing. This is how I let the LLM nudge an allocation without letting it invent one. The critic also runs its rule checks in Python first, then the LLM only critiques the reasoning — it never decides on its own whether something like QQQ concentration is a problem.

**Technical patterns:** Each agent has a private `_validate()` that range-checks the LLM output (like `0 <= confidence <= 1`), wrapped together with the LLM call inside a retry. That way a response that parses fine but has a bad value (like `recession_signal = 1.5`) gets retried instead of crashing.

**Financial concepts:** A diversified index fund can legitimately be 50-70% of a portfolio, so I raised the single-position warning from 50% to 75% (a failing test caught that the honest all-equity default, VTI 70% / VXUS 30%, was flagging itself). Concentration risk is about *narrow* bets (single stocks, one sector), not a total-market ETF doing its job.

**How it connects:** `main.py` chains snapshots → macro → valuation → allocator → critic, and skips gracefully if the API key or local data is missing. I spot-checked the macro agent at 2007-12-31, 2020-03-31, and 2022-09-30 and it gave sensible, different regime reads for each date.

---

## [Phase 5] — Deterministic engine + multi-regime backtester — 2026-07-01

**Files changed:** `engine/returns.py`, `engine/metrics.py`, `engine/costs.py`, `engine/optimizer.py`, `backtest/regimes.py`, `backtest/baselines.py`, `backtest/engine.py`, `main.py`, `requirements.txt`, tests

**What I built:** The quant core and a backtester. `engine/returns.py` computes daily returns, annualized return, volatility, and covariance. `engine/metrics.py` turns an equity curve into CAGR, Sharpe, Sortino, max drawdown, worst year, and days-to-recovery. `engine/costs.py` converts an annual expense ratio into a daily drag. `engine/optimizer.py` wraps PyPortfolioOpt for mean-variance plus a simple inverse-volatility option. `backtest/engine.py` simulates one allocation over one date window using share-based tracking (so weights actually *drift* between rebalances), with annual rebalancing, monthly contributions, and cost drag baked into each fund's price. `backtest/regimes.py` names four stress windows (2008, 2020, 2022, the 2010s bull) plus a held-out period.

**Why this way:** I bake costs into each ticker's price series *before* the simulation runs, not as an adjustment at the end. That matches how expense ratios really work (they continuously shave the fund's value), so rebalancing and contributions stack on top correctly without double-counting. For no-lookahead protection I used two independent layers: the loader already asserts internally, and the backtester adds its own `assert index.max() <= end` against the external end date. That second layer isn't busywork — my first attempt at it was accidentally always-true, so I rewrote it and wrote a test that feeds in fake future data to prove it actually catches leaks now.

**Technical patterns:** `compute_metrics()` gets CAGR from actual calendar days elapsed (`.days / 365.25`), not an assumed row frequency, so the same function handles both a monthly test fixture and a daily backtest. I also hand-computed expected results for the rebalance test instead of just checking "did it run" — matching my own arithmetic is what confirmed the share-tracking was really correct.

**Financial concepts:**
* **Max drawdown and recovery time** are the "can I actually hold this" numbers. A 46% drop that takes 2+ years to recover feels totally different from the same CAGR delivered smoothly.
* **Estimation error in mean-variance:** the max-Sharpe weights are very sensitive to tiny changes in the return estimate, which is why the optimizer output is only ever *context* for the allocator, never a target.
* **Survivorship/inception bias:** `VXUS` didn't exist until 2011, so a literal 2008 backtest would break. I added a `BACKTEST_PROXIES` map (VXUS → EFA) used only inside `backtest/`, and every substitution is logged, never silent — that keeps the 2008 stress test honest instead of quietly skipping it.

**How it connects:** `main.py` now also computes an unconstrained mean-variance read and passes it to the allocator as an optional hint — it changes the prompt context, never the tilt-clamping from Phase 4. Running `python -m backtest.engine` on real data matched known history: 2008 drawdowns of -47% to -57%, COVID's ~120-day V-shaped recovery, and 2022's negative return with no recovery in the window.

---

## [Phase 6] — Tool use: agents pull their own data — 2026-07-01

**Files changed:** `utils/llm.py`, `agents/tools/macro_tools.py`, `agents/tools/valuation_tools.py`, `agents/tools/portfolio_tools.py`, `agents/macro_agent.py`, `agents/valuation_agent.py`, `main.py`, tests

**What I built:** Agents can now fetch their own data instead of being handed a snapshot. `utils/llm.py` gained `ToolSpec` and `call_llm_with_tools()`, a loop that gives the model a toolset, runs whatever it calls, feeds the results back, and repeats until it answers with valid JSON. Three tool modules under `agents/tools/` cover macro (FRED series, yield curve, real rate), valuation (fundamentals, historical P/E), and portfolio (drift, rule checks, growth projection). The macro and valuation agents kept their Phase-4 logic renamed to `run_static()` and gained a new tool-driven `run()`.

**Why this way:** Every tool is a thin adapter — it does the I/O and calls into `engine/`/`core/` for any actual math, so the "no math in agents" rule still holds even though agents now decide what to fetch. `fetch_fred_series` sits behind a `FRED_ALLOWLIST` because it could technically request *any* series ID; the allowlist is the real limit on how far an agent can reach. `main.py` tries the tool-driven `run()` first and falls back to the snapshot-based `run_static()` if anything fails, so the Phase-4 pipeline becomes the safety net.

**Technical patterns:** A tool that throws doesn't crash the agent — `call_llm_with_tools` catches it and feeds an `{"error": ...}` message back so the model can react. I also learned the hard way that `ToolSpec` captures the function reference at import time, so patching the module-level function in a test does nothing; you have to patch the tools list itself with a replacement `ToolSpec`.

**Financial concepts:** None new this phase — it's purely an architecture shift (agents pulling data instead of receiving it) built on Phase 4's reasoning and Phase 5's engine.

**How it connects:** An offline test stubs the Groq client with a two-round script (round 1 calls `fetch_fred_series`, round 2 answers) and confirms the tool actually ran. A live `python main.py` run confirmed the same thing end-to-end, with sensibly lower confidence than the static path since the agent gathered less data.

---

## [Phase 7] — Critic ↔ allocator loop as a LangGraph cycle — 2026-07-08

**Files changed:** `agents/graph/__init__.py`, `agents/graph/state.py`, `agents/graph/nodes.py`, `agents/graph/build.py`, `agents/allocator_agent.py`, `main.py`, `requirements.txt`, `tests/test_graph.py`, `tests/test_agents.py`

**What I built:** A LangGraph `StateGraph` with an allocator node and a critic node. A conditional edge after the critic loops back to the allocator (feeding it the critic's serious findings) until the allocation passes `check_rules` or a `max_iterations` counter in the shared state runs out. `main.py`'s agent stage now builds an initial state and calls `build_graph().invoke(...)` instead of calling the allocator and critic directly.

**Why this way:** LangGraph lets me express "loop until the critic is satisfied" as a graph edge instead of a hand-written `while` loop. The core idea is a **shared state** object (here, a `TypedDict` called `AdvisorState`) that every node reads from and writes to — instead of passing return values manually between function calls, each node returns a partial update and LangGraph merges it in. A **node** is just a plain function `(state) -> dict`; the allocator node and critic node are thin wrappers that call the *exact same* `allocator_agent.run()` / `critic_agent.run()` I already had, so the agents stay independently testable outside the graph too. A **conditional edge** (`add_conditional_edges`) is a routing function that inspects state and returns which node name to go to next — that's how "loop back if serious, else stop" becomes a declared edge (`_after_critic`) instead of buried control flow. I capped iterations in the *state itself* (an `int` the allocator node increments) rather than trusting the LLM to know when to stop — two LLMs critiquing each other can disagree forever, so a hard ceiling plus "return best-so-far" is the safe default. The allocator still only ever proposes bounded tilts (nothing about `apply_tilts` or `MAX_TILT` changed), so looping can't accidentally open a door for the LLM to invent a percentage.

**Technical patterns:** `StateGraph(AdvisorState)` + `add_node`/`add_edge`/`add_conditional_edges` + `compile()` gives a runnable graph via `.invoke(initial_state)`. Nodes return partial dicts, not the full state — LangGraph does the merge, which is why `critic_node` can just return `{"critic_report": ..., "history": ..., "feedback": ...}` without re-stating `profile`, `macro`, etc. I also hit a real circular-import trap: `critic_agent.py` imports `AllocationProposal` from `allocator_agent.py`, so having `allocator_agent.py` import `CriticReport` back for a type hint would cycle. Fixed with `TYPE_CHECKING` — the import only happens for static type-checkers, never at runtime, since `from __future__ import annotations` already makes the annotation lazy.

**Financial concepts:** Concentration risk — the acceptance criterion asks for an "80% QQQ tilt" to trigger a revision, but that exact scenario can't happen through the real allocator (QQQ isn't in the tiltable ticker set, and tilts clamp at ±10%), which is itself proof the anti-invention mechanism works. So I proved the loop two ways: a *realistic* one (an aggressive all-equity baseline pushed to 80% VTI by a legal-but-too-large tilt, breaching the 75% single-position cap) and a *literal* one (a hand-built proposal with `{"QQQ": 0.8, "BND": 0.2}`, bypassing the allocator to test the critic-loop machinery directly against the spec's exact example). Both converge back toward the diversified baseline, not toward a different concentrated bet — the fix for concentration risk is always "come back toward the anchor."

**How it connects:** Replaces `main.py`'s single allocator-then-critic pass with a loop; `run_agent_stage` now returns the final `AdvisorState` (previously returned nothing) so Phase 8's dashboard can render the whole revision history without recomputing anything. A live `python main.py` run against the real Groq API converged in a single iteration with a clean critic pass, confirming the graph wiring works end-to-end, not just against mocks. The same graph gets a supervisor and crisis route bolted on in Phase 9 — `build.py` was written short enough on purpose that the Phase 9 diff should be almost entirely additions.

---

## [Phase 8] — Streamlit dashboard — 2026-07-08

**Files changed:** `dashboard/app.py`, `requirements.txt`

**What I built:** A one-file Streamlit dashboard (`streamlit run dashboard/app.py`) with seven sections: profile + emergency-fund gate, the agent stage (critic notes and the Phase 7 revision history, behind a "Run agents" button so the LLM only gets called when I ask), default vs. proposed allocation, the glide path, the DCA growth projection, the multi-regime backtest suite vs. baselines, and a final baseline comparison table.

**Why this way:** The dashboard is explicitly a *view*, not a second implementation. Every number on the page comes from calling an existing function — `core.portfolios.default_for`, `core.contributions.project_growth`, `core.accounts.emergency_fund_status`, `backtest.engine.run_regime_suite`, or (for anything LLM-driven) `main.run_agent_stage`, which is the same Phase-7 code path `python main.py` uses. If I'd reimplemented any of that logic in `dashboard/app.py` instead of calling it, the dashboard and the CLI could quietly drift apart and show two different "truths." I put the agent stage behind a button rather than running it automatically on page load: Streamlit reruns the *entire script* top-to-bottom on every interaction, so without a button an LLM call (with real cost and latency) would fire on every single widget click anywhere on the page, not just the ones related to the agent stage.

**Technical patterns:** `@st.cache_data` memoizes pure, deterministic calls (the profile load, the glide-path table, the backtest suite) so Streamlit's rerun-on-every-interaction model doesn't redo expensive work like the multi-regime backtest on every click. `st.session_state` is different — it's a dict that survives across reruns *within one browser session*, which is what lets the "Run agents" button's result stick around after the page reruns for an unrelated reason (I use it, not `@st.cache_data`, for the agent stage because a `dataclass`-based `AdvisorState` isn't a hashable cache key and because I *want* a fresh graph run each time I click the button, not a cached one). To verify the dashboard actually renders without a browser, I used Streamlit's own `streamlit.testing.v1.AppTest`, which executes the script headlessly and lets me assert on `at.exception`, `at.header`, `at.info`, etc. — much more reliable than `curl`ing the page, since Streamlit's real UI is a single-page app that only appears after a websocket connects.

**Financial concepts:** None new — Phase 8 is presentation, not new financial logic. The one thing worth internalizing from *looking* at the rendered backtest table: the honest baselines (three-fund, 60/40, target-date, all-equity) are the yardstick, and if the LLM-tilted proposal doesn't clearly beat them after costs in a given regime, the dashboard shows that plainly instead of hiding it — that's the "the honest baseline is sacred" rule made visible, not just enforced in code.

**How it connects:** Reuses `agents/graph/build.py` and `main.run_agent_stage` from Phase 7 with zero duplication — the dashboard is proof that the graph's final state (`AdvisorState`) is a genuinely useful, self-contained object for a caller other than `main.py`. Verified twice with `AppTest`: once with a real `GROQ_API_KEY` (all 7 sections render, backtest suite runs against real local price history with the VXUS→EFA 2008 proxy substitution logging correctly), and once with `.env` temporarily removed (confirmed restored afterward) to prove the "GROQ_API_KEY not set" info message and graceful skip actually fire when the key is genuinely absent, not just unset in the shell — `utils.config.get_settings()` is `lru_cache`d and `load_dotenv()`s from the `.env` file directly, so a shell-only `unset` doesn't fully simulate a missing key.

---

## [Phase 9] — Supervisor planning, crisis agent, and dynamic routing on the advisor graph — 2026-07-08

**Files changed:** `agents/crisis_agent.py`, `agents/supervisor.py`, `agents/critic_agent.py`, `agents/graph/nodes.py`, `agents/graph/build.py`, `agents/graph/state.py`, `main.py`, `dashboard/app.py`, `tests/test_graph.py`, `tests/test_agents.py`

**What I built:** The Phase 7 graph gets a real front half. A `supervisor` node plans the steps for a goal (a qualitative LLM call, purely advisory — it never blocks the pipeline if it fails). A `macro` node reads the regime and, critically, makes the routing call: it calls the new `supervisor.decide_route(macro)` — a **pure Python function**, not an LLM call — which returns `"crisis"` if `recession_signal > 0.8` or the macro agent already said `suggested_route == "crisis"`. A conditional edge sends `"crisis"` reads straight to a new `crisis_agent` (skipping `valuation` entirely) and everything else through the existing `valuation → allocator` path. Both branches converge on the same `critic` node, and the critic's revise-loop now sends a rejected proposal back to *whichever* node produced it (`f"revise_{state['route']}"`), not always the allocator.

**Why this way:** The spec draws a line between "the supervisor plans and routes" and I split that literally into two different kinds of code: `supervisor.run()` is an LLM call (qualitative, "what steps make sense for this goal"), while `supervisor.decide_route()` is a plain `if` statement over a typed `MacroAssessment` field. Routing decisions are exactly the kind of thing that must stay deterministic and testable — I don't want "should I go defensive" to depend on an LLM's mood on a given call, and a pure function meant I could unit-test the 0.8 threshold directly (`test_decide_route_threshold`) with no mocking at all. The crisis agent is a near-duplicate of the allocator (same `_REQUIRED_KEYS`, same clamp-and-filter mechanism, same `AllocationProposal` return type) — I mirrored the structure on purpose rather than trying to share more code, because the two prompts genuinely say different things (defensive tilts toward BND/BIL vs. normal tilts), and the critic/graph machinery needs to treat their outputs identically regardless of which one produced them.

**Technical patterns:** This is the first genuinely *conditional* piece of the graph — Phase 7's only branch was "loop or stop"; Phase 9 adds "which of two specialists handles this." The trick that keeps `build.py` readable is that the macro node writes `route` into state itself (`{"macro": macro, "route": decide_route(macro)}`), so the conditional-edge functions (`_route_after_macro`, `_after_critic`) stay one-line pure reads of `state["route"]` — no routing logic duplicated between the node and the edge. I also added a small `AgentDataUnavailable` exception in `nodes.py`: previously this fallback logic (tool-driven read fails → try the snapshot-based path → give up with a warning) lived inline in `main.py`; moving it into `macro_node`/`valuation_node` meant `main.py` needed a way to hear "truly give up" from deep inside a compiled graph, and a custom exception was the cleanest way to do that without smuggling a sentinel value through state. `critic_agent.run` and its prompt builder now accept `valuation: ValuationAssessment | None` — on the crisis branch, `valuation` is `None` because that node never ran, so the prompt just says "(skipped on the crisis path)" instead of crashing on `None.cheap`.

**Financial concepts:** Capital preservation / defensive allocation — when recession risk is genuinely high, the right move isn't panic-selling everything, it's a *bounded* lean toward short-duration Treasuries and cash-equivalents (BND, BIL) while keeping the diversified core intact. That's why the crisis agent still tilts within ±10 points of the same baseline instead of getting a separate "crisis allocation" universe — the honest baseline is still the anchor even when the environment gets scary, which is the same "sacred baseline" principle from Phase 1 showing up again in a new place.

**How it connects:** Every existing Phase 7 test had to be updated, not just extended — since the graph's entry point moved from `START → allocator` to `START → supervisor → macro → ...`, tests that used to hand-build an `AdvisorState` with `macro`/`valuation` already filled in now have to patch `agents.macro_agent.run` / `agents.valuation_agent.run` / `agents.supervisor.run` so the new front-half nodes don't make real LLM calls. That's a good sign, not a smell: it means the graph is a single real pipeline, not two disconnected halves bolted together. A live `python main.py` run showed the whole chain working end-to-end — supervisor plan logged, macro read `early-cycle accommodative` with `recession_signal` low enough to route `normal`, and the dashboard (clicked through via `AppTest`, not just imported) rendered the new "Route: normal" banner and the supervisor's plan in an expander. `pytest tests/` is 207 tests green, including two new routing tests (`test_graph_routes_to_crisis_on_high_recession_signal`, `test_graph_routes_to_normal_on_low_signal`) that each assert the *other* branch's agent was never called at all — proof the routing actually skips work, not just that it produces the right-looking numbers.
