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

    ==== Phases 4+ (not built yet) ====
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

## Phases 4 to 8: Not Yet Built

| Phase | What | Key concepts I will encounter |
|---|---|---|
| **4** | Macro regime agent | Yield curve interpretation, real rates, LLM confidence scoring |
| **5** | Portfolio engine + backtester | Mean-variance optimization, Sharpe ratio, max drawdown, no-lookahead rule |
| **6** | Multi-agent allocation | Allocator and critic pattern, agent orchestration in `main.py` |
| **7** | Streamlit dashboard | Data visualization, glide path charts, scenario tables |
| **8** | Stretch: tax-aware location, Monte Carlo | Asset location theory, probability distributions for goal planning |

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
