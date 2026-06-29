# Learning Journal — Investment Allocation Advisor

> I update this whenever I execute a plan.
> My goal: understand every design decision I make, not just accept the output.

---

## The Mental Model: Three Zones

The core idea I'm building around:

| Zone | Folders | Rule |
|---|---|---|
| Deterministic math | `core/`, `engine/` | Uses pure python. No LLM, no network. Fully unit-tested. |
| Qualitative reasoning | `agents/` | LLM reads pre-computed numbers and gives opinions. Never does math. |
| Infrastructure | `data/`, `utils/`, `profile/` | Fetches data, handles config/secrets, defines the investor. |

**Why I split it this way:** LLMs are statistically great at language and synthesis, but they aren't good at arithmetic — they hallucinate numbers, lose precision, and can't be unit-tested. Python has none of those problems. So I let Python own all calculation, and the LLM only ever receives finished numbers to reason about.

> The key insight I keep coming back to: if the LLM ever outputs a percentage from thin air, I treat that as a bug. Allocations come from `core/` and `engine/`; the LLM can adjust them within bounds or explain them, but it doesn't create them.

---

## The Full Pipeline (When `python main.py` Runs)

```
1.  Load investor profile            → profile/investor_profile.py
2.  Emergency fund check             → core/accounts.py      ← HARD GATE: stop here if underfunded
3.  Compute default allocation       → core/portfolios.py    ← usable on its own, no AI needed
4.  DCA schedule + growth projection → core/contributions.py

    ──── Phases 4+ (not built yet) ────
5.  Fetch + cache macro data         → data/fetch_macro.py (FRED API)
6.  Fetch ETF prices + fundamentals  → data/fetch_prices.py (yfinance)
7.  Compute engine signals           → engine/ (returns, Sharpe, optimizer)
8.  Macro agent: read the regime     → agents/macro_agent.py (LLM)
9.  Valuation agent: cheap/rich?     → agents/valuation_agent.py (LLM)
10. Allocator agent: propose plan    → agents/allocator_agent.py (LLM, bounded by engine output)
11. Critic agent: stress-test it     → agents/critic_agent.py (LLM, always runs)
12. Display result — I make the call
```

I made the emergency fund check at step 2 a hard gate, not an optional choice — it's the first investment principle encoded directly into the flow. The system literally won't proceed past it if I'm underfunded.

---

## Phase 0 — Environment Setup ✅ Complete

**What I built:** Python venv, `.env` for secrets, project folder structure, git repo, `requirements.txt`.

**Technical patterns I learned:**
- `venv` isolates this project's packages from system Python — I realized this matters when different projects need different library versions
- `.env` + `python-dotenv` pattern: secrets live in a file that's gitignored; `utils/config.py` loads them at runtime via `get_settings()`. I never hardcode API keys.
- `tenacity` is a retry decorator library — it wraps bad network calls so they automatically retry with exponential backoff instead of crashing on a timeout

**Financial context:** I'm currently using three APIs in this project, all free:
- **FRED** (Federal Reserve Economic Data) — macro indicators: interest rates, CPI, unemployment, yield curve
- **yfinance** — ETF price history, pulled from Yahoo Finance
- **Groq** — free-tier LLM inference using Llama 3.3 70B

---

## Phase 1 — The Core (Foundations) Complete

**What I built:** `core/` — deterministic allocations, DCA math, emergency fund check. It doesn't use any AI. This alone is already a pretty useful investing tool.

### `core/portfolios.py` — Default Allocations

I built three allocation strategies, all around three ETFs: **VTI** (US total market), **VXUS** (international), **BND** (total bond market).

**Three-fund portfolio** (`three_fund(risk_tolerance)`)
The Boglehead classic, named after Vanguard founder John Bogle's philosophy of low-cost index investing. I vary the equity/bond split by risk tolerance:
- Conservative: 60% equity (40 VTI / 20 VXUS), 40% bonds
- Moderate: 80% equity, 20% bonds
- Aggressive: 90% equity, 10% bonds

**Glide path** (`glide_path(horizon_years)`)
I scale bond% with an age proxy (`65 - horizon_years`). The idea: with 35 years left I can ride out volatility, so I hold mostly equity; with 5 years left I can't afford a crash, so I hold more bonds. This is why target-date funds automatically shift allocation as you age.

**Why I picked these three ETFs specifically:**
- VTI: ~4,000 US companies in one fund. Expense ratio ~0.03%. I'm buying "the whole US economy."
- VXUS: Adds ~8,000 non-US companies. Geographic diversification reduces country-specific risk.
- BND: ~10,000 US bonds (government + corporate). Negative correlation to equities in most crashes — it's the ballast.

**Technical pattern:** `_normalize()` divides each weight by the total. I added this to guard against weights not summing to exactly 1.0 due to floating-point arithmetic, which could silently break the optimizer later.

---

### `core/contributions.py` — DCA + Compound Growth

**Dollar-cost averaging (DCA):** Invest a fixed amount on a fixed schedule regardless of market conditions. The math behind why this works, and what I find elegant about it: when prices are low my fixed amount buys more shares; when high, fewer. Over time, my average cost per share is lower than the average price — I mechanically buy more at dips.

**The growth projection formula:**

The future value of a lump sum: `FV = PV × (1 + r)^n`

The future value of a recurring contribution (ordinary annuity):
`FV = PMT × [((1 + r)^n - 1) / r]`

Where `r` is the monthly rate = `(1 + annual_rate)^(1/12) - 1` (exact monthly compounding, not `annual/12`).

I default to a **7% real return** — the historical long-run equity return after inflation (~10% nominal minus ~3% inflation). I treat this as the honest figure; anything higher is optimistic.

**Why this matters to me:** a $500/month contribution over 30 years at 7% real = ~$567,000. Total contributed: $180,000. The remaining ~$387,000 is pure compounding. I noticed that starting 10 years earlier roughly doubles the final balance — this is why "time in market" is the first principle I keep front and center.

---

### `core/accounts.py` — Emergency Fund + Account Priority

**Emergency fund check:** I compute `months_covered = emergency_fund / monthly_expenses`. If < 3, it returns a hard recommendation to stop investing and build cash first. I built it this way because investing while underfunded means I'd be forced to sell equities during a downturn to cover an emergency — locking in losses at exactly the wrong time.

**Account priority:** Roth IRA → 401k → Traditional IRA → taxable brokerage. My reasoning:
- **Roth IRA**: contributions taxed now (at my current low rate), growth and withdrawals are tax-free. Best for young people who expect to be in a higher bracket later.
- **401k**: pre-tax contributions lower my taxable income today; taxed on withdrawal. Employer match is free money — I always capture it first.
- **Taxable brokerage**: no tax advantages, but no contribution limits. Gains taxed at capital gains rates — long-term (> 1 year) is taxed much more favorably than short-term.

**Capital gains logic** (`gains_treatment(holding_period_days)`): > 365 days = long-term = 0%/15%/20% rate. ≤ 365 days = short-term = ordinary income rate (can be 22–37%). I realized this is a real dollar difference and a core reason why low-turnover index investing outperforms active trading for most people even before fees.

---

## Phase 2 — Education Layer (Status: TBD)

**What I'll build:** `education/explainer_agent.py` — my first real LLM usage. It asks Groq to explain any financial term from a curated glossary in plain language.

**Why it matters to me:** This is a learning project. Being able to ask "what's a yield curve inversion?" or "explain the wash-sale rule" without leaving the codebase keeps my educational loop tight.

**The pattern I'll follow:**
```python
from utils.llm import call_llm_json   # always via the shared interface, never import groq directly

def run(term: str) -> ExplainerOutput:
    prompt = f"Explain '{term}' for an investor who knows the basics..."
    data = call_llm_json(prompt, required_keys=["explanation", "example", "why_it_matters"])
    return ExplainerOutput(term=term, **data)
```

---

## Phase 3 — Data Layer Mostly Complete

**What I built:** `data/` — fetches from FRED and yfinance, stores everything with timestamps, never overwrites history.

### `data/fetch_macro.py` — FRED Data

FRED is the Federal Reserve's public data API. The series I track:
- **Fed Funds Rate (FEDFUNDS):** The overnight lending rate the Fed sets. I consider this the most important single number in macro — it propagates through mortgages, bond yields, corporate borrowing costs, and equity valuations.
- **10Y Treasury (DGS10) and 2Y Treasury (DGS2):** Government bond yields. The spread (10Y − 2Y) is the yield curve.
- **Yield curve spread:** When the 2Y yield > 10Y yield (inversion), it has historically preceded every US recession in the last 50 years. Not a perfect signal, but I find it hard to ignore.
- **CPI (CPIAUCSL):** Consumer Price Index. Measures inflation. Relevant to my real return calculations and the Fed's rate decisions.
- **Unemployment (UNRATE):** Lagging indicator — rises after recessions begin. Combined with CPI it forms the "dual mandate" the Fed is targeting.

**Technical patterns:**
- `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))` — if FRED's API is slow or returns a 503, this retries 3 times with increasing wait (2s, 4s, 8s) before giving up. I learned exponential backoff is standard practice for API clients.
- Every row stores `fetched_at` (when I pulled it) and `date` (when the data point was published). I realized this distinction matters enormously for backtesting — see next point.

### `data/store.py` — Why Timestamps Matter for Backtesting

I stamp every data point with both `date` (when it happened) and `fetched_at` (when I pulled it). During backtesting, I hold myself to a strict rule: **when simulating a decision on date X, only use data where `date <= X`**. Using future data — even accidentally — is lookahead bias, and it produces backtest results that are impossible to replicate live. I learned this is one of the most common ways quant strategies fail out-of-sample.

### `data/fetch_prices.py` — ETF Price History

I download OHLCV (Open, High, Low, Close, Volume) for each ETF in the universe using `yfinance`. I auto-adjust for splits and dividends so my historical comparisons are accurate. I store in Parquet format (columnar, compressed, much faster than CSV for time-series queries).

---

## Phases 4–8 — Not Yet Built

| Phase | What | Key concepts I'll encounter |
|---|---|---|
| **4** | Macro regime agent | Yield curve interpretation, real rates, LLM confidence scoring |
| **5** | Portfolio engine + backtester | Mean-variance optimization, Sharpe ratio, max drawdown, no-lookahead assertion |
| **6** | Multi-agent allocation | Allocator + critic pattern, agent orchestration in `main.py` |
| **7** | Streamlit dashboard | Data visualization, glide path charts, scenario tables |
| **8** | Stretch: tax-aware location, Monte Carlo | Asset location theory, probability distributions for goal planning |

---

## Key Python Patterns I'm Using in This Project

### `@dataclass` — Structured Return Values
```python
@dataclass
class GrowthProjection:
    final_balance: float
    total_contributed: float
    ...
```
Instead of returning a plain dict (where I can't know what keys exist), every function that produces structured output returns a typed dataclass. This means my editor can autocomplete `proj.final_balance` and a typo raises an error immediately.

### `from __future__ import annotations` — Deferred Type Hints
Allows me to use type hints that reference types defined later in the file, or in Python versions before 3.10. I put it at the top of every file.

### `call_llm_json(prompt, required_keys=[...])` — Safe LLM Calls
The pattern I use in `utils/llm.py`: always ask for JSON, always validate that specific keys are present, retry once on failure. I need this because LLMs occasionally produce malformed output. `required_keys` acts as a minimal schema check before the data hits my code.

### `tenacity` Retries — Defensive API Calls
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_series(series_id): ...
```
Network calls fail. This decorator wraps the function so failures trigger automatic retries with exponential backoff. `reraise=True` means if all retries fail, the original exception surfaces rather than a tenacity wrapper error.

---

## Financial Concepts Quick Reference

| Term | One-liner |
|---|---|
| **Expense ratio** | Annual % fee a fund charges. 0.03% (VTI) vs 1% (typical active fund) = massive long-run difference. |
| **Sharpe ratio** | Return per unit of risk. Higher is better. `(portfolio_return - risk_free_rate) / volatility`. |
| **Max drawdown** | Largest peak-to-trough loss. A portfolio with -60% max drawdown will test your psychology hard. |
| **CAGR** | Compound Annual Growth Rate. The single annualized return number. |
| **Sortino ratio** | Like Sharpe but only penalizes downside volatility (upside volatility isn't bad). |
| **Yield curve inversion** | 2Y yield > 10Y yield. Signals market expects future rate cuts (i.e., slowdown). Recession predictor. |
| **Real return** | Nominal return minus inflation. 10% nominal − 3% inflation = 7% real. What your purchasing power actually grows by. |
| **Lookahead bias** | Using future data in a backtest. Makes results look better than they'd ever be live. Cardinal sin. |
| **DCA** | Dollar-cost averaging. Fixed contribution on a fixed schedule. Removes the emotion from timing. |
| **Rebalancing** | Selling winners and buying laggards to restore target weights. Low-churn = usually once/year is enough. |

---

## [Phase 3] — Centralized the asset universe in config — 2026-06-15

**Files changed:** `utils/config.py`

**What I built:**
- I added module-level constants for the asset universe: `CORE_UNIVERSE` (the three-fund holdings), `FULL_UNIVERSE` (everything the optimizer may propose), `BENCHMARKS` (comparison-only tickers), `MUTUAL_FUND_EQUIVALENTS`, and `THREE_FUND`.
- I replaced the leftover trading-era default tickers (`AAPL, MSFT, NVDA, SPY`) with `_DEFAULT_TICKERS`, derived from `FULL_UNIVERSE + BENCHMARKS` (deduped, order-preserving).

**Why I did it this way:**
- I noticed the data pipeline pulled from `settings.tickers`, which defaulted to individual stocks — out of sync with `core/portfolios.py`, which already anchored on ETFs (VTI/VXUS/BND). A single source of truth removes that drift.
- I deliberately kept holdings and benchmarks in separate lists. Near-duplicate funds (VOO/IVV vs VTI, AGG vs BND) would corrupt a mean-variance optimizer — it arbitrarily splits weight between near-identical assets. Keeping them in `BENCHMARKS` lets me fetch their price history for comparison without ever letting them leak into an allocation proposal.

**Technical patterns:**
- `list(dict.fromkeys(...))` to dedupe while preserving order (I find this cleaner than a set, which loses order).

**Financial concepts:**
- **Redundant exposure:** I realized VTI, VOO, and IVV are ~80%+ overlapping; holding several of them adds complexity without diversification. One broad fund per asset class is the point.
- **Wrapper vs index:** VOO (ETF) and VFIAX (mutual fund) hold the same S&P 500 index — different tax/liquidity wrappers, same exposure. I use `MUTUAL_FUND_EQUIVALENTS` to record this without duplicating universe entries.

**How it connects:**
- `get_settings().tickers` now returns the ETF universe, so `data/pipeline.py` and the fetchers (`fetch_prices`, `fetch_news`, `fetch_filings`) pull the right instruments by default. Downstream `core/` and my future `engine/`/`agents/` consume the same canonical lists.

---

## Side learning — LangChain & LangGraph (edX) — 2026-06-28

**Context:** I'm also working through an edX course on LangChain and LangGraph in parallel with this project. That explains gaps between journal entries — some weeks go to agent orchestration coursework instead of the next phase here.

**Why I'm doing both:**
- This repo deliberately defers LangGraph until Phase 4+ (simple orchestration in `main.py` first). The course gives me vocabulary and patterns before I wire multi-agent flows here.
- LangChain/LangGraph are general agent frameworks; this project's rule (math in `core/`/`engine/`, LLM only for qualitative reasoning) still applies — the course is infrastructure learning, not a reason to let the LLM invent allocations.

**How it connects:**
- When Phase 4–6 land (`agents/`, critic/allocator orchestration), I'll know whether plain Python wiring is enough or LangGraph earns its place for stateful multi-step runs.
- `utils/llm.py` stays the single LLM entry point either way; course material informs *orchestration*, not provider imports inside agents.
