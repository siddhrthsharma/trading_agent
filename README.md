# Portfolio Pilot (trading_agent)

An agentic AI investment allocation assistant, written in plain Python. You ask it a question about your portfolio, and instead of sending everything to one giant prompt, it hands the work to specialized LLM agents: one for market research, one for risk assessment, one for asset allocation.

Everything runs on Llama 3.3 70B through the Groq API.

## How it works

- **Separate agents for separate jobs.** Market research, risk assessment, and asset allocation each get their own prompt and their own tools.
- **Tool calling.** Agents call tools to pull data instead of guessing from memory.
- **Schema-validated JSON.** Each agent returns structured JSON that gets checked against a schema before the next agent sees it. If an agent returns garbage, it gets caught early instead of breaking things downstream.
- **Backtesting.** There's a backtest module so allocations can be checked against historical data instead of just sounding reasonable.

## Status

Work in progress. Right now the agents run in a fixed sequence. I'm moving to LangGraph with conditional routing, so a query only goes to the agents it actually needs.

## Repo layout

| Folder | What's in it |
| --- | --- |
| `agents/` | The LLM agents (research, risk, allocation) |
| `core/` | Shared core logic |
| `engine/` | Orchestration that runs the agents |
| `backtest/` | Backtesting |
| `data/` | Market data and data loading |
| `dashboard/` | Dashboard for viewing results |
| `education/` | Educational content |
| `profile/` | User profile handling |
| `utils/` | Helpers |
| `tests/` | Tests |

`main.py` is the entry point. `LEARNING_JOURNAL.md` is my running notes on what I learned building this.

## Setup

```bash
git clone https://github.com/siddhrthsharma/trading_agent.git
cd trading_agent
pip install -r requirements.txt
```

You'll need a Groq API key. Set it as an environment variable:

```bash
export GROQ_API_KEY=your_key_here
```

Copy `profile.example.json` and fill in your own details:

```bash
cp profile.example.json profile.json
```

Then run it:

```bash
python main.py
```

## Tests

```bash
pytest
```

## Disclaimer

This is a personal project for learning. It is not financial advice, and you shouldn't make real investment decisions based on its output.
