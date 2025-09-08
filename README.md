# ChatBuy

Use AI to implement subjective quantitative strategies. Strategies can be formulated through conversations, and backtesting can also be performed through conversations.

## 1. Installation (uv)

Recommended: manage the environment with uv.

```bash
# create venv and install deps from pyproject + uv.lock
uv venv && source .venv/bin/activate
uv sync --frozen

# (optional) install package in editable mode for local imports
uv pip install -e .
```

## 2. VectorBT Demo: ETH Dual Moving Average

This repo includes a simple dual moving‑average backtest demo for ETH built with `vectorbt`.

Run from the repo root:

```bash
# 2.1 Your own CSV (must contain Date and Close columns)
uv run python scripts/eth_dual_ma_vectorbt.py --source csv --csv-path data/ETH-USD.csv --fast 20 --slow 50

# 2.2 Yahoo Finance (requires network and yfinance)
uv add yfinance
uv run python scripts/eth_dual_ma_vectorbt.py --source yfinance --symbol ETH-USD --start 2020-01-01 --interval 1d --plot

# 2.3 Grid search across windows
uv run python scripts/eth_dual_ma_vectorbt.py --source yfinance \
  --grid --fast-list 5,10,20,50 --slow-list 60,100,150,200 --top 10
```

Notes
- Fees are proportional (default 0.0007 = 7 bps per trade). Adjust via `--fees`.
- Initial cash defaults to 10,000. Adjust via `--cash`.
- Use `--plot` for an interactive Plotly equity curve and MA overlay. A heatmap is shown for grid runs.
- For CSV input, the script detects typical date/close column names. Minimal format:

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,128.1,130.0,120.5,129.4,123456
...
```

Disclaimer: This demo is for educational purposes and not financial advice.
