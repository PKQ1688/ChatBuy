# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

### Setup Commands
```bash
# Create environment and install dependencies
uv venv && source .venv/bin/activate && uv sync --frozen

# Install package in editable mode
uv pip install -e .

# Add/remove dependencies
uv add <pkg> / uv remove <pkg>
```

### Testing and Quality Assurance
```bash
# Run component tests
uv run python scripts/test_components.py

# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .

# Run demo scripts
uv run python scripts/chatbuy_demo.py
uv run python scripts/simple_demo.py
```

## Project Architecture

ChatBuy is an interactive quantitative trading system that converts natural language strategy descriptions into executable backtests. The system uses a modular architecture with these key components:

### Core Modules
- **NLP Module** (`chatbuy/nlp/`): Processes natural language descriptions using OpenAI's API to extract strategy parameters
- **Strategy Factory** (`chatbuy/strategies/`): Creates trading strategy instances using factory pattern
- **Backtest Engine** (`chatbuy/backtest/`): Executes strategies using vectorbt for high-performance backtesting
- **Data Management** (`chatbuy/data/`): Fetches market data from Yahoo Finance or CSV files
- **UI Module** (`chatbuy/ui/`): Provides CLI interface with rich output formatting

### Key Patterns
- **Factory Pattern**: StrategyFactory creates strategy instances based on parsed NLP output
- **Template Pattern**: BaseStrategy defines interface, specific strategies implement generate_signals() and calculate_indicators()
- **Wrapper Pattern**: VectorbtWrapper encapsulates vectorbt operations for cleaner API
- **LLM Integration**: StrategyParser uses OpenAI API to convert Chinese trading descriptions to structured parameters

### Current Strategy Support
- Moving average crossover strategies (dual MA)
- RSI-based strategies (oversold/overbought)
- Bollinger bands strategies

### Data Flow
1. User inputs strategy description in Chinese
2. NLP parser extracts parameters using LLM
3. Strategy factory creates appropriate strategy instance
4. Data fetcher retrieves market data
5. Backtest engine executes strategy and generates metrics
6. Results displayed with rich CLI formatting

## Development Guidelines

### Adding New Strategies
1. Create strategy class inheriting from BaseStrategy in `chatbuy/strategies/templates/`
2. Implement `generate_signals()` and `calculate_indicators()` methods
3. Register strategy in StrategyFactory._strategies dictionary
4. Update NLP parser prompt to recognize new strategy types

### Environment Configuration
- Requires OpenAI API key in environment variables (API_KEY, MODEL_URL, MODEL_NAME)
- Uses vectorbt for backtesting, yfinance for data fetching
- Configuration loaded via python-dotenv

### Code Style
- Python 3.12+ with type hints required
- Google-style docstrings (pydocstyle configured in pyproject.toml)
- Double quotes for strings, 4-space indentation
- Ruff for linting and formatting (configuration in pyproject.toml)

### Important Notes
- The system is designed for Chinese language strategy descriptions
- All strategies must validate parameters before execution
- Backtest results include performance metrics, equity curves, and trade analysis
- Error handling is built into all major components