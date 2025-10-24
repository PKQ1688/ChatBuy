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

### Running the Application

#### Web Interface (Primary)
```bash
# Run Gradio web interface (recommended)
uv run python scripts/run_gradio_app.py
# Opens at http://localhost:7860 automatically
```

#### CLI Interface
```bash
# Interactive CLI demo
uv run python scripts/chatbuy_demo.py

# Simple demo with predefined examples
uv run python scripts/simple_demo.py
```

### Testing and Quality Assurance
```bash
# Run component tests
uv run python scripts/test_components.py

# Test dynamic strategy generation
uv run python scripts/test_dynamic_strategy.py

# Test Gradio interface
uv run python scripts/test_gradio.py

# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .
```

## Project Architecture

ChatBuy is an interactive quantitative trading system that converts Chinese natural language strategy descriptions into executable backtests. The system features dynamic strategy generation and dual interface support.

### Core Modules
- **NLP Module** (`chatbuy/nlp/`): Processes Chinese strategy descriptions using OpenAI's API to extract parameters
- **Strategy Factory** (`chatbuy/strategies/`): Dynamic strategy generation using factory pattern
- **Backtest Engine** (`chatbuy/backtest/`): High-performance backtesting with vectorbt
- **Data Management** (`chatbuy/data/`): Fetches market data from Yahoo Finance, processes with pandas
- **UI Module** (`chatbuy/ui/`): Dual interface - Gradio web app and rich CLI

### Key Patterns
- **Dynamic Strategy Generation**: Single `DynamicStrategy` class handles arbitrary buy/sell conditions
- **Factory Pattern**: StrategyFactory creates strategy instances based on NLP parsing
- **Template Pattern**: BaseStrategy defines interface, implementations handle signal generation
- **LLM Integration**: StrategyParser uses OpenAI API for Chinese language processing
- **Dual Interface Architecture**: Shared backend components serve both web and CLI interfaces

### Dynamic Strategy System
The system now uses a single `DynamicStrategy` class that can handle any combination of:
- Moving averages (any period)
- RSI indicators (any period)
- MACD signals
- Bollinger Bands
- Custom condition combinations

### Data Flow
1. User inputs Chinese strategy description via web/CLI
2. NLP parser extracts strategy parameters using LLM
3. Strategy factory creates `DynamicStrategy` instance
4. Data fetcher retrieves market data (Yahoo Finance)
5. Backtest engine executes strategy and generates comprehensive metrics
6. Results displayed via rich CLI or interactive web interface

## Development Guidelines

### Environment Configuration
- Requires OpenAI API key in environment variables: `API_KEY`, `MODEL_URL`, `MODEL_NAME`
- Uses vectorbt for backtesting, yfinance for data fetching, gradio for web UI
- Configuration loaded via python-dotenv

### Code Style and Standards
- Python 3.12+ with type hints required
- Google-style docstrings (pydocstyle configured in pyproject.toml)
- Double quotes for strings, 4-space indentation
- Ruff for linting and formatting (configuration in pyproject.toml)
- Basic type checking with pyright (non-strict mode)

### Architecture Principles
- **KISS**: Simple, straightforward implementations
- **DRY**: Shared backend components for web/CLI interfaces
- **SOLID**: Clear separation of concerns between modules
- **YAGNI**: Feature-focused development without over-engineering

### Important Notes
- System optimized for Chinese language strategy descriptions
- All strategies validate parameters before execution
- Comprehensive error handling throughout the pipeline
- Web interface provides real-time strategy preview and interactive charts
- Backtest results include performance metrics, equity curves, and detailed trade analysis
