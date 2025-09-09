# ChatBuy

🤖 Interactive Quantitative Trading System - Use AI to implement subjective quantitative strategies through natural language conversations.

## ✨ Features

- **Natural Language Processing**: Convert Chinese strategy descriptions to executable code
- **Strategy Generation**: Automatically create trading strategies from text descriptions
- **Backtesting Engine**: High-performance backtesting with vectorbt
- **Interactive Interface**: User-friendly CLI with rich output
- **Data Management**: Fetch data from Yahoo Finance or load from CSV

## 🚀 Quick Start

### 1. Installation

```bash
# Create environment and install dependencies
uv venv && source .venv/bin/activate
uv sync --frozen

# Install package in editable mode
uv pip install -e .
```

### 2. Interactive Demo

```bash
# Run the interactive interface
uv run python scripts/chatbuy_demo.py

# Run a simple demo
uv run python scripts/simple_demo.py

# Test all components
uv run python scripts/test_components.py
```

### 3. Usage Examples

Try these strategy descriptions:

```
双均线金叉买入，20日均线和50日均线
快线10日，慢线30日，金叉买入死叉卖出
短期均线交叉长期均线，快线20慢线60
```

## 📁 Project Structure

```
chatbuy/
├── nlp/                    # Natural language processing
├── strategies/             # Strategy generation
├── backtest/              # Backtesting engine
├── data/                  # Data management
├── ui/                    # User interface
└── scripts/               # Demo scripts
```

## 🎯 Currently Supported Strategies

- **Moving Average Crossover**: Dual moving average cross strategies
- More strategies coming soon!

## 📊 Backtesting Features

- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Equity curve visualization
- Trade analysis
- Risk management

## 🛠️ Development

### Adding New Strategies

1. Create strategy template in `chatbuy/strategies/templates/`
2. Implement NLP patterns in `chatbuy/nlp/`
3. Register in `StrategyFactory`

### Testing

```bash
# Run component tests
uv run python scripts/test_components.py

# Run linting
uv run ruff check . --fix
uv run ruff format .
```

## 📚 Documentation

- [Usage Guide](USAGE.md) - Detailed usage instructions
- [Agent Guidelines](AGENTS.md) - Development guidelines

## ⚠️ Disclaimer

This system is for educational purposes only. Past performance does not guarantee future results. Please do your own research before making any investment decisions.

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
