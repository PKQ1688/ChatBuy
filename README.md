# ChatBuy

🤖 Interactive Quantitative Trading System - Use AI to implement subjective quantitative strategies through natural language conversations.

## ✨ Features

- **Dynamic Strategy Generation**: Convert any buy/sell conditions to executable strategies
- **Natural Language Processing**: Convert Chinese strategy descriptions to executable code
- **Multiple Technical Indicators**: Support for MA, RSI, MACD, Bollinger Bands, and more
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

# Test dynamic strategy generation
uv run python scripts/test_dynamic_strategy.py
```

### 3. Usage Examples

Try these strategy descriptions:

```
5日均线上穿20日均线时买入，下穿时卖出
RSI低于30买入，高于70卖出
MACD金叉买入，死叉卖出
当10日均线上穿30日均线时买入，反之下穿时卖出
双均线金叉买入，20日均线和50日均线
快线10日，慢线30日，金叉买入死叉卖出
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

- **Dynamic Strategy Generation**: Any buy/sell conditions you can describe
- **Moving Average Crossover**: Fast and slow moving average cross strategies  
- **RSI Strategies**: RSI oversold/overbought strategies
- **MACD Strategies**: MACD signal line crossover strategies
- **Bollinger Bands**: Price band-based strategies

### Technical Indicators Supported
- Moving Averages (any period)
- RSI (any period)
- MACD (line, signal, histogram)
- Bollinger Bands (any period)
- Custom indicator combinations

## 📊 Backtesting Features

- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Equity curve visualization
- Trade analysis
- Risk management

## 🛠️ Development

### Adding New Strategies

With the new dynamic strategy system, you don't need to create individual strategy templates. Simply describe your buy/sell conditions in natural language, and the system will automatically generate the strategy.

For custom indicators or advanced patterns:
1. Add indicator calculation to `DynamicStrategy.calculate_indicators()`
2. Add condition evaluation to `DynamicStrategy._evaluate_condition()`
3. Update NLP prompts to recognize new patterns

### Testing

```bash
# Run component tests
uv run python scripts/test_components.py

# Test dynamic strategy generation
uv run python scripts/test_dynamic_strategy.py

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
