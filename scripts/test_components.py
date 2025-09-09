#!/usr/bin/env python3
"""Simple test script for ChatBuy components."""

import pandas as pd

from chatbuy.backtest.engine import BacktestEngine
from chatbuy.data.fetcher import DataFetcher
from chatbuy.data.processor import DataProcessor
from chatbuy.nlp.strategy_parser import StrategyParser
from chatbuy.strategies.strategy_factory import StrategyFactory


def test_nlp_parsing():
    """Test NLP strategy parsing."""
    print("Testing NLP Strategy Parsing...")

    parser = StrategyParser()

    test_cases = [
        "双均线金叉买入，20日均线和50日均线",
        "快线10日，慢线30日，金叉买入死叉卖出",
        "短期均线交叉长期均线，快线20慢线60",
    ]

    for test_case in test_cases:
        print(f"\nInput: {test_case}")
        result = parser.parse(test_case)
        if result:
            print(f"Strategy Type: {result['strategy_type']}")
            print(f"Parameters: {result['parameters']}")
            print(f"Confidence: {result['confidence']}")
        else:
            print("Failed to parse")


def test_strategy_creation():
    """Test strategy creation."""
    print("\n\nTesting Strategy Creation...")

    factory = StrategyFactory()

    # Test MA strategy
    parameters = {"fast_period": 20, "slow_period": 50}
    strategy = factory.create_strategy("moving_average_cross", parameters)

    if strategy:
        print(f"Strategy created: {strategy.name}")
        print(f"Parameters: {strategy.parameters}")
        print(f"Validation: {strategy.validate_parameters()}")
    else:
        print("Failed to create strategy")


def test_data_fetching():
    """Test data fetching."""
    print("\n\nTesting Data Fetching...")

    fetcher = DataFetcher()
    processor = DataProcessor()

    # Test with a small dataset
    data = fetcher.fetch_yfinance(
        "BTC-USD", start_date="2024-01-01", end_date="2024-02-01"
    )

    if data is not None:
        print(f"Data fetched: {len(data)} rows")
        print(f"Columns: {list(data.columns)}")

        # Test data processing
        cleaned_data = processor.clean_data(data)
        print(f"Cleaned data: {len(cleaned_data)} rows")
        print(f"Data validation: {processor.validate_data(cleaned_data)}")

        # Test strategy signals
        factory = StrategyFactory()
        strategy = factory.create_strategy(
            "moving_average_cross", {"fast_period": 5, "slow_period": 10}
        )

        if strategy:
            signals = strategy.generate_signals(cleaned_data)
            print(f"Signals generated: {len(signals)}")
            print(f"Buy signals: {(signals == 1).sum()}")
            print(f"Sell signals: {(signals == -1).sum()}")
    else:
        print("Failed to fetch data")


def test_backtest_engine():
    """Test backtest engine."""
    print("\n\nTesting Backtest Engine...")

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    prices = [100 + i * 0.5 + (i % 10) * 2 for i in range(100)]

    sample_data = pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Open": [p - 1 for p in prices],
            "High": [p + 2 for p in prices],
            "Low": [p - 2 for p in prices],
            "Volume": [1000000] * 100,
        }
    )

    # Test strategy
    factory = StrategyFactory()
    strategy = factory.create_strategy(
        "moving_average_cross", {"fast_period": 5, "slow_period": 10}
    )

    if strategy:
        engine = BacktestEngine()
        results = engine.run_strategy_backtest(strategy, sample_data)

        if results:
            print("Backtest completed successfully!")
            print(f"Strategy: {results['strategy']['name']}")
            print(f"Total trades: {results['stats'].get('total_trades', 'N/A')}")
            print(
                f"Total return: {results['stats'].get('formatted', {}).get('Total Return', 'N/A')}"
            )
        else:
            print("Backtest failed")
    else:
        print("Failed to create strategy")


def main():
    """Run all tests."""
    print("=" * 50)
    print("ChatBuy Component Tests")
    print("=" * 50)

    test_nlp_parsing()
    test_strategy_creation()
    test_data_fetching()
    test_backtest_engine()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
