from chatbuy.trader.rotation_strategy import RotationStrategy


def run_rotation_strategy(
    symbols=["BTC-USD", "ETH-USD"],
    start="2022-01-01 UTC",
    end="2024-01-01 UTC",
    lookback_period=20,
):
    """Run a rotation strategy between the specified crypto assets.

    Args:
        symbols: List of symbols to rotate between
        start: Start date for backtest
        end: End date for backtest
        lookback_period: Number of days to look back for performance comparison

    Returns:
        The rotation strategy instance
    """
    # Create and run the rotation strategy
    strategy = RotationStrategy(
        symbols=symbols, start=start, end=end, lookback_period=lookback_period
    )

    strategy.run_and_visualize()
    return strategy


if __name__ == "__main__":
    print("\n--- Crypto Rotation Strategy ---")
    # Run the default BTC/ETH rotation strategy
    run_rotation_strategy()
