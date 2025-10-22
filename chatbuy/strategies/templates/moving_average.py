import pandas as pd

from ..base_strategy import BaseStrategy, StrategyParamValue


class MovingAverageCrossStrategy(BaseStrategy):
    """Moving Average Crossover Strategy."""

    def __init__(self, parameters: dict[str, StrategyParamValue]):
        super().__init__(parameters)
        self.fast_period = parameters.get("fast_period", 20)
        self.slow_period = parameters.get("slow_period", 50)

    def validate_parameters(self) -> bool:
        """Validate MA parameters."""
        return (
            isinstance(self.fast_period, int)
            and isinstance(self.slow_period, int)
            and self.fast_period > 0
            and self.slow_period > 0
            and self.fast_period < self.slow_period
        )

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        df = data.copy()

        # Calculate fast and slow moving averages
        df["fast_ma"] = df["Close"].rolling(window=self.fast_period).mean()
        df["slow_ma"] = df["Close"].rolling(window=self.slow_period).mean()

        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on MA crossover."""
        df = self.calculate_indicators(data)

        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signals = pd.Series(0, index=df.index)

        # Buy signal: fast MA crosses above slow MA
        buy_condition = (df["fast_ma"] > df["slow_ma"]) & (
            df["fast_ma"].shift(1) <= df["slow_ma"].shift(1)
        )
        signals[buy_condition] = 1

        # Sell signal: fast MA crosses below slow MA
        sell_condition = (df["fast_ma"] < df["slow_ma"]) & (
            df["fast_ma"].shift(1) >= df["slow_ma"].shift(1)
        )
        signals[sell_condition] = -1

        return signals

    def get_info(self) -> dict[str, str | int | dict[str, StrategyParamValue]]:
        """Get strategy information."""
        info = super().get_info()
        info.update(
            {
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "description": f"Moving Average Crossover Strategy (Fast: {self.fast_period}, Slow: {self.slow_period})",
            }
        )
        return info
