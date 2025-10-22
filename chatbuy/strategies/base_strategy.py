from abc import ABC, abstractmethod

import pandas as pd

# Strategy parameter value types
type StrategyParamValue = (
    int | float | str | bool | list[dict[str, str | int | float]] | dict[str, list[int]]
)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, parameters: dict[str, StrategyParamValue]):
        self.parameters = parameters
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals (1 for buy, -1 for sell, 0 for hold)."""
        pass

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        pass

    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        return True

    def get_info(self) -> dict[str, str | dict[str, StrategyParamValue]]:
        """Get strategy information."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "description": self.__doc__ or "No description available",
        }
