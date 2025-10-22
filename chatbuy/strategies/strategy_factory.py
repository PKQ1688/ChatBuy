from .base_strategy import BaseStrategy
from .templates.dynamic_strategy import DynamicStrategy
from .templates.moving_average import MovingAverageCrossStrategy

# Strategy parameter value types
type StrategyParamValue = (
    int | float | str | bool | list[dict[str, str | int | float]] | dict[str, list[int]]
)
type StrategyParameters = dict[str, StrategyParamValue]


class StrategyFactory:
    """Factory class for creating strategy instances."""

    _strategies: dict[str, type[BaseStrategy]] = {
        "moving_average_cross": MovingAverageCrossStrategy,
        "dynamic_strategy": DynamicStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_type: str, parameters: StrategyParameters
    ) -> BaseStrategy | None:
        """Create a strategy instance based on type and parameters."""
        if strategy_type not in cls._strategies:
            return None

        strategy_class = cls._strategies[strategy_type]
        try:
            strategy = strategy_class(parameters)
            if strategy.validate_parameters():
                return strategy
            else:
                return None
        except Exception:
            return None

    @classmethod
    def get_available_strategies(cls) -> dict[str, str]:
        """Get list of available strategy types."""
        return {
            strategy_type: strategy_class.__name__
            for strategy_type, strategy_class in cls._strategies.items()
        }

    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: type[BaseStrategy]):
        """Register a new strategy type."""
        cls._strategies[strategy_type] = strategy_class
