"""ChatBuy - Interactive Quantitative Trading System.

A natural language processing system for converting trading strategy descriptions
into executable backtests using vectorbt.
"""

from .backtest.engine import BacktestEngine
from .nlp.strategy_parser import StrategyParser
from .strategies.strategy_factory import StrategyFactory
from .ui.cli import ChatBuyCLI

__version__ = "0.1.1"
__author__ = "adofe"

__all__ = [
    "ChatBuyCLI",
    "StrategyParser", 
    "StrategyFactory",
    "BacktestEngine"
]