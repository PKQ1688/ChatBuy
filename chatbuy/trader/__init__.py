"""
Trader module for implementing trading strategies.
"""

from .rotation_strategy import BTCETHRotationStrategy
from .data_manager import CryptoDataManager
from .portfolio import Portfolio

__all__ = ['BTCETHRotationStrategy', 'CryptoDataManager', 'Portfolio']