# import pandas as pd
from talipp.indicators import MACD, RSI, SMA, BollingerBands


def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        prices: List of price values
        slow: Slow period for MACD calculation (default: 26)
        fast: Fast period for MACD calculation (default: 12)
        signal: Signal period for MACD calculation (default: 9)

    Returns:
        tuple: (MACD value, Signal line value)
    """
    macd = MACD(fast_period=fast, slow_period=slow, signal_period=signal)
    for price in prices:
        macd.add_input_value(price)
    macd.calculate()
    return macd[-1].macd, macd[-1].signal


def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    """Calculate Bollinger Bands technical indicator.

    Args:
        prices: List of price values
        window: Period for calculating moving average (default: 20)
        num_std_dev: Number of standard deviations for bands (default: 2)

    Returns:
        tuple: (Middle band, Upper band, Lower band)
    """
    bb = BollingerBands(period=window, std_dev=num_std_dev)
    for price in prices:
        bb.add_input_value(price)
    bb.calculate()
    return bb[-1].middle_band, bb[-1].upper_band, bb[-1].lower_band


def calculate_rsi(prices, window=14):
    """Calculate the Relative Strength Index (RSI) indicator.

    Args:
        prices: List of price values
        window: Period for calculating RSI (default: 14)

    Returns:
        float: RSI value
    """
    rsi = RSI(period=window)
    for price in prices:
        rsi.add_input_value(price)
    rsi.calculate()
    return rsi[-1]


def calculate_sma(prices, window):
    """Calculate the Simple Moving Average (SMA) indicator.

    Args:
        prices: List of price values
        window: Period for calculating SMA

    Returns:
        float: SMA value
    """
    sma = SMA(period=window)
    for price in prices:
        sma.add_input_value(price)
    sma.calculate()
    return sma[-1]
