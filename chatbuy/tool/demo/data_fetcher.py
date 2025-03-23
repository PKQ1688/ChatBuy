import time

import ccxt
import pandas as pd
from tool.demo.technical_indicator import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from utils import logger

exchange = ccxt.binance({"rateLimit": 1200, "enableRateLimit": True})


def fetch_historical_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    limit: int = 1000,
    max_retries: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch historical K-line data from Binance in batches with retry mechanism.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Time period (e.g., '1m', '1h', '1d')
        start_date: Start time (ISO format, e.g., '2017-07-01T00:00:00Z')
        limit: Maximum number of entries per request (default 1000)
        max_retries: Maximum number of retries for fetching data (default 5)
        verbose: If True, enable verbose logging (default True)

    :return: DataFrame containing all historical data
    """
    all_data = []
    since = exchange.parse8601(start_date)
    retries = 0

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Start next batch from the latest time
            if verbose:
                logger.info(
                    f"Fetched {len(ohlcv)} rows. Current timestamp: {ohlcv[-1][0]}"
                )
            time.sleep(exchange.rateLimit / 1000)  # Avoid rate limit
            retries = 0  # Reset retries after a successful fetch
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            retries += 1
            if retries > max_retries:
                logger.error("Max retries reached. Exiting.")
                break
            sleep_time = min(2**retries, 60)  # Exponential backoff with a cap
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_current_price(symbol: str) -> float:
    """Fetch the current price of a trading pair from Binance.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')

    :return: Current price as a float, or None if an error occurs
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker["last"]
        logger.debug(f"Current price for {symbol}: {current_price}")
        return current_price
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        raise


def fetch_and_calculate_indicators(
    symbol: str, timeframe: str, start_date: str, limit: int = 1000
):
    """Fetch historical data and calculate various technical indicators.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Time period (e.g., '1m', '1h', '1d')
        start_date: Start time (ISO format, e.g., '2017-07-01T00:00:00Z')
        limit: Maximum number of entries per request (default 1000)

    Returns:
        dict: Dictionary containing calculated indicators (MACD, Bollinger Bands, RSI, SMA)
    """
    df = fetch_historical_data(symbol, timeframe, start_date, limit)
    prices = df["close"].tolist()

    macd, signal_line = calculate_macd(prices)
    middle_band, upper_band, lower_band = calculate_bollinger_bands(prices)
    rsi = calculate_rsi(prices)
    sma = calculate_sma(prices)

    indicators = {
        "MACD": macd,
        "Signal Line": signal_line,
        "Bollinger Bands": {
            "Middle Band": middle_band,
            "Upper Band": upper_band,
            "Lower Band": lower_band,
        },
        "RSI": rsi,
        "SMA": sma,
    }

    return indicators
