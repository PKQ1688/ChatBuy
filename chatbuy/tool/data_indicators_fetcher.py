import time

import ccxt
import pandas as pd
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

    :param symbol: Trading pair (e.g., 'BTC/USDT')
    :param timeframe: Time period (e.g., '1m', '1h', '1d')
    :param start_date: Start time (ISO format, e.g., '2017-07-01T00:00:00Z')
    :param limit: Maximum number of entries per request (default 1000)
    :param max_retries: Maximum number of retries for fetching data (default 5)
    :param verbose: If True, enable verbose logging (default True)

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
