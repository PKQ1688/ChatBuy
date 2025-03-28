import os
import time

import ccxt
import pandas as pd

# Initialize Binance client
exchange = ccxt.binance({"rateLimit": 1200, "enableRateLimit": True})


def fetch_historical_data(symbol, timeframe, start_date, limit=1000, max_retries=3):
    """Fetch historical K-line data from Binance in batches with retry mechanism.

    :param symbol: Trading pair (e.g., 'BTC/USDT')
    :param timeframe: Time period (e.g., '1m', '1h', '1d')
    :param start_date: Start time (ISO format, e.g., '2017-07-01T00:00:00Z')
    :param limit: Maximum number of entries per request (default 1000)
    :param max_retries: Maximum number of retries for fetching data (default 5)
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
            print(f"Fetched {len(ohlcv)} rows. Current timestamp: {ohlcv[-1][0]}")
            time.sleep(exchange.rateLimit / 1000)  # Avoid rate limit
            retries = 0  # Reset retries after a successful fetch
        except Exception as e:
            print(f"Error fetching data: {e}")
            retries += 1
            if retries > max_retries:
                print("Max retries reached. Exiting.")
                break
            sleep_time = min(2 ** retries, 60)  # Exponential backoff with a cap
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def main(symbol="BTC", timeframe="1d", start_date="2017-07-01T00:00:00Z"):
    """Fetch and save historical cryptocurrency data from Binance.

    Args:
        symbol (str): The cryptocurrency symbol (default: 'BTC')
        timeframe (str): The time interval for data points (default: '1d')
        start_date (str): The start date in ISO format (default: '2017-07-01T00:00:00Z')
    """
    # Get historical data for the trading pair
    symbol_pair = f"{symbol}/USDT"
    timeframe = timeframe  # Data for each day
    start_date = start_date  # Binance launch date

    df = fetch_historical_data(symbol_pair, timeframe, start_date)

    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save as CSV file
    df.to_csv(f"data/{symbol}_USDT_{timeframe}.csv", index=False)
    print(f"History data is stored in {symbol}_USDT_{timeframe}.csv")


if __name__ == "__main__":
    main(symbol="BTC")
