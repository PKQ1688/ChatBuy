import os
import sys
import time
import traceback

import ccxt
import pandas as pd

from chatbuy.scripts.basic_indicators import add_basic_indicators

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
            print(
                f"Requesting data: symbol={symbol}, timeframe={timeframe}, since={exchange.iso8601(since)}"
            )
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            if not ohlcv:
                print(f"No data returned for {symbol} since {exchange.iso8601(since)}.")
                # If this is our first request and we get no data, there's likely an issue
                if len(all_data) == 0:
                    print(
                        "No initial data returned. Please check the symbol and timeframe are valid."
                    )
                    if retries < max_retries:
                        retries += 1
                        sleep_time = min(2**retries, 60)
                        print(
                            f"Retrying in {sleep_time} seconds... (Attempt {retries}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                break

            print(f"Successfully fetched {len(ohlcv)} data points.")
            all_data.extend(ohlcv)

            # Check if we've reached the current time
            last_timestamp = ohlcv[-1][0]
            current_timestamp = exchange.milliseconds()
            since = last_timestamp + 1  # Start next batch from the latest time

            if (
                last_timestamp
                >= current_timestamp - exchange.parse_timeframe(timeframe) * 1000
            ):
                print("Reached current time. Completed data fetch.")
                break

            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            retries = 0  # Reset retries after a successful fetch

        except Exception as e:
            print(f"Error fetching data: {e}")
            retries += 1
            if retries > max_retries:
                print(f"Max retries ({max_retries}) reached. Exiting.")
                break
            sleep_time = min(2**retries, 60)  # Exponential backoff with a cap
            print(
                f"Retrying in {sleep_time} seconds... (Attempt {retries}/{max_retries})"
            )
            time.sleep(sleep_time)

    # If we didn't get any data at all
    if not all_data:
        raise ValueError(
            f"No data could be retrieved for {symbol} with timeframe {timeframe}."
        )

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    print(f"Total data points fetched: {len(df)}")
    return df


def main(symbol="BTC", timeframe="1d", start_date="2017-07-01T00:00:00Z", debug=False):
    """Fetch and save historical cryptocurrency data from Binance.

    Args:
        symbol (str): The cryptocurrency symbol (default: 'BTC')
        timeframe (str): The time interval for data points (default: '1d')
        start_date (str): The start date in ISO format (default: '2017-07-01T00:00:00Z')
        debug (bool): Enable detailed debugging output (default: False)
    """
    # Get historical data for the trading pair
    symbol_pair = f"{symbol}/USDT"

    print(
        f"Fetching {symbol_pair} data with timeframe {timeframe} from {start_date}..."
    )

    try:
        df = fetch_historical_data(symbol_pair, timeframe, start_date)

        if len(df) == 0:
            print(f"Error: No data was returned for {symbol_pair}.")
            return

        # Create a backup of the raw data before adding indicators
        if not os.path.exists("data"):
            os.makedirs("data")

        # Save raw data first in case indicator calculation fails
        raw_output_path = f"data/{symbol}_USDT_{timeframe}_raw.csv"
        df.to_csv(raw_output_path, index=False)
        print(f"Saved raw data to {raw_output_path}")

        print(f"Adding technical indicators to {len(df)} rows of data...")
        df = add_basic_indicators(df)

        output_path = f"data/{symbol}_USDT_{timeframe}_with_indicators.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully saved data with indicators to {output_path}")
        print(f"Data shape: {df.shape} rows × {df.shape[1]} columns")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch and process cryptocurrency data"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Cryptocurrency symbol (default: BTC)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d", help="Time interval (default: 1d)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2017-07-01T00:00:00Z",
        help="Start date in ISO format (default: 2017-07-01T00:00:00Z)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable detailed debugging information"
    )

    args = parser.parse_args()

    main(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        debug=args.debug,
    )
