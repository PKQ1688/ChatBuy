import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc


def visualize_btc_with_indicators(
    data_path="data/BTC_USDT_1d_with_indicators.csv", output_dir="data"
):
    """Read BTC data with technical indicators and create visualization charts.

    Parameters:
    data_path: Path to the BTC data file
    output_dir: Output directory for charts
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(data_path)
    data = data[-120:]

    # Print column names for debugging
    print("CSV file columns:", data.columns.tolist())

    # Check if time-related columns exist
    date_column = None
    possible_date_columns = [
        "date",
        "time",
        "timestamp",
        "Date",
        "Time",
        "Timestamp",
        "datetime",
        "Datetime",
    ]

    for col in possible_date_columns:
        if col in data.columns:
            date_column = col
            break

    # If no date column is found, use index as date
    if date_column is None:
        print("No date column found, using index for X axis")
        data["index"] = np.arange(len(data))
        x_values = data["index"]
        x_label = "Data Points"
        date_num = data["index"].values
    else:
        # Convert date column to datetime format
        print(f"Using '{date_column}' as date column")
        data[date_column] = pd.to_datetime(data[date_column])
        x_values = data[date_column]
        x_label = "Date"
        # Convert datetime to float for candlestick
        date_num = mdates.date2num(data[date_column].values)

    # Create figure
    plt.figure(figsize=(14, 12))

    # Three subplots: Price and Bollinger Bands, Volume, MACD
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    # If using date column, set date format
    if date_column is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Check if required columns exist
    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "macd",
        "signal",
        "histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        print(f"Warning: Missing columns in data: {missing_columns}")

    # Plot price with candlestick and Bollinger Bands
    ohlc_columns = ["open", "high", "low", "close"]
    if all(col in data.columns for col in ohlc_columns):
        # Prepare OHLC data for candlestick chart
        ohlc = []
        for i in range(len(data)):
            ohlc.append(
                [
                    date_num[i],
                    data["open"].iloc[i],
                    data["high"].iloc[i],
                    data["low"].iloc[i],
                    data["close"].iloc[i],
                ]
            )

        # Plot candlesticks
        candlestick_ohlc(
            ax1, ohlc, width=0.6, colorup="green", colordown="red", alpha=0.8
        )

        # If Bollinger Bands indicators exist, plot them
        if all(col in data.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            ax1.plot(
                x_values,
                data["bb_upper"],
                label="Upper Band",
                color="blue",
                alpha=0.7,
                linewidth=1.0,
            )
            ax1.plot(
                x_values,
                data["bb_middle"],
                label="Middle Band",
                color="purple",
                alpha=0.7,
                linewidth=1.0,
            )
            ax1.plot(
                x_values,
                data["bb_lower"],
                label="Lower Band",
                color="blue",
                alpha=0.7,
                linewidth=1.0,
            )
            ax1.fill_between(
                x_values, data["bb_upper"], data["bb_lower"], color="gray", alpha=0.1
            )
            ax1.set_title("BTC/USDT Price and Bollinger Bands", fontsize=12)
        else:
            ax1.set_title("BTC/USDT Price", fontsize=12)

        ax1.set_ylabel("Price (USDT)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
    elif "close" in data.columns:
        # Fallback to line chart if only close price is available
        ax1.plot(
            x_values, data["close"], label="BTC Price", color="black", linewidth=1.5
        )

        # If Bollinger Bands indicators exist, plot them
        if all(col in data.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            ax1.plot(
                x_values, data["bb_upper"], label="Upper Band", color="red", alpha=0.7
            )
            ax1.plot(
                x_values,
                data["bb_middle"],
                label="Middle Band",
                color="blue",
                alpha=0.7,
            )
            ax1.plot(
                x_values, data["bb_lower"], label="Lower Band", color="green", alpha=0.7
            )
            ax1.fill_between(
                x_values, data["bb_upper"], data["bb_lower"], color="gray", alpha=0.1
            )
            ax1.set_title("BTC/USDT Price and Bollinger Bands", fontsize=12)
        else:
            ax1.set_title("BTC/USDT Price", fontsize=12)

        ax1.set_ylabel("Price (USDT)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No price data",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Plot volume
    if "volume" in data.columns:
        ax2.bar(x_values, data["volume"], color="blue", alpha=0.5, width=0.8)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No volume data",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Plot MACD
    if all(col in data.columns for col in ["macd", "signal", "histogram"]):
        ax3.plot(x_values, data["macd"], label="MACD", color="blue", linewidth=1.2)
        ax3.plot(x_values, data["signal"], label="Signal", color="red", linewidth=1.2)

        # Create histogram
        colors = ["green" if val >= 0 else "red" for val in data["histogram"]]
        ax3.bar(x_values, data["histogram"], color=colors, alpha=0.5, width=0.8)

        ax3.set_title("MACD Indicator", fontsize=12)
        ax3.set_ylabel("MACD Value")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No MACD data",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Set x-axis label
    plt.xlabel(x_label)

    # If using date, rotate date labels to avoid overlap
    if date_column is not None:
        plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Generate output filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"BTC_indicators_{current_time}.png")

    # Save chart
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_file}")

    # Show chart
    plt.show()


if __name__ == "__main__":
    try:
        visualize_btc_with_indicators()
    except Exception as e:
        print(f"Error occurred: {e}")
        # Try using absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_path = os.path.join(project_dir, "data", "BTC_USDT_1d_with_indicators.csv")
        output_dir = os.path.join(project_dir, "data")
        print(f"Trying absolute path: {data_path}")

        try:
            visualize_btc_with_indicators(data_path, output_dir)
        except Exception as detailed_error:
            print(f"Detailed error information: {detailed_error}")

            # Try to read and output the first few lines of the CSV file to help debugging
            try:
                print("\nTrying to read first 5 rows of CSV file:")
                temp_data = pd.read_csv(data_path, nrows=5)
                print(temp_data.head())
            except Exception as csv_error:
                print(f"Cannot read CSV file: {csv_error}")
