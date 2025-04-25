import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc


class IndicatorVisualizer:
    """A class for visualizing financial time series data and technical indicators (such as Bollinger Bands, MACD).
    
    Provides functionality for generating single charts and batch processing.
    """

    DEFAULT_POSSIBLE_DATE_COLUMNS: list[str] = [
        "date",
        "time",
        "timestamp",
        "Date",
        "Time",
        "Timestamp",
        "datetime",
        "Datetime",
    ]
    REQUIRED_INDICATOR_COLUMNS: list[str] = [
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
    OHLC_COLUMNS: list[str] = ["open", "high", "low", "close"]
    BB_COLUMNS: list[str] = ["bb_upper", "bb_middle", "bb_lower"]
    MACD_COLUMNS: list[str] = ["macd", "signal", "histogram"]

    def __init__(self, possible_date_columns: list[str] | None = None):
        """Initialize IndicatorVisualizer.

        Args:
            possible_date_columns (list[str] | None): List of column names that might contain date/time information.
                                                     If None, the default list will be used.
        """
        self.possible_date_columns = (
            possible_date_columns or self.DEFAULT_POSSIBLE_DATE_COLUMNS
        )

    def _find_date_column(self, data: pd.DataFrame) -> str | None:
        """Try to find the date/time column in the DataFrame."""
        for col in self.possible_date_columns:
            if col in data.columns:
                return col
        return None

    def _prepare_data_and_xaxis(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, str | None, np.ndarray, pd.Series, str]:
        """Prepare plotting data and X-axis information."""
        data = data.copy()
        date_column = self._find_date_column(data)

        if date_column:
            try:
                data[date_column] = pd.to_datetime(data[date_column])
                x_values = data[date_column]
                x_label = "Date"
                date_num = mdates.date2num(data[date_column].values)
            except Exception as e:
                print(
                    f"Warning: Could not convert column '{date_column}' to datetime: {e}. Falling back to index."
                )
                date_column = None  # Fallback to index if conversion fails
        else:
            date_column = None  # Explicitly set to None if not found initially

        if date_column is None:
            data["index"] = np.arange(len(data))
            x_values = data["index"]
            x_label = "Data Points"
            date_num = data["index"].values

        return data, date_column, date_num, x_values, x_label

    def _setup_axes(
        self, date_column: str | None
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
        """Setup matplotlib figure and subplots."""
        fig = plt.figure(figsize=(14, 12))
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)  # Price and BB
        ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)  # Volume
        ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)  # MACD

        # Hide X-axis labels and ticks on ax1 and ax2 since they share the X-axis with ax3
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax2.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if date_column:
            # Initial setup, will be further adjusted in _format_xaxis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        return fig, ax1, ax2, ax3

    def _plot_price_and_bb(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        date_num: np.ndarray,
        x_values: pd.Series,
    ):
        """Plot price (candlestick or close line) and Bollinger Bands."""
        has_ohlc = all(col in data.columns for col in self.OHLC_COLUMNS)
        has_bb = all(col in data.columns for col in self.BB_COLUMNS)
        has_close = "close" in data.columns

        if has_ohlc:
            ohlc = data[self.OHLC_COLUMNS].copy()
            ohlc.insert(0, "date_num", date_num)
            candlestick_ohlc(
                ax, ohlc.values, width=0.6, colorup="green", colordown="red", alpha=0.8
            )
            title = "Price and Bollinger Bands" if has_bb else "Price"
        elif has_close:
            ax.plot(
                x_values, data["close"], label="Price", color="black", linewidth=1.5
            )
            title = "Price and Bollinger Bands" if has_bb else "Price"
        else:
            ax.text(
                0.5,
                0.5,
                "No price data (OHLC or Close)",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("Price Data Missing")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            return  # No price to plot BB against

        if has_bb and (has_ohlc or has_close):
            ax.plot(
                x_values,
                data["bb_upper"],
                label="Upper Band",
                color="blue",
                alpha=0.7,
                linewidth=1.0,
            )
            ax.plot(
                x_values,
                data["bb_middle"],
                label="Middle Band",
                color="purple",
                alpha=0.7,
                linewidth=1.0,
            )
            ax.plot(
                x_values,
                data["bb_lower"],
                label="Lower Band",
                color="blue",
                alpha=0.7,
                linewidth=1.0,
            )
            ax.fill_between(
                x_values, data["bb_upper"], data["bb_lower"], color="gray", alpha=0.1
            )

        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    def _plot_volume(self, ax: plt.Axes, data: pd.DataFrame, x_values: pd.Series):
        """Plot volume as a bar chart."""
        if "volume" in data.columns:
            ax.bar(
                x_values, data["volume"], color="blue", alpha=0.5, width=0.8
            )  # Use width parameter to adjust bar width
            ax.set_ylabel("Volume")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No volume data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel("Volume")  # Keep label for consistency

    def _plot_macd(self, ax: plt.Axes, data: pd.DataFrame, x_values: pd.Series):
        """Plot MACD indicator."""
        if all(col in data.columns for col in self.MACD_COLUMNS):
            ax.plot(x_values, data["macd"], label="MACD", color="blue", linewidth=1.2)
            ax.plot(
                x_values, data["signal"], label="Signal", color="red", linewidth=1.2
            )
            colors = ["green" if val >= 0 else "red" for val in data["histogram"]]
            ax.bar(
                x_values, data["histogram"], color=colors, alpha=0.5, width=0.8
            )  # Use width parameter
            ax.set_title("MACD Indicator", fontsize=12)
            ax.set_ylabel("MACD Value")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No MACD data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_title("MACD Indicator", fontsize=12)  # Keep title for consistency
            ax.set_ylabel("MACD Value")  # Keep label for consistency

    def _format_xaxis(
        self,
        ax: plt.Axes,
        date_column: str | None,
        date_num: np.ndarray,
        x_values: pd.Series,
        x_label: str,
    ):
        """Format X-axis labels and ticks."""
        ax.set_xlabel(x_label)

        # Try to get a reasonable number of ticks
        num_ticks = 10  # Can be adjusted as needed
        locator = plt.MaxNLocator(
            nbins=num_ticks, prune="both"
        )  # 'both' avoids edge ticks
        ax.xaxis.set_major_locator(locator)

        if date_column:
            # Ensure DateFormatter is used
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            # Auto-rotate labels to avoid overlap
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        else:
            # For non-date axes, ensure labels are integers
            def format_func(value, tick_number):
                return f"{int(value)}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # Force display of first and last tick labels (if they're not in the auto-generated ticks)
        # Note: This might make labels too dense, use with caution or find a better strategy
        # current_ticks = ax.get_xticks()
        # first_val = x_values.iloc[0]
        # last_val = x_values.iloc[-1]
        # Ensure first/last logic might need refinement depending on locator behavior

    def visualize(self, data: pd.DataFrame, output_file_path: str, show: bool = False):
        """Generate a candlestick+indicator chart from a DataFrame and save it to the specified path.

        Args:
            data (pd.DataFrame): DataFrame containing price and indicator data.
            output_file_path (str): Complete file path for the output image.
            show (bool): Whether to display the image after generation.

        Returns:
            str: Path to the output file.

        Raises:
            ValueError: If the output directory cannot be created.
        """
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create output directory '{output_dir}': {e}")

        # Check for missing columns (optional, but helpful for debugging)
        # present_cols = set(data.columns)
        # missing_required = [col for col in self.REQUIRED_INDICATOR_COLUMNS if col not in present_cols]
        # if missing_required:
        #     print(f"Warning: Missing some standard indicator columns: {missing_required}")

        df_prepared, date_col, date_num_vals, x_axis_vals, x_axis_label = (
            self._prepare_data_and_xaxis(data)
        )
        fig, ax_price, ax_vol, ax_macd = self._setup_axes(date_col)

        self._plot_price_and_bb(ax_price, df_prepared, date_num_vals, x_axis_vals)
        self._plot_volume(ax_vol, df_prepared, x_axis_vals)
        self._plot_macd(ax_macd, df_prepared, x_axis_vals)

        self._format_xaxis(ax_macd, date_col, date_num_vals, x_axis_vals, x_axis_label)

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.97]
        )  # Adjust layout to prevent title and label overlap

        try:
            plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving chart to {output_file_path}: {e}")
            plt.close(fig)  # Ensure figure is closed even on save error
            return None  # Indicate failure

        if show:
            plt.show()

        plt.close(fig)  # Close figure to free memory

        return output_file_path

    def batch_generate(
        self,
        data_path: str,
        output_dir: str,
        length: int = 120,
        end_time: str | None = None,
        start_time: str | None = None,
        step: int = 1,
        show: bool = False,
        filename_prefix: str = "chart",
    ):
        """Batch generate charts using a sliding window approach on time series data.

        Args:
            data_path (str): Path to the CSV data file.
            output_dir (str): Directory for output images.
            length (int): Number of data points in each chart (window length).
            end_time (str | None): End time for data filtering ('YYYY-MM-DD'). None means use end of data.
            start_time (str | None): Start time for data filtering ('YYYY-MM-DD'). None means use start of data.
            step (int): Step size for the sliding window.
            show (bool): Whether to display each generated chart.
            filename_prefix (str): Prefix for generated image filenames.

        Raises:
            ValueError: If time column cannot be found or specified time range is invalid.
            FileNotFoundError: If data file doesn't exist.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Cannot read or parse CSV file '{data_path}': {e}")

        date_column = self._find_date_column(df)
        if date_column is None:
            raise ValueError(
                f"No recognizable time column found in {data_path} (check if column names are in {self.possible_date_columns}). Cannot filter by time."
            )

        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(
                f"Cannot convert column '{date_column}' to datetime format: {e}"
            )

        df = df.sort_values(by=date_column).reset_index(
            drop=True
        )  # Ensure time sorting

        # Determine start and end indices for data processing
        start_idx = 0
        if start_time:
            start_time_dt = pd.to_datetime(start_time)
            # Find first index greater than or equal to start_time
            start_indices = df.index[df[date_column] >= start_time_dt]
            if not start_indices.empty:
                start_idx = start_indices.min()
            else:
                raise ValueError(
                    f"Specified start time '{start_time}' is later than all dates in the data."
                )

        end_idx = len(df)
        if end_time:
            end_time_dt = pd.to_datetime(end_time)
            # Find last index less than or equal to end_time + 1 as slice endpoint
            end_indices = df.index[df[date_column] <= end_time_dt]
            if not end_indices.empty:
                end_idx = end_indices.max() + 1  # Add 1 to include end_time day
            else:
                raise ValueError(
                    f"Specified end time '{end_time}' is earlier than all dates in the data."
                )

        if start_idx >= end_idx:
            raise ValueError(
                f"Calculated start index ({start_idx}) is greater than or equal to end index ({end_idx}). Please check time range."
            )

        print(
            f"Starting batch generation: from index {start_idx} to {end_idx - 1}, window length {length}, step size {step}"
        )
        generated_count = 0
        # Generate images with sliding window
        for i in range(start_idx, end_idx - length + 1, step):
            sub_df = df.iloc[i : i + length]
            if (
                len(sub_df) < length
            ):  # Theoretically shouldn't happen unless end_idx is calculated incorrectly
                print(
                    f"Skipping index {i}: Not enough data points ({len(sub_df)} < {length})"
                )
                continue

            # Build output filename
            window_end_date = sub_df[date_column].iloc[-1]
            # Format date to ensure legal filename
            date_str = window_end_date.strftime("%Y%m%d")
            output_filename = f"{filename_prefix}_{date_str}_len{length}_idx{i}.png"
            output_file_path = os.path.join(output_dir, output_filename)

            print(
                f"  Generating chart for window ending {date_str} (index {i} to {i + length - 1})..."
            )
            try:
                self.visualize(sub_df, output_file_path, show=show)
                generated_count += 1
            except Exception as e:
                print(f"  Error generating chart for window ending {date_str}: {e}")
                # Decide whether to continue or stop on error
                # continue

        print(f"Batch generation complete, {generated_count} images generated.")


if __name__ == "__main__":
    visualizer = IndicatorVisualizer()

    # --- Example 1: Generate single chart (using most recent 120 data points) ---
    print("\n--- Example 1: Generate Single Chart ---")
    DATA_FILE = "data/BTC_USDT_1d_with_indicators.csv"
    # OUTPUT_SINGLE_DIR = "output/single_chart"
    # OUTPUT_SINGLE_FILE = os.path.join(OUTPUT_SINGLE_DIR, "latest_120_days.png")
    # try:
    #     df_full = pd.read_csv(DATA_FILE)
    #     # Assume time column is 'date' or similar name and already sorted
    #     date_col_name = visualizer._find_date_column(df_full)
    #     if date_col_name:
    #         df_full[date_col_name] = pd.to_datetime(df_full[date_col_name])
    #         df_full = df_full.sort_values(by=date_col_name)
    #     else:
    #         print("Warning: Date column not found, will use last 120 rows of data.")

    #     df_subset = df_full.tail(120)
    #     if not df_subset.empty:
    #         visualizer.visualize(df_subset, OUTPUT_SINGLE_FILE, show=False)
    #     else:
    #         print("Unable to get data for single chart example.")

    # except FileNotFoundError:
    #     print(f"Error: Data file '{DATA_FILE}' not found. Skipping Example 1.")
    # except Exception as e:
    #     print(f"Error running Example 1: {e}")

    # --- Example 2: Batch generate charts ---
    print("\n--- Example 2: Batch Generate Charts ---")
    OUTPUT_BATCH_DIR = "data/btc_daily_refactored"
    try:
        visualizer.batch_generate(
            data_path=DATA_FILE,
            output_dir=OUTPUT_BATCH_DIR,
            length=120,
            start_time="2021-06-30",
            end_time="2021-12-31",
            step=1,  # Generate a chart every day
            show=False,
            filename_prefix="btc_daily",
        )
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found. Skipping Example 2.")
    except Exception as e:
        print(f"Error running Example 2: {e}")
