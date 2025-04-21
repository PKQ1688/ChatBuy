import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc
from typing import List, Optional, Tuple


class IndicatorVisualizer:
    """
    用于可视化金融时间序列数据及其技术指标（如布林带、MACD）的类。
    提供单个图表生成和批量生成功能。
    """

    DEFAULT_POSSIBLE_DATE_COLUMNS: List[str] = [
        "date",
        "time",
        "timestamp",
        "Date",
        "Time",
        "Timestamp",
        "datetime",
        "Datetime",
    ]
    REQUIRED_INDICATOR_COLUMNS: List[str] = [
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
    OHLC_COLUMNS: List[str] = ["open", "high", "low", "close"]
    BB_COLUMNS: List[str] = ["bb_upper", "bb_middle", "bb_lower"]
    MACD_COLUMNS: List[str] = ["macd", "signal", "histogram"]

    def __init__(self, possible_date_columns: Optional[List[str]] = None):
        """
        初始化 IndicatorVisualizer。

        Args:
            possible_date_columns (Optional[List[str]]): 可能包含日期/时间信息的列名列表。
                                                        如果为 None，则使用默认列表。
        """
        self.possible_date_columns = (
            possible_date_columns or self.DEFAULT_POSSIBLE_DATE_COLUMNS
        )

    def _find_date_column(self, data: pd.DataFrame) -> Optional[str]:
        """尝试在 DataFrame 中查找日期/时间列。"""
        for col in self.possible_date_columns:
            if col in data.columns:
                return col
        return None

    def _prepare_data_and_xaxis(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[str], np.ndarray, pd.Series, str]:
        """准备绘图数据和 X 轴信息。"""
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
        self, date_column: Optional[str]
    ) -> Tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
        """设置 matplotlib 图表和子图。"""
        fig = plt.figure(figsize=(14, 12))
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)  # Price and BB
        ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)  # Volume
        ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)  # MACD

        # 隐藏 ax1 和 ax2 的 X 轴标签和刻度，因为它们共享 ax3 的 X 轴
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax1.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax2.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if date_column:
            # 初始设置，后面会在 _format_xaxis 中进一步调整
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        return fig, ax1, ax2, ax3

    def _plot_price_and_bb(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        date_num: np.ndarray,
        x_values: pd.Series,
    ):
        """绘制价格（K线或收盘价线）和布林带。"""
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
        """绘制成交量柱状图。"""
        if "volume" in data.columns:
            ax.bar(
                x_values, data["volume"], color="blue", alpha=0.5, width=0.8
            )  # 使用 width 参数调整柱子宽度
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
        """绘制 MACD 指标。"""
        if all(col in data.columns for col in self.MACD_COLUMNS):
            ax.plot(x_values, data["macd"], label="MACD", color="blue", linewidth=1.2)
            ax.plot(
                x_values, data["signal"], label="Signal", color="red", linewidth=1.2
            )
            colors = ["green" if val >= 0 else "red" for val in data["histogram"]]
            ax.bar(
                x_values, data["histogram"], color=colors, alpha=0.5, width=0.8
            )  # 使用 width 参数
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
        date_column: Optional[str],
        date_num: np.ndarray,
        x_values: pd.Series,
        x_label: str,
    ):
        """格式化 X 轴标签和刻度。"""
        ax.set_xlabel(x_label)

        # 尝试获取合理的刻度数量
        num_ticks = 10  # 可以根据需要调整
        locator = plt.MaxNLocator(nbins=num_ticks, prune="both")  # 'both' 避免边缘刻度
        ax.xaxis.set_major_locator(locator)

        if date_column:
            # 确保使用 DateFormatter
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            # 自动旋转标签以避免重叠
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        else:
            # 对于非日期轴，确保标签是整数
            def format_func(value, tick_number):
                return f"{int(value)}"

            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        # 强制显示第一个和最后一个刻度标签（如果它们不在自动生成的刻度中）
        # 注意：这可能会导致标签过于密集，谨慎使用或寻找更好的策略
        # current_ticks = ax.get_xticks()
        # first_val = x_values.iloc[0]
        # last_val = x_values.iloc[-1]
        # Ensure first/last logic might need refinement depending on locator behavior

    def visualize(self, data: pd.DataFrame, output_file_path: str, show: bool = False):
        """
        根据传入的 DataFrame 生成一张 K 线+指标图片，并保存到指定路径。

        Args:
            data (pd.DataFrame): 包含价格和指标数据的 DataFrame。
            output_file_path (str): 图片输出的完整文件路径。
            show (bool): 是否在生成后显示图片。

        Returns:
            str: 输出文件的路径。

        Raises:
            ValueError: 如果无法创建输出目录。
        """
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"无法创建输出目录 '{output_dir}': {e}")

        # 检查缺失的列（可选，但有助于调试）
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 调整布局防止标题和标签重叠

        try:
            plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving chart to {output_file_path}: {e}")
            plt.close(fig)  # Ensure figure is closed even on save error
            return None  # Indicate failure

        if show:
            plt.show()

        plt.close(fig)  # 关闭图形以释放内存

        return output_file_path

    def batch_generate(
        self,
        data_path: str,
        output_dir: str,
        length: int = 120,
        end_time: Optional[str] = None,
        start_time: Optional[str] = None,
        step: int = 1,
        show: bool = False,
        filename_prefix: str = "chart",
    ):
        """
        批量生成图表，使用滑动窗口处理时间序列数据。

        Args:
            data_path (str): CSV 数据文件路径。
            output_dir (str): 图片输出目录。
            length (int): 每个图表包含的数据点数量（窗口长度）。
            end_time (Optional[str]): 数据筛选的结束时间 ('YYYY-MM-DD')。None 表示使用数据末尾。
            start_time (Optional[str]): 数据筛选的开始时间 ('YYYY-MM-DD')。None 表示使用数据开头。
            step (int): 滑动窗口的步长。
            show (bool): 是否显示每个生成的图表。
            filename_prefix (str): 生成的图片文件名前缀。

        Raises:
            ValueError: 如果找不到时间列或指定的时间范围无效。
            FileNotFoundError: 如果数据文件不存在。
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"无法读取或解析 CSV 文件 '{data_path}': {e}")

        date_column = self._find_date_column(df)
        if date_column is None:
            raise ValueError(
                f"在 {data_path} 中未找到可识别的时间列 (检查列名是否在 {self.possible_date_columns} 中)。无法按时间筛选。"
            )

        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"无法将列 '{date_column}' 转换为日期时间格式: {e}")

        df = df.sort_values(by=date_column).reset_index(drop=True)  # 确保按时间排序

        # 确定数据处理的起始和结束索引
        start_idx = 0
        if start_time:
            start_time_dt = pd.to_datetime(start_time)
            # 找到第一个大于等于 start_time 的索引
            start_indices = df.index[df[date_column] >= start_time_dt]
            if not start_indices.empty:
                start_idx = start_indices.min()
            else:
                raise ValueError(
                    f"指定的起始时间 '{start_time}' 晚于数据中的所有日期。"
                )

        end_idx = len(df)
        if end_time:
            end_time_dt = pd.to_datetime(end_time)
            # 找到最后一个小于等于 end_time 的索引 + 1 作为切片终点
            end_indices = df.index[df[date_column] <= end_time_dt]
            if not end_indices.empty:
                end_idx = end_indices.max() + 1  # 加 1 使其包含 end_time 当天的数据
            else:
                raise ValueError(f"指定的结束时间 '{end_time}' 早于数据中的所有日期。")

        if start_idx >= end_idx:
            raise ValueError(
                f"计算出的起始索引 ({start_idx}) 大于或等于结束索引 ({end_idx})。请检查时间范围。"
            )

        print(
            f"开始批量生成图片：从索引 {start_idx} 到 {end_idx - 1}，窗口长度 {length}，步长 {step}"
        )
        generated_count = 0
        # 滑动窗口生成图片
        for i in range(start_idx, end_idx - length + 1, step):
            sub_df = df.iloc[i : i + length]
            if len(sub_df) < length:  # 理论上不应发生，除非 end_idx 计算错误
                print(
                    f"Skipping index {i}: Not enough data points ({len(sub_df)} < {length})"
                )
                continue

            # 构建输出文件名
            window_end_date = sub_df[date_column].iloc[-1]
            # 格式化日期，确保文件名合法
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

        print(f"批量生成完成，共生成 {generated_count} 张图片。")


if __name__ == "__main__":
    visualizer = IndicatorVisualizer()

    # --- 示例 1：生成单张图表 (使用最近 120 条数据) ---
    print("\n--- 示例 1: 生成单张图表 ---")
    DATA_FILE = "data/BTC_USDT_1d_with_indicators.csv"
    OUTPUT_SINGLE_DIR = "output/single_chart"
    OUTPUT_SINGLE_FILE = os.path.join(OUTPUT_SINGLE_DIR, "latest_120_days.png")
    try:
        df_full = pd.read_csv(DATA_FILE)
        # 假设时间列是 'date' 或类似名称，并已排序
        date_col_name = visualizer._find_date_column(df_full)
        if date_col_name:
            df_full[date_col_name] = pd.to_datetime(df_full[date_col_name])
            df_full = df_full.sort_values(by=date_col_name)
        else:
            print("Warning: 未找到日期列，将使用最后 120 行数据。")

        df_subset = df_full.tail(120)
        if not df_subset.empty:
            visualizer.visualize(df_subset, OUTPUT_SINGLE_FILE, show=False)
        else:
            print("无法获取用于单张图表示例的数据。")

    except FileNotFoundError:
        print(f"错误：数据文件 '{DATA_FILE}' 未找到。跳过示例 1。")
    except Exception as e:
        print(f"运行示例 1 时出错: {e}")

    # --- 示例 2：批量生成图表 ---
    print("\n--- 示例 2: 批量生成图表 ---")
    OUTPUT_BATCH_DIR = "data/btc_daily_refactored"
    try:
        visualizer.batch_generate(
            data_path=DATA_FILE,
            output_dir=OUTPUT_BATCH_DIR,
            length=120,
            start_time="2021-06-30",
            end_time="2021-12-31",
            step=1,  # 每隔 30 天生成一张图
            show=False,
            filename_prefix="btc_daily",
        )
    except FileNotFoundError:
        print(f"错误：数据文件 '{DATA_FILE}' 未找到。跳过示例 2。")
    except Exception as e:
        print(f"运行示例 2 时出错: {e}")
