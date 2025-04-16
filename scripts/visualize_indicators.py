import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc


def visualize_btc_with_indicators(  # noqa: C901
    data: pd.DataFrame,
    output_dir="data",
    output_file_prefix="BTC_indicators",
    show=True,
):
    """根据传入的DataFrame生成一张K线+指标图片.

    Parameters:
    - data: pd.DataFrame，已筛选好时间区间的数据
    - output_dir: 图片输出目录
    - output_file_prefix: 图片文件名前缀
    - show: 是否显示图片
    """
    os.makedirs(output_dir, exist_ok=True)

    # 自动识别时间列
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

    if date_column is None:
        data["index"] = np.arange(len(data))
        x_values = data["index"]
        x_label = "Data Points"
        date_num = data["index"].values
    else:
        data[date_column] = pd.to_datetime(data[date_column])
        x_values = data[date_column]
        x_label = "Date"
        date_num = mdates.date2num(data[date_column].values)

    plt.figure(figsize=(14, 12))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0), sharex=ax1)
    ax3 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    if date_column is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

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

    ohlc_columns = ["open", "high", "low", "close"]
    if all(col in data.columns for col in ohlc_columns):
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
        candlestick_ohlc(
            ax1, ohlc, width=0.6, colorup="green", colordown="red", alpha=0.8
        )
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
        ax1.plot(
            x_values, data["close"], label="BTC Price", color="black", linewidth=1.5
        )
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

    if all(col in data.columns for col in ["macd", "signal", "histogram"]):
        ax3.plot(x_values, data["macd"], label="MACD", color="blue", linewidth=1.2)
        ax3.plot(x_values, data["signal"], label="Signal", color="red", linewidth=1.2)
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

    plt.xlabel(x_label)
    if date_column is not None:
        # 获取当前xticks并转为list
        ticks = list(ax1.get_xticks())
        # 保证首尾日期一定在xticks中
        first = date_num[0]
        last = date_num[-1]
        if first not in ticks:
            ticks = [first] + ticks
        if last not in ticks:
            ticks = ticks + [last]
        # 只保留在数据范围内的刻度
        ticks = [t for t in ticks if first <= t <= last]
        # 设置xticks和对应标签
        ax1.set_xticks(ticks)
        labels = [mdates.num2date(t).strftime("%Y-%m-%d") for t in ticks]
        ax1.set_xticklabels(labels, rotation=45)
    else:
        # 非时间索引，强制首尾索引显示
        ticks = list(ax1.get_xticks())
        first = x_values.iloc[0]
        last = x_values.iloc[-1]
        if first not in ticks:
            ticks = [first] + ticks
        if last not in ticks:
            ticks = ticks + [last]
        ticks = [t for t in ticks if first <= t <= last]
        ax1.set_xticks(ticks)
        labels = [str(int(t)) for t in ticks]
        ax1.set_xticklabels(labels, rotation=45)
    plt.tight_layout()

    # 文件名带上时间区间
    if date_column is not None:
        start_str = data[date_column].iloc[0].strftime("%Y%m%d")
        end_str = data[date_column].iloc[-1].strftime("%Y%m%d")
        output_file = os.path.join(
            output_dir, f"{output_file_prefix}_{start_str}_{end_str}.png"
        )
    else:
        output_file = os.path.join(
            output_dir,
            f"{output_file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Chart saved to: {output_file}")
    if show:
        plt.show()
    plt.close()


def batch_generate_images_with_time_range(
    data_path="data/BTC_USDT_1d_with_indicators.csv",
    output_dir="data",
    length=120,
    end_time=None,
    start_time=None,
    step=1,
    show=False,
):
    """批量生成图片，支持指定时间长度、起始时间和结束时间节点.

    Parameters:
    - data_path: 数据文件路径
    - output_dir: 图片输出目录
    - length: 每张图片包含的数据条数
    - end_time: 结束时间（字符串'YYYY-MM-DD'或None，None则为数据最后一天）
    - start_time: 起始时间（字符串'YYYY-MM-DD'或None，None则为数据最早一天）
    - step: 滑动窗口步长
    - show: 是否显示图片
    """
    df = pd.read_csv(data_path)
    # 自动识别时间列
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
        if col in df.columns:
            date_column = col
            break
    if date_column is None:
        raise ValueError("未找到时间列，无法按时间筛选！")

    df[date_column] = pd.to_datetime(df[date_column])

    # 确定结束时间
    if end_time is None:
        end_idx = len(df)
    else:
        end_time_dt = pd.to_datetime(end_time)
        end_idx = df[df[date_column] <= end_time_dt].shape[0]
        if end_idx == 0:
            raise ValueError("指定的结束时间早于数据最早时间！")

    # 确定起始时间
    if start_time is None:
        start_idx = 0
    else:
        start_time_dt = pd.to_datetime(start_time)
        start_idx = df[df[date_column] >= start_time_dt].index.min()
        if pd.isna(start_idx):
            raise ValueError("指定的起始时间晚于数据最晚时间！")
        start_idx = int(start_idx)

    # 批量滑动窗口生成图片
    for i in range(start_idx, end_idx - length + 1, step):
        sub_df = df.iloc[i : i + length].copy()
        if len(sub_df) < length:
            continue
        visualize_btc_with_indicators(
            sub_df,
            output_dir=output_dir,
            output_file_prefix=f"coin_{length}",
            show=show,
        )


if __name__ == "__main__":
    # 示例：生成最后120天的图片
    batch_generate_images_with_time_range(
        data_path="data/BTC_USDT_1d_with_indicators.csv",
        output_dir="data/btc_daily",
        length=120,
        start_time="2021-6-30",  # 或如 '2021-01-01'
        end_time="2021-12-31",  # 或如 '2021-12-31'
        step=1,
        show=False,
    )
