import pandas as pd
import vectorbt as vbt


class BaseStrategy:
    """Base class for backtesting strategies."""

    vbt.settings.portfolio["init_cash"] = 10000.0  # 10000$
    vbt.settings.portfolio["fees"] = 0.001  # 0.1%
    vbt.settings.portfolio["slippage"] = 0.0025  # 0.25%

    def __init__(
        self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC"
    ):  # 更新默认日期范围
        self.symbol = symbol
        self.start = start
        self.end = end
        self.price = vbt.YFData.download(
            self.symbol, start=self.start, end=self.end
        ).get("Close")
        if self.price is None or self.price.empty:
            print(
                f"警告: 未能加载 {self.symbol} 从 {self.start} 到 {self.end} 的价格数据。"
            )
            # 考虑是否在此处引发异常或允许程序继续（可能会在后续步骤失败）
        self.entries = None
        self.exits = None

    def init_entries_exits(self):
        """Initialize entries and exits for the strategy. This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement init_entries_exits")

    def run(self):
        """Run the backtest and return the portfolio."""
        if self.entries is None or self.exits is None:
            self.init_entries_exits()

        # 确保 entries 和 exits 是布尔类型的 NumPy 数组
        entries_np = (
            self.entries.values.astype(bool)
            if isinstance(self.entries, pd.Series)
            else self.entries.astype(bool)
        )
        exits_np = (
            self.exits.values.astype(bool)
            if isinstance(self.exits, pd.Series)
            else self.exits.astype(bool)
        )

        pf = vbt.Portfolio.from_signals(self.price, entries_np, exits_np)
        return pf


class MaCross(BaseStrategy):
    """A strategy that uses two moving averages to generate buy and sell signals."""

    def __init__(
        self,
        symbol="BTC-USD",
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        fast_window=5,
        slow_window=10,
    ):  # 更新默认日期范围
        super().__init__(symbol, start, end)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.fast_ma = vbt.MA.run(self.price, self.fast_window, short_name="fast")
        self.slow_ma = vbt.MA.run(self.price, self.slow_window, short_name="slow")
        self.init_entries_exits()

    def init_entries_exits(self):
        """Initialize entries and exits for the strategy."""
        self.entries = self.fast_ma.ma_crossed_above(self.slow_ma)
        self.exits = self.fast_ma.ma_crossed_below(self.slow_ma)


class MacdBullishDivergence(BaseStrategy):
    """MACD底背离策略：价格创新低但MACD未创新低，产生买入信号."""

    def __init__(
        self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC"
    ):  # 更新默认日期范围
        super().__init__(symbol, start, end)
        # 计算MACD
        macd_indicator = vbt.MACD.run(self.price)
        self.macd_line = macd_indicator.macd
        self.signal_line = macd_indicator.signal
        self.macd_hist = macd_indicator.hist  # 红绿柱子
        self.init_entries_exits()

    def init_entries_exits(self):
        self.entries = self.detect_macd_weaker_cross(buy=True)
        self.exits = self.detect_macd_weaker_cross(buy=False)

    def detect_macd_weaker_cross(self, buy=True):
        # buy=True: 死叉柱子更弱后等金叉买入；buy=False: 金叉柱子更强后等死叉卖出
        macd = self.macd_line.values
        signal = self.signal_line.values
        hist = self.macd_hist.values  # 红绿柱子
        n = len(macd)
        cross_idx = []
        cross_hist = []

        # 找所有死叉（金叉）点
        for i in range(1, n):
            if buy:
                # 死叉: macd从上穿到下
                if macd[i-1] > signal[i-1] and macd[i] <= signal[i]:
                    cross_idx.append(i)
                    cross_hist.append(hist[i])
            else:
                # 金叉: macd从下穿到上
                if macd[i-1] < signal[i-1] and macd[i] >= signal[i]:
                    cross_idx.append(i)
                    cross_hist.append(hist[i])

        # 检查本次死叉（金叉）柱子是否比上一次更弱（更低/更高）
        signal_points = []
        for j in range(1, len(cross_idx)):
            if buy:
                # 死叉柱子更弱（更低）
                if cross_hist[j] < cross_hist[j-1]:
                    # 在下一个金叉买入
                    for k in range(cross_idx[j], n):
                        if macd[k-1] < signal[k-1] and macd[k] >= signal[k]:
                            signal_points.append(k)
                            break
            else:
                # 金叉柱子更强（更高）
                if cross_hist[j] > cross_hist[j-1]:
                    # 在下一个死叉卖出
                    for k in range(cross_idx[j], n):
                        if macd[k-1] > signal[k-1] and macd[k] <= signal[k]:
                            signal_points.append(k)
                            break

        result = [False] * n
        for idx in signal_points:
            if idx < n:
                result[idx] = True
        return pd.Series(result, index=self.price.index)


if __name__ == "__main__":
    import time

    # print("\n--- MA Cross策略回测 ---")
    # start_time_ma = time.time()
    # ma_cross = MaCross()
    # print(f"loading data took {time.time() - start_time_ma:.4f} seconds")
    # pf_ma = ma_cross.run()
    # print(pf_ma.stats())
    # # pf_ma.plot().show()
    # end_time_ma = time.time()
    # print(f"MA Cross策略执行时间: {end_time_ma - start_time_ma:.4f} seconds")

    # 运行MACD底背离策略
    print("\n--- MACD底背离策略回测 ---")
    start_time_macd = time.time()
    macd_div = MacdBullishDivergence(
        symbol="BTC-USD", start="2023-01-01 UTC", end="2025-06-01 UTC"
    )  # 明确传递日期
    print(f"loading data took {time.time() - start_time_macd:.4f} seconds")
    if macd_div.price is not None and not macd_div.price.empty:
        pf_macd = macd_div.run()
        print(pf_macd.stats())
        print("\n--- MACD策略交易订单 ---")
        print(pf_macd.orders.records)
        # 打印每笔订单的日期、买卖方向和价格
        orders = pf_macd.orders.records
        price_index = macd_div.price.index
        print("\n--- MACD策略订单详细（含日期） ---")
        for _, row in orders.iterrows():
            dt = price_index[int(row["idx"])]
            side = "买入" if row["side"] == 0 else "卖出"
            print(
                f"{side} 日期: {dt.strftime('%Y-%m-%d %H:%M:%S')}, 价格: {row['price']:.2f}, 数量: {row['size']:.6f}"
            )

        pf_macd.plot().show()
    else:
        print("由于未能加载价格数据，无法运行MACD策略回测。")

    end_time_macd = time.time()
    print(f"MACD底背离策略执行时间: {end_time_macd - start_time_macd:.4f} seconds")
