import pandas as pd
import vectorbt as vbt


class BaseStrategy:
    """Base class for backtesting strategies."""

    vbt.settings.portfolio["init_cash"] = 10000.0  # 10000$
    vbt.settings.portfolio["fees"] = 0.001  # 0.1%
    vbt.settings.portfolio["slippage"] = 0.0025  # 0.25%

    def __init__(self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC"): # 更新默认日期范围
        self.symbol = symbol
        self.start = start
        self.end = end
        self.price = vbt.YFData.download(self.symbol, start=self.start, end=self.end).get("Close")
        if self.price is None or self.price.empty:
            print(f"警告: 未能加载 {self.symbol} 从 {self.start} 到 {self.end} 的价格数据。")
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
        entries_np = self.entries.values.astype(bool) if isinstance(self.entries, pd.Series) else self.entries.astype(bool)
        exits_np = self.exits.values.astype(bool) if isinstance(self.exits, pd.Series) else self.exits.astype(bool)
        
        pf = vbt.Portfolio.from_signals(self.price, entries_np, exits_np)
        return pf


class MaCross(BaseStrategy):
    """A strategy that uses two moving averages to generate buy and sell signals."""

    def __init__(self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC", fast_window=5, slow_window=10): # 更新默认日期范围
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

    def __init__(self, symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC"): # 更新默认日期范围
        super().__init__(symbol, start, end)
        # 计算MACD
        macd_indicator = vbt.MACD.run(self.price)
        self.macd_line = macd_indicator.macd
        self.signal_line = macd_indicator.signal
        self.init_entries_exits()

    def init_entries_exits(self):
        self.entries = self.detect_bullish_divergence()
        self.exits = self.detect_bearish_divergence() # 顶背离作为卖出信号

    def detect_bullish_divergence(self):
        # 简单检测底背离：价格创新低但MACD未创新低
        price_vals = self.price.values
        macd_vals = self.macd_line.values
        entries = [False] * len(price_vals)
        for i in range(2, len(price_vals)):
            # 价格创新低
            if price_vals[i] < price_vals[i - 1] and price_vals[i - 1] < price_vals[i - 2]:
                # MACD未创新低
                if macd_vals[i] > macd_vals[i - 1] and macd_vals[i - 1] < macd_vals[i - 2]:
                    entries[i] = True
        return pd.Series(entries, index=self.price.index)

    def detect_bearish_divergence(self):
        """检测顶背离：价格创新高但MACD未创新高，产生卖出信号."""
        price_vals = self.price.values
        macd_vals = self.macd_line.values
        exits = [False] * len(price_vals)
        for i in range(2, len(price_vals)):
            # 价格创新高
            if price_vals[i] > price_vals[i - 1] and price_vals[i - 1] > price_vals[i - 2]:
                # MACD未创新高
                if macd_vals[i] < macd_vals[i - 1] and macd_vals[i - 1] > macd_vals[i - 2]:
                    exits[i] = True
        return pd.Series(exits, index=self.price.index)


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
    macd_div = MacdBullishDivergence(symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC") # 明确传递日期
    print(f"loading data took {time.time() - start_time_macd:.4f} seconds")
    if macd_div.price is not None and not macd_div.price.empty:
        pf_macd = macd_div.run()
        print(pf_macd.stats())
        print("\n--- MACD策略交易订单 ---")
        print(pf_macd.orders)
        pf_macd.plot().show()
    else:
        print("由于未能加载价格数据，无法运行MACD策略回测。")
    end_time_macd = time.time()
    print(f"MACD底背离策略执行时间: {end_time_macd - start_time_macd:.4f} seconds")
