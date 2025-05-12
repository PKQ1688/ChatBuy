import pandas as pd
import vectorbt as vbt


class MaCross:
    """A strategy that uses two moving averages to generate buy and sell signals."""

    vbt.settings.portfolio["init_cash"] = 10000.0  # 10000$
    vbt.settings.portfolio["fees"] = 0.001  # 0.1%
    vbt.settings.portfolio["slippage"] = 0.0025  # 0.25%

    def __init__(self):
        start = "2017-01-01 UTC"  # crypto is in UTC
        end = "2025-05-12 UTC"
        self.price = vbt.YFData.download("BTC-USD", start=start, end=end).get("Close")

        self.fast_ma = vbt.MA.run(self.price, 5, short_name="fast")
        self.slow_ma = vbt.MA.run(self.price, 10, short_name="slow")

        self.init_entries_exits()

    def init_entries_exits(self):
        """Initialize entries and exits for the strategy."""
        self.entries = self.fast_ma.ma_crossed_above(self.slow_ma)
        self.exits = self.fast_ma.ma_crossed_below(self.slow_ma)

    def run(self):
        pf = vbt.Portfolio.from_signals(self.price, self.entries, self.exits)
        return pf


class MacdBullishDivergence:
    """MACD底背离策略：价格创新低但MACD未创新低，产生买入信号."""

    vbt.settings.portfolio["init_cash"] = 10000.0
    vbt.settings.portfolio["fees"] = 0.001
    vbt.settings.portfolio["slippage"] = 0.0025

    def __init__(self):
        start = "2017-01-01 UTC"  # crypto is in UTC
        end = "2025-05-12 UTC"
        self.price = vbt.YFData.download("BTC-USD", start=start, end=end).get("Close")

        # 计算MACD
        macd = vbt.MACD.run(self.price)
        self.macd = macd.macd
        self.signal = macd.signal

        self.entries = self.detect_bullish_divergence()
        self.exits = self.macd < self.signal  # MACD死叉作为卖出

    def detect_bullish_divergence(self):
        # 简单检测底背离：价格创新低但MACD未创新低
        price = self.price.values
        macd = self.macd.values
        entries = [False] * len(price)
        for i in range(2, len(price)):
            # 价格创新低
            if price[i] < price[i - 1] and price[i - 1] < price[i - 2]:
                # MACD未创新低
                if macd[i] > macd[i - 1] and macd[i - 1] < macd[i - 2]:
                    entries[i] = True
        return pd.Series(entries, index=self.price.index)

    def run(self):
        pf = vbt.Portfolio.from_signals(self.price, self.entries, self.exits)
        return pf


if __name__ == "__main__":
    import time

    # start_time = time.time()

    # ma_cross = MaCross()
    # print(f"loading data took {time.time() - start_time:.4f} seconds")

    # pf = ma_cross.run()
    # # print(pf.stats())
    # # pf.plot().show()

    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time:.4f} seconds")

    # 运行MACD底背离策略
    print("\n--- MACD底背离策略回测 ---")
    start_time = time.time()
    macd_div = MacdBullishDivergence()
    print(f"loading data took {time.time() - start_time:.4f} seconds")
    pf_macd = macd_div.run()
    print(pf_macd.stats())

    breakpoint()
    pf_macd.plot().show()
    end_time = time.time()
    print(f"MACD底背离策略执行时间: {end_time - start_time:.4f} seconds")
