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
        data = vbt.YFData.download(self.symbol, start=self.start, end=self.end)
        self.price = data.get("Close")
        
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


def optimize_ma_params(symbol="BTC-USD", start="2022-01-01 UTC", end="2024-01-01 UTC", fast_range=range(3, 20, 2), slow_range=range(10, 50, 5), metric="Total Return"):
    """遍历不同均线参数，返回表现最好的参数组合和结果."""
    results = []
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue  # 快线必须小于慢线
            ma_cross = MaCross(symbol=symbol, start=start, end=end, fast_window=fast, slow_window=slow)
            pf = ma_cross.run()
            stats = pf.stats()
            results.append({
                "fast": fast,
                "slow": slow,
                "stats": stats
            })
    # 选出最佳参数
    best = max(results, key=lambda x: x["stats"][metric])
    print(f"最佳参数: fast={best['fast']}, slow={best['slow']}, {metric}={best['stats'][metric]:.2f}")
    
    # 打印一次 stats 的 keys 以便用户选择正确的 metric
    if results:
        print("可用的评估指标:", list(results[0]["stats"].index))
        
    return best


if __name__ == "__main__":
    print("\n--- MA Cross参数优化 ---")
    best = optimize_ma_params(
        symbol="BTC-USD",
        start="2023-01-01 UTC",
        end="2025-01-01 UTC",
        fast_range=range(3, 20, 2),
        slow_range=range(10, 30, 2),
        metric="Total Return [%]"  # 你可以改为 "Sharpe Ratio" 等
    )
    print("\n--- 最佳参数回测结果 ---")
    ma_cross = MaCross(
        symbol="BTC-USD",
        start="2023-01-01 UTC",
        end="2025-01-01 UTC",
        fast_window=best["fast"],
        slow_window=best["slow"],
    )
    pf_ma = ma_cross.run()
    print(pf_ma.stats())
    trades = pf_ma.trades.records_readable
    trades_subset = trades[
        [
            "Entry Timestamp",
            "Avg Entry Price",
            "Exit Timestamp",
            "Avg Exit Price",
            "PnL",
            "Return",
            "Direction",
        ]
    ]
    print(trades_subset)
    pf_ma.plot().show()
