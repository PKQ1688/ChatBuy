import matplotlib.pyplot as plt
import pandas as pd
import vectorbt as vbt


class RotationStrategy:
    """资产轮动策略：基于历史表现在不同资产间进行选择.

    该策略基于BTC和ETH在过去20天的表现进行切换，只投资于表现较好且回报为正的加密货币.
    若两者回报均为负，则持有现金.
    """

    # 投资组合设置
    vbt.settings.portfolio["init_cash"] = 10000.0
    vbt.settings.portfolio["fees"] = 0.001
    vbt.settings.portfolio["slippage"] = 0.0025

    def __init__(
        self,
        symbols=["BTC-USD", "ETH-USD"],
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        lookback_period=20,
    ):
        """初始化轮动策略.

        参数:
            symbols: 轮动资产列表（默认：BTC-USD和ETH-USD）
            start: 回测起始日期
            end: 回测结束日期
            lookback_period: 计算表现的历史周期（天数）
        """
        # 基础设置
        self.symbols = symbols
        self.start = start
        self.end = end
        self.lookback_period = lookback_period
        self.portfolio = None

        # 数据准备与信号初始化
        self._prepare_price_data()
        self._initialize_signals()
        self.init_entries_exits()

    def _initialize_signals(self):
        """初始化交易信号."""
        # 为每个资产创建空的信号Series
        self.entries = {
            symbol: pd.Series(False, index=self.common_index) for symbol in self.symbols
        }
        self.exits = {
            symbol: pd.Series(False, index=self.common_index) for symbol in self.symbols
        }

    def _prepare_price_data(self):
        """获取并对齐所有资产的价格数据."""
        # 并行下载所有资产的价格数据
        self.data = vbt.YFData.download(self.symbols, start=self.start, end=self.end)

        # 提取收盘价并确保所有数据使用相同索引
        self.prices = self.data.get("Close")
        self.common_index = self.prices.index

        # 检查并处理缺失值
        if self.prices.isna().any().any():
            self.prices = self.prices.dropna(how="any")

    def calculate_returns(self):
        """计算每个资产在历史周期内的回报率."""
        # 直接使用DataFrame的pct_change方法
        return self.prices.pct_change(self.lookback_period).fillna(0)

    def init_entries_exits(self):
        """根据轮动策略初始化买入卖出信号（整体轮动，仅持有一个币或空仓）."""
        returns_df = self.calculate_returns()
        best_assets = returns_df.idxmax(axis=1)
        best_returns = returns_df.max(axis=1)
        trading_dates = returns_df.index[self.lookback_period :]
        # 构建整体持仓DataFrame
        self.positions = pd.DataFrame(
            False, index=self.common_index, columns=self.symbols
        )
        for date in trading_dates:
            best_symbol = best_assets[date]
            best_return = best_returns[date]
            if best_return > 0:
                # 只持有表现最好的币
                self.positions.loc[date, :] = False
                self.positions.loc[date, best_symbol] = True
            else:
                # 空仓
                self.positions.loc[date, :] = False
        # 生成entries/exits信号（用于可视化等）
        self.entries = {
            symbol: (
                self.positions[symbol]
                & ~self.positions[symbol].shift(1, fill_value=False)
            )
            for symbol in self.symbols
        }
        self.exits = {
            symbol: (
                ~self.positions[symbol]
                & self.positions[symbol].shift(1, fill_value=False)
            )
            for symbol in self.symbols
        }

    def run(self):
        """执行回测并返回整体轮动投资组合."""
        # 用整体持仓矩阵生成Portfolio
        self.portfolio = vbt.Portfolio.from_signals(
            self.prices,
            entries=self.positions.shift(1, fill_value=False),  # 避免未来函数
            exits=~self.positions.shift(1, fill_value=False),
            init_cash=vbt.settings.portfolio["init_cash"],
            fees=vbt.settings.portfolio["fees"],
            slippage=vbt.settings.portfolio["slippage"],
        )
        return self.portfolio

    def visualize(self):
        """可视化投资组合表现."""
        if self.portfolio is None:
            self.run()

        # 创建图表：两个子图 - 收益率对比和回撤
        fig, axes = plt.subplots(2, 1, figsize=(14, 14))

        # 1. 绘制收益率对比图 - 策略vs原始资产
        # 计算策略的累计收益率曲线
        strategy_returns = (
            self.portfolio.value() / vbt.settings.portfolio["init_cash"]
        ) - 1

        # 计算原始资产的累计收益率曲线（假设从起点开始买入持有）
        asset_returns = {}
        for symbol in self.symbols:
            normalized = self.prices[symbol] / self.prices[symbol].iloc[0] - 1
            asset_returns[symbol] = normalized

        # 绘制累计收益率对比
        axes[0].plot(
            strategy_returns.index,
            strategy_returns.values * 100,
            "b-",
            linewidth=2,
            label="Rotation Strategy",
        )
        for symbol in self.symbols:
            axes[0].plot(
                asset_returns[symbol].index,
                asset_returns[symbol].values * 100,
                "--",
                linewidth=1.5,
                label=f"{symbol} Buy & Hold",
            )

        axes[0].set_title("Cumulative Returns: Strategy vs Assets", fontsize=14)
        axes[0].set_ylabel("Return (%)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper left")
        axes[0].axhline(y=0, color="k", linestyle="-", alpha=0.2)

        # 2. 绘制回撤
        # 获取总体投资组合drawdown - 对全部列计算平均值
        drawdown = self.portfolio.drawdown().mean(axis=1)  # 对所有资产的drawdown取平均
        # 保证x/y长度一致，避免fill_between报错
        dd_index = drawdown.index
        dd_values = -drawdown.values * 100  # 转为正值百分比
        axes[1].fill_between(dd_index, 0, dd_values, color="r", alpha=0.5)
        axes[1].set_title("Rotation Strategy: Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)

        # 添加投资组合统计概要
        stats = self.get_stats()

        # 给主要资产计算买入持有收益
        buyhold_returns = {}
        for symbol in self.symbols:
            first_price = self.prices[symbol].iloc[0]
            last_price = self.prices[symbol].iloc[-1]
            buyhold_returns[symbol] = (last_price / first_price - 1) * 100

        # 创建统计摘要文字
        summary_text = (
            f"Strategy Total Return: {stats['Total Return [%]']:.2f}%\n"
            f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%\n"
            f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n\n"
        )

        # 添加每个资产的买入持有收益
        for symbol in self.symbols:
            summary_text += (
                f"{symbol} Buy & Hold Return: {buyhold_returns[symbol]:.2f}%\n"
            )

        fig.text(0.01, 0.01, summary_text, fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # 为底部文本留出空间
        plt.show()

    def get_stats(self):
        """返回投资组合统计数据."""
        if self.portfolio is None:
            self.run()

        # 获取整体统计数据
        return self.portfolio.stats(metrics=None)

    def run_and_visualize(self):
        """运行策略并可视化结果."""
        self.run()

        # 输出统计结果
        print("\n=== Rotation Strategy Results ===")
        stats = self.get_stats()
        important_metrics = [
            "Total Return [%]",
            "Max Drawdown [%]",
            "Sharpe Ratio",
            "Win Rate [%]",
            "Profit Factor",
        ]
        for metric in important_metrics:
            print(f"{metric}: {stats[metric]:.4f}")

        # 计算并显示原始资产的买入持有收益
        print("\n=== Buy & Hold Returns ===")
        for symbol in self.symbols:
            first_price = self.prices[symbol].iloc[0]
            last_price = self.prices[symbol].iloc[-1]
            roi = (last_price / first_price - 1) * 100
            print(f"{symbol}: {roi:.4f}%")

        # 显示图表
        self.visualize()
        return self.portfolio


if __name__ == "__main__":
    # 创建并运行BTC和ETH的轮动策略
    strategy = RotationStrategy(
        symbols=["BTC-USD", "ETH-USD"],
        start="2021-01-01 UTC",
        end="2025-06-01 UTC",
        lookback_period=20,
    )

    # 执行回测并展示结果
    portfolio = strategy.run_and_visualize()
