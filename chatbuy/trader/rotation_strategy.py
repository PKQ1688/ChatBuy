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
        """根据轮动策略初始化买入卖出信号.

        策略比较过去周期的回报率，投资于表现最佳的资产（若回报为正）；
        若所有资产回报均为负，则持有现金.
        """
        # 计算回报率
        returns_df = self.calculate_returns()

        # 使用向量化操作找出每天回报率最高的资产
        best_assets = returns_df.idxmax(axis=1)
        best_returns = returns_df.max(axis=1)

        # 跟踪当前持有的资产
        active_symbol = None
        trading_dates = returns_df.index[self.lookback_period :]

        for date in trading_dates:
            best_symbol = best_assets[date]
            best_return = best_returns[date]

            # 更新信号
            if best_return > 0:  # 只在正回报时投资
                if active_symbol and active_symbol != best_symbol:
                    # 切换资产: 卖出旧资产，买入新资产
                    self.exits[active_symbol].loc[date] = True
                    self.entries[best_symbol].loc[date] = True
                    active_symbol = best_symbol
                elif not active_symbol:
                    # 初始买入
                    self.entries[best_symbol].loc[date] = True
                    active_symbol = best_symbol
            elif active_symbol:
                # 所有资产回报为负: 清仓
                self.exits[active_symbol].loc[date] = True
                active_symbol = None

    def run(self):
        """执行回测并返回投资组合."""
        # 为每个资产创建独立的投资组合
        portfolios = {}

        for symbol in self.symbols:
            # 确保信号是布尔数组
            entries_arr = self.entries[symbol].values.astype(bool)
            exits_arr = self.exits[symbol].values.astype(bool)

            # 创建单资产投资组合
            pf = vbt.Portfolio.from_signals(
                self.prices[symbol],
                entries=entries_arr,
                exits=exits_arr,
                init_cash=vbt.settings.portfolio["init_cash"],
                fees=vbt.settings.portfolio["fees"],
                slippage=vbt.settings.portfolio["slippage"],
            )
            portfolios[symbol] = pf

        # 使用第一个资产的投资组合作为主投资组合
        self.portfolio = portfolios[self.symbols[0]]
        self.all_portfolios = portfolios

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
        drawdown = self.portfolio.drawdown()
        axes[1].fill_between(
            drawdown.index, 0, -drawdown.values * 100, color="r", alpha=0.5
        )
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

        return self.portfolio.stats()

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
