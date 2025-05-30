import pandas as pd
import vectorbt as vbt


class RotationStrategy:
    """资产轮动策略：基于历史表现在不同资产间进行选择.

    该策略基于BTC和ETH在过去N天的表现进行切换，只投资于表现较好且回报为正的加密货币.
    若两者回报均为负，则持有现金.
    """

    def __init__(
        self,
        symbols=["BTC-USD", "ETH-USD"],
        start="2022-01-01 UTC",
        end="2024-01-01 UTC",
        lookback_period=20,
        init_cash=10000.0,
        fees=0.001,
        slippage=0.0025,
    ):
        """初始化轮动策略.

        参数:
            symbols: 轮动资产列表（默认：BTC-USD和ETH-USD）
            start: 回测起始日期
            end: 回测结束日期
            lookback_period: 计算表现的历史周期（天数）
            init_cash: 初始资金
            fees: 交易费用率
            slippage: 滑点
        """
        self.symbols = symbols
        self.start = start
        self.end = end
        self.lookback_period = lookback_period
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage
        self.portfolio = None
        self.buy_hold_portfolios = {}

        # 获取数据并生成信号
        self._prepare_data()
        self._generate_signals()

    def _prepare_data(self):
        """获取并处理价格数据."""
        # 下载价格数据
        self.data = vbt.YFData.download(self.symbols, start=self.start, end=self.end)
        self.prices = self.data.get("Close")

        # 处理缺失值
        if self.prices.isna().any().any():
            self.prices = self.prices.dropna(how="any")

    def _generate_signals(self):
        """生成轮动策略的交易信号."""
        # 计算滚动收益率，填充NaN为0
        returns = self.prices.pct_change(self.lookback_period).fillna(0)

        # 找到每日表现最好的资产（使用前一天的数据）
        best_asset_idx = returns.idxmax(axis=1)
        best_returns = returns.max(axis=1)

        # 生成目标权重矩阵（每次只持有一个资产或空仓）
        target_weights = pd.DataFrame(
            0.0, index=self.prices.index, columns=self.symbols
        )

        # 从lookback_period+1开始设置权重，避免前瞻偏差
        for i in range(self.lookback_period + 1, len(self.prices)):
            date = self.prices.index[i]
            # 使用前一天的信号来决定今天的仓位
            prev_best_asset = best_asset_idx.iloc[i - 1]
            prev_best_return = best_returns.iloc[i - 1]

            # 明确处理所有情况
            if prev_best_return > 0:  # 有正回报的资产
                target_weights.loc[date, prev_best_asset] = 1.0
            # 其他情况保持0权重（持有现金）

        self.target_weights = target_weights

    def run(self):
        """执行回测."""
        # 计算仓位变化以生成买入/卖出信号
        position_changes = self.target_weights.diff().fillna(self.target_weights)

        # 使用vectorbt的Portfolio.from_signals创建轮动策略组合
        entries = position_changes > 0  # 买入信号
        exits = position_changes < 0  # 卖出信号

        self.portfolio = vbt.Portfolio.from_signals(
            self.prices,
            entries=entries,
            exits=exits,
            size=1.0,  # 全仓操作
            size_type="percent",
            fees=self.fees,
            slippage=self.slippage,
            slippage_type="percent",  # 明确滑点类型为百分比
            init_cash=self.init_cash,
            cash_sharing=True,  # 多资产共享现金
            freq="1D",  # 明确设置频率为1天
            stop_loss=0.1,  # 10%止损
            stop_exit_price="close",  # 使用收盘价执行止损
            log=True,  # 启用交易日志
        )

        # 创建买入持有策略作为对比
        for symbol in self.symbols:
            self.buy_hold_portfolios[symbol] = vbt.Portfolio.from_holding(
                self.prices[symbol],
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                freq="1D",  # 明确设置频率
            )

        return self.portfolio

    def visualize(self):
        """使用vectorbt原生功能可视化投资组合表现."""
        if self.portfolio is None:
            self.run()

        # 使用vectorbt原生的绘图功能
        # 1. 绘制投资组合价值曲线（去掉不支持分组数据的图表）
        fig = self.portfolio.plot(subplots=["cum_returns", "drawdowns"])
        fig.show()

        # 2. 绘制与买入持有策略的对比
        all_values = pd.DataFrame()
        all_values["Rotation Strategy"] = self.portfolio.value()

        for symbol in self.symbols:
            all_values[f"{symbol} Buy&Hold"] = self.buy_hold_portfolios[symbol].value()

        # 使用vectorbt绘制价值曲线对比
        fig_comp = all_values.vbt.plot(title="Strategy vs Buy & Hold Value Comparison")
        fig_comp.show()

    def get_stats(self):
        """返回投资组合统计数据."""
        if self.portfolio is None:
            self.run()

        # 使用vectorbt原生统计功能
        stats = {}
        stats["rotation"] = self.portfolio.stats()

        # 添加买入持有策略统计
        for symbol in self.symbols:
            stats[f"{symbol}_buyhold"] = self.buy_hold_portfolios[symbol].stats()

        return stats

    def print_stats(self):
        """打印关键统计数据."""
        stats = self.get_stats()

        print("\n=== Rotation Strategy Results ===")
        rotation_stats = stats["rotation"]

        # 安全获取统计指标，处理可能的键名变化
        total_return = rotation_stats.get(
            "Total Return [%]", rotation_stats.get("Total Return", "N/A")
        )
        max_dd = rotation_stats.get(
            "Max Drawdown [%]", rotation_stats.get("Max Drawdown", "N/A")
        )
        sharpe = rotation_stats.get("Sharpe Ratio", rotation_stats.get("Sharpe", "N/A"))
        win_rate = rotation_stats.get(
            "Win Rate [%]", rotation_stats.get("Win Rate", "N/A")
        )

        print(f"Total Return: {total_return}")
        print(f"Max Drawdown: {max_dd}")
        print(f"Sharpe Ratio: {sharpe}")
        print(f"Win Rate: {win_rate}")

        print("\n=== Buy & Hold Comparison ===")
        for symbol in self.symbols:
            bh_stats = stats[f"{symbol}_buyhold"]
            bh_total_return = bh_stats.get(
                "Total Return [%]", bh_stats.get("Total Return", "N/A")
            )
            bh_max_dd = bh_stats.get(
                "Max Drawdown [%]", bh_stats.get("Max Drawdown", "N/A")
            )
            bh_sharpe = bh_stats.get("Sharpe Ratio", bh_stats.get("Sharpe", "N/A"))

            print(
                f"{symbol} - Total Return: {bh_total_return}, Max DD: {bh_max_dd}, Sharpe: {bh_sharpe}"
            )

    def run_and_visualize(self):
        """运行策略并可视化结果."""
        self.run()
        self.print_stats()
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
