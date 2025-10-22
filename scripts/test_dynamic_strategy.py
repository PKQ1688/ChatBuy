#!/usr/bin/env python3
"""测试动态策略生成器."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel

from chatbuy.backtest.engine import BacktestEngine
from chatbuy.data.fetcher import DataFetcher
from chatbuy.nlp.strategy_parser import StrategyConfig, StrategyParser
from chatbuy.strategies.strategy_factory import StrategyFactory

console = Console()


def test_dynamic_strategy():
    """测试动态策略功能."""
    console.print(Panel("测试动态策略生成器", title="开始测试", style="bold blue"))

    # 测试用例
    test_cases = [
        "5日均线上穿20日均线时买入，下穿时卖出",
        "RSI低于30买入，高于70卖出",
        "MACD金叉买入，死叉卖出",
        "当10日均线上穿30日均线时买入，反之下穿时卖出",
    ]

    parser = StrategyParser()
    factory = StrategyFactory()
    data_fetcher = DataFetcher()
    backtest_engine = BacktestEngine()

    for i, test_case in enumerate(test_cases, 1):
        console.print(f"\n[bold]测试用例 {i}:[/bold] {test_case}")

        try:
            # 1. 解析策略
            console.print("  📝 正在解析策略...")
            parsed_result: StrategyConfig | None = parser.parse(test_case)

            if not parsed_result:
                console.print("  ❌ 策略解析失败")
                continue

            console.print(f"  ✅ 策略解析成功: {parsed_result['strategy_type']}")
            console.print(f"  📊 置信度: {parsed_result['confidence']:.2f}")

            # 2. 创建策略
            console.print("  🔧 正在创建策略...")
            strategy = factory.create_strategy(
                parsed_result["strategy_type"], parsed_result["parameters"]
            )

            if not strategy:
                console.print("  ❌ 策略创建失败")
                continue

            console.print("  ✅ 策略创建成功")

            # 3. 获取数据
            console.print("  📈 正在获取数据...")
            data = data_fetcher.fetch_yfinance("AAPL", "2023-01-01", "2024-12-31")

            if data is None or data.empty:
                console.print("  ❌ 数据获取失败")
                continue

            console.print("  ✅ 数据获取成功")

            # 4. 执行回测
            console.print("  🔄 正在执行回测...")
            results = backtest_engine.run_strategy_backtest(strategy, data)

            if results:
                console.print("  ✅ 回测执行成功")

                # 显示关键指标
                stats = results.get("statistics", {})
                console.print(f"  📊 总收益: {stats.get('total_return', 0):.2%}")
                console.print(f"  📊 年化收益: {stats.get('annualized_return', 0):.2%}")
                console.print(f"  📊 最大回撤: {stats.get('max_drawdown', 0):.2%}")
                console.print(f"  📊 夏普比率: {stats.get('sharpe_ratio', 0):.2f}")

                # 显示策略详情
                strategy_info = strategy.get_info()
                if "strategy_description" in strategy_info:
                    console.print(
                        f"  📝 策略描述: {strategy_info['strategy_description']}"
                    )

            else:
                console.print("  ❌ 回测执行失败")

        except Exception as e:
            console.print(f"  ❌ 测试过程中发生错误: {str(e)}")
            import traceback

            console.print(traceback.format_exc())

    console.print(Panel("测试完成", style="bold green"))


if __name__ == "__main__":
    test_dynamic_strategy()
