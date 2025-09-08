"""
BTC和ETH轮动策略示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotation_strategy import BTCETHRotationStrategy
import pandas as pd


def main():
    """运行BTC和ETH轮动策略示例"""
    
    print("=== BTC和ETH轮动策略回测 ===\n")
    
    # 创建策略实例
    strategy = BTCETHRotationStrategy(
        initial_balance=10000.0,      # 初始资金10000 USDT
        lookback_period=20,           # 技术指标回望期
        momentum_period=10,           # 动量计算期
        rsi_period=14,               # RSI计算期
        rebalance_frequency="weekly"  # 每周再平衡
    )
    
    try:
        # 运行回测（最近1年数据）
        print("开始回测...")
        results = strategy.backtest(days=365)
        
        # 打印详细结果
        print("\n" + "="*50)
        print("回测结果摘要")
        print("="*50)
        
        strategy_perf = results['strategy_performance']
        benchmark_perf = results['benchmark_performance']
        
        print(f"初始资金: ${strategy_perf['initial_balance']:,.2f}")
        print(f"最终价值: ${strategy_perf['final_value']:,.2f}")
        print(f"策略总收益率: {strategy_perf['total_return_pct']:.2f}%")
        print(f"基准收益率 (50% BTC + 50% ETH): {benchmark_perf['total_return_pct']:.2f}%")
        print(f"超额收益: {strategy_perf['total_return_pct'] - benchmark_perf['total_return_pct']:.2f}%")
        print(f"最大回撤: {strategy_perf['max_drawdown_pct']:.2f}%")
        print(f"夏普比率: {strategy_perf['sharpe_ratio']:.3f}")
        print(f"总交易次数: {strategy_perf['total_trades']}")
        print(f"交易天数: {strategy_perf['trading_days']}")
        
        # 打印信号分析
        print("\n" + "="*50)
        print("信号分析")
        print("="*50)
        
        signal_analysis = strategy.get_signal_analysis()
        if not signal_analysis.empty:
            print(signal_analysis.round(3))
        
        # 打印最近的交易记录
        print("\n" + "="*50)
        print("最近10笔交易")
        print("="*50)
        
        trade_history = results['trade_history']
        if not trade_history.empty:
            recent_trades = trade_history.tail(10)
            for _, trade in recent_trades.iterrows():
                print(f"{trade['timestamp'].strftime('%Y-%m-%d')}: "
                      f"{trade['action']} {trade['symbol']} "
                      f"{trade['quantity']:.6f} @ ${trade['price']:.2f} "
                      f"(金额: ${trade['amount']:.2f})")
        
        # 保存结果到文件
        print("\n保存结果到文件...")
        
        # 保存投资组合历史
        portfolio_history = results['portfolio_history']
        portfolio_history.to_csv('data/rotation_portfolio_history.csv', index=False)
        print("投资组合历史已保存到: data/rotation_portfolio_history.csv")
        
        # 保存交易历史
        if not trade_history.empty:
            trade_history.to_csv('data/rotation_trade_history.csv', index=False)
            print("交易历史已保存到: data/rotation_trade_history.csv")
        
        # 保存信号历史
        signals_df = results['signals']
        signals_df.to_csv('data/rotation_signals.csv')
        print("信号历史已保存到: data/rotation_signals.csv")
        
        # 绘制结果图表
        print("\n绘制结果图表...")
        strategy.plot_results('data/rotation_strategy_results.png')
        
        # 打印当前投资组合状态
        print("\n" + "="*50)
        print("当前投资组合状态")
        print("="*50)
        
        # 获取最新价格
        latest_prices = {
            "BTC": strategy.data["BTC"].iloc[-1]["Close"],
            "ETH": strategy.data["ETH"].iloc[-1]["Close"]
        }
        
        strategy.portfolio.print_status(latest_prices)
        
    except Exception as e:
        print(f"运行策略时出错: {e}")
        import traceback
        traceback.print_exc()


def analyze_signals():
    """分析轮动信号的有效性"""
    
    print("\n=== 信号分析 ===")
    
    strategy = BTCETHRotationStrategy()
    
    # 获取数据
    data = strategy.data_manager.get_multiple_data(["BTC", "ETH"], "1d", 365)
    
    if len(data) != 2:
        print("无法获取完整数据")
        return
    
    # 计算技术指标
    for symbol in ["BTC", "ETH"]:
        data[symbol] = strategy.calculate_technical_indicators(data[symbol])
    
    # 计算信号
    signals = strategy.calculate_rotation_signals(data["BTC"], data["ETH"])
    
    # 分析信号统计
    signal_df = pd.DataFrame(signals)
    
    print("各信号选择BTC的比例:")
    for col in signal_df.columns:
        btc_ratio = (signal_df[col] == 'BTC').mean()
        print(f"{col}: {btc_ratio:.2%}")
    
    print("\n信号变化频率:")
    for col in signal_df.columns:
        changes = (signal_df[col] != signal_df[col].shift(1)).sum()
        print(f"{col}: {changes} 次变化")
    
    # 保存信号分析
    signal_df.to_csv('data/signal_analysis.csv')
    print("\n信号分析已保存到: data/signal_analysis.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BTC和ETH轮动策略")
    parser.add_argument("--mode", choices=["backtest", "analyze"], 
                       default="backtest", help="运行模式")
    
    args = parser.parse_args()
    
    if args.mode == "backtest":
        main()
    elif args.mode == "analyze":
        analyze_signals()