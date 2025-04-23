import os

import pandas as pd

from chatbuy.logger import log  # 导入日志记录器


def evaluate_signals(signals_path: str, prices_path: str) -> dict:
    """根据信号文件和价格文件模拟交易并评估表现.

    Args:
        signals_path (str): 交易信号文件的路径 (CSV, 需要包含 'action' 和 'trade_time'/'timestamp' 列)。
        prices_path (str): 价格数据文件的路径 (CSV, 需要包含 'timestamp' 和 'close' 列)。

    Returns:
        dict: 包含评估结果的字典，例如:
              {
                  "success": bool,
                  "error": str | None,
                  "total_trades": int,
                  "total_profit": float,
                  "win_count": int,
                  "win_rate": float,
                  "avg_profit": float,
                  "trade_details": list[tuple] # [(timestamp, action, price), ...]
              }
              如果文件读取或处理失败，success 为 False 并包含 error 信息。
    """
    try:
        # --- 数据加载和准备 ---
        if not os.path.exists(signals_path):
            log.error(f"信号文件未找到: {signals_path}")
            return {"success": False, "error": f"信号文件未找到: {signals_path}"}
        if not os.path.exists(prices_path):
            log.error(f"价格文件未找到: {prices_path}")
            return {"success": False, "error": f"价格文件未找到: {prices_path}"}

        signals = pd.read_csv(signals_path)
        # 兼容旧列名 'trade_time'
        if "trade_time" in signals.columns and "timestamp" not in signals.columns:
            signals = signals.rename(columns={"trade_time": "timestamp"})
        if "timestamp" not in signals.columns or "action" not in signals.columns:
            log.error("信号文件缺少 'timestamp' 或 'action' 列。")
            return {
                "success": False,
                "error": "信号文件缺少 'timestamp' 或 'action' 列。",
            }

        prices = pd.read_csv(prices_path)
        if "timestamp" not in prices.columns or "close" not in prices.columns:
            log.error("价格文件缺少 'timestamp' 或 'close' 列。")
            return {
                "success": False,
                "error": "价格文件缺少 'timestamp' 或 'close' 列。",
            }

        # 合并信号和价格
        df = pd.merge(
            signals, prices[["timestamp", "close"]], on="timestamp", how="inner"
        )
        if df.empty:
            log.error("信号和价格数据没有共同的时间戳，无法合并。")
            return {
                "success": False,
                "error": "信号和价格数据没有共同的时间戳，无法合并。",
            }

        # --- 模拟交易 ---
        position = 0
        buy_price = 0
        trades = []
        trade_details = []

        for idx, row in df.iterrows():
            if row["action"] == "buy" and position == 0:
                buy_price = row["close"]
                position = 1
                trade_details.append(
                    (row["timestamp"], "buy", float(buy_price))
                )
            elif row["action"] == "sell" and position == 1:
                sell_price = row["close"]
                profit = sell_price - buy_price
                trades.append(profit)
                trade_details.append(
                    (row["timestamp"], "sell", float(sell_price))
                )
                position = 0

        # 如果最后还有持仓，强制在最后一根K线卖出
        if position == 1 and not df.empty:
            last_row = df.iloc[-1]
            sell_price = last_row["close"]
            profit = sell_price - buy_price
            trades.append(profit)
            trade_details.append(
                (last_row["timestamp"], "sell", float(sell_price))
            )
            position = 0

        # --- 统计结果 ---
        total_trades = len(trades)
        total_profit = sum(trades)
        win_count = sum(1 for t in trades if t > 0)
        win_rate = (win_count / total_trades) if total_trades > 0 else 0
        avg_profit = (total_profit / total_trades) if total_trades > 0 else 0

        return {
            "success": True,
            "error": None,
            "total_trades": total_trades,
            "total_profit": total_profit,
            "win_count": win_count,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "trade_details": trade_details,
        }

    except Exception as e:
        log.error("评估信号时出错", exc_info=True)
        return {"success": False, "error": f"评估信号时出错: {e}"}


# --- 保留原始脚本的 __main__ 部分，用于独立测试 ---
if __name__ == "__main__":
    signals_file = "output/trade_advice_unified_results_one.csv"
    prices_file = "data/BTC_USDT_1d_with_indicators.csv"

    results = evaluate_signals(signals_file, prices_file)

    if results["success"]:
        log.info(f"总交易次数: {results['total_trades']}")
        log.info(f"总收益: {results['total_profit']:.2f}")
        log.info(f"胜率: {results['win_rate']:.2%}")
        log.info(f"平均单笔收益: {results['avg_profit']:.2f}")
        log.info("\n每笔交易明细:")
        for detail in results["trade_details"]:
            log.info(detail)
    else:
        log.error(f"评估失败: {results['error']}")
