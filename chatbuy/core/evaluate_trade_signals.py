import os

import pandas as pd

from chatbuy.logger import log


def evaluate_signals(signals_path: str, prices_path: str) -> dict:
    """Simulate trading and evaluate performance based on signal and price files.

    Args:
        signals_path (str): Path to the trading signal file (CSV, must contain 'action' and 'trade_time'/'timestamp' columns).
        prices_path (str): Path to the price data file (CSV, must contain 'timestamp' and 'close' columns).

    Returns:
        dict: Dictionary containing evaluation results, for example:
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
              If file reading or processing fails, success is False and error contains the message.
    """
    try:
        # --- Data loading and preparation ---
        if not os.path.exists(signals_path):
            log.error(f"Signal file not found: {signals_path}")
            return {"success": False, "error": f"Signal file not found: {signals_path}"}
        if not os.path.exists(prices_path):
            log.error(f"Price file not found: {prices_path}")
            return {"success": False, "error": f"Price file not found: {prices_path}"}

        signals = pd.read_csv(signals_path)
        # Compatible with old column name 'trade_time'
        if "trade_time" in signals.columns and "timestamp" not in signals.columns:
            signals = signals.rename(columns={"trade_time": "timestamp"})
        if "timestamp" not in signals.columns or "action" not in signals.columns:
            log.error("Signal file missing 'timestamp' or 'action' column.")
            return {
                "success": False,
                "error": "Signal file missing 'timestamp' or 'action' column.",
            }

        prices = pd.read_csv(prices_path)
        if "timestamp" not in prices.columns or "close" not in prices.columns:
            log.error("Price file missing 'timestamp' or 'close' column.")
            return {
                "success": False,
                "error": "Price file missing 'timestamp' or 'close' column.",
            }

        # Merge signals and prices
        df = pd.merge(
            signals, prices[["timestamp", "close"]], on="timestamp", how="inner"
        )
        if df.empty:
            log.error("No common timestamps between signals and prices, cannot merge.")
            return {
                "success": False,
                "error": "No common timestamps between signals and prices, cannot merge.",
            }

        # --- Simulate trading ---
        position = 0
        buy_price = 0
        trades = []
        trade_details = []

        for idx, row in df.iterrows():
            if row["action"] == "buy" and position == 0:
                buy_price = row["close"]
                position = 1
                trade_details.append((row["timestamp"], "buy", float(buy_price)))
            elif row["action"] == "sell" and position == 1:
                sell_price = row["close"]
                profit = sell_price - buy_price
                trades.append(profit)
                trade_details.append((row["timestamp"], "sell", float(sell_price)))
                position = 0

        # If there is still a position at the end, force sell at the last candle
        if position == 1 and not df.empty:
            last_row = df.iloc[-1]
            sell_price = last_row["close"]
            profit = sell_price - buy_price
            trades.append(profit)
            trade_details.append((last_row["timestamp"], "sell", float(sell_price)))
            position = 0

        # --- Statistics ---
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
        log.error("Error occurred during signal evaluation", exc_info=True)
        return {
            "success": False,
            "error": f"Error occurred during signal evaluation: {e}",
        }


if __name__ == "__main__":
    signals_file = "output/trade_advice_unified_results_one.csv"
    prices_file = "data/BTC_USDT_1d_with_indicators.csv"

    results = evaluate_signals(signals_file, prices_file)

    if results["success"]:
        log.info(f"Total trades: {results['total_trades']}")
        log.info(f"Total profit: {results['total_profit']:.2f}")
        log.info(f"Win rate: {results['win_rate']:.2%}")
        log.info(f"Average profit per trade: {results['avg_profit']:.2f}")
        log.info("\nTrade details:")
        for detail in results["trade_details"]:
            log.info(detail)
    else:
        log.error(f"Evaluation failed: {results['error']}")
